import json
import os
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests


BASE_URL = "https://intervals.icu/api/v1"
AUTH_USER = "API_KEY"
TZ = ZoneInfo("Australia/Sydney")

DATA_DIR = Path("data/latest")
DATA_DIR.mkdir(parents=True, exist_ok=True)


RUN_TYPES = {"Run", "VirtualRun", "TrailRun", "TreadmillRun", "Treadmill", "VirtualRideRun"}
RIDE_TYPES = {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide"}
PRIMARY_TRAINING_TYPES = RUN_TYPES | RIDE_TYPES | {"Workout"}
NON_PRIMARY_TYPES = {"Walk", "Hike", "Breathwork", "Yoga", "Meditation"}


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_KEY = require_env("INTERVALS_API_KEY")


def get_text(url: str, params: dict | None = None) -> str:
    r = requests.get(
        url,
        auth=(AUTH_USER, API_KEY),
        params=params,
        timeout=60,
    )
    r.raise_for_status()
    return r.text


def read_csv_text(text: str) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


def safe(v):
    if pd.isna(v):
        return None
    try:
        return v.item()
    except Exception:
        return v


def pick(row, *names):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return safe(row[name])
    return None


def to_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def iso_date_only(value):
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date().isoformat()
    except Exception:
        return None


def load_data():
    now = datetime.now(TZ)
    today = now.date()

    wellness_oldest = today - timedelta(days=28)
    wellness_newest = today + timedelta(days=1)

    activities_oldest = today - timedelta(days=14)
    activities_newest = today + timedelta(days=1)

    wellness_cols = ",".join([
        "weight",
        "restingHR",
        "hrv",
        "hrvSDNN",
        "readiness",
        "sleepSecs",
        "sleepScore",
        "sleepQuality",
        "avgSleepingHR",
        "spO2",
        "soreness",
        "fatigue",
        "stress",
        "mood",
        "motivation",
        "injury",
        "ctl",
        "atl",
        "rampRate",
        "ctlLoad",
        "atlLoad",
        "Ride_eftp",
        "Run_eftp",
    ])

    wellness_url = f"{BASE_URL}/athlete/0/wellness.csv"
    activities_url = f"{BASE_URL}/athlete/0/activities.csv"

    wellness_text = get_text(
        wellness_url,
        params={
            "oldest": wellness_oldest.isoformat(),
            "newest": wellness_newest.isoformat(),
            "cols": wellness_cols,
        },
    )

    activities_text = get_text(
        activities_url,
        params={
            "oldest": activities_oldest.isoformat(),
            "newest": activities_newest.isoformat(),
        },
    )

    (DATA_DIR / "wellness.csv").write_text(wellness_text, encoding="utf-8")
    (DATA_DIR / "activities.csv").write_text(activities_text, encoding="utf-8")

    wellness_df = read_csv_text(wellness_text)
    activities_df = read_csv_text(activities_text)

    (DATA_DIR / "wellness_columns.json").write_text(
        json.dumps(list(wellness_df.columns), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DATA_DIR / "activities_columns.json").write_text(
        json.dumps(list(activities_df.columns), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return wellness_df, activities_df, now


def normalize_wellness(df: pd.DataFrame, today_date: datetime.date) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        df = df[df["date"].dt.date <= today_date]
        df = df.sort_values("date")
    return df


def summarize_recovery_trends(df: pd.DataFrame, today_date):
    if df.empty:
        return {}
    start_date = today_date - timedelta(days=6)
    recent = df[df["date"].dt.date >= start_date].copy()
    if recent.empty:
        return {}

    out = {
        "window_start": start_date.isoformat(),
        "window_end": today_date.isoformat(),
        "days_with_recovery_data": int(len(recent)),
    }

    if "sleepSecs" in recent.columns:
        sleep_hours = recent["sleepSecs"] / 3600
        non_missing = sleep_hours.dropna()
        out["avg_sleep_hours"] = round(float(non_missing.mean()), 2) if not non_missing.empty else None
        out["min_sleep_hours"] = round(float(non_missing.min()), 2) if not non_missing.empty else None

    for src, dst in [
        ("sleepScore", "avg_sleep_score"),
        ("hrv", "avg_hrv"),
        ("restingHR", "avg_resting_hr"),
        ("avgSleepingHR", "avg_sleeping_hr"),
    ]:
        if src in recent.columns:
            s = recent[src].dropna()
            out[dst] = round(float(s.mean()), 2) if not s.empty else None

    if "sleepScore" in recent.columns:
        s = recent["sleepScore"].dropna()
        out["low_sleep_score_days"] = int((s < 70).sum()) if not s.empty else 0

    return out


def prepare_wellness(df: pd.DataFrame, today_date):
    df = normalize_wellness(df, today_date)
    if df.empty:
        return {}, {}, {}, pd.DataFrame(), {}

    latest_state = df.iloc[-1]

    recovery_fields = [
        "restingHR",
        "hrv",
        "sleepSecs",
        "sleepScore",
        "sleepQuality",
        "avgSleepingHR",
        "spO2",
        "readiness",
        "fatigue",
        "soreness",
        "stress",
    ]
    recovery_df = df.dropna(subset=recovery_fields, how="all")
    latest_recovery = recovery_df.iloc[-1] if not recovery_df.empty else latest_state

    recovery = {
        "date": safe(latest_recovery.get("date")),
        "weight": pick(latest_recovery, "weight"),
        "resting_hr": pick(latest_recovery, "restingHR"),
        "hrv": pick(latest_recovery, "hrv"),
        "hrv_sdnn": pick(latest_recovery, "hrvSDNN"),
        "readiness": pick(latest_recovery, "readiness"),
        "sleep_secs": pick(latest_recovery, "sleepSecs"),
        "sleep_hours": round(float(latest_recovery["sleepSecs"]) / 3600, 2)
        if "sleepSecs" in latest_recovery.index and pd.notna(latest_recovery["sleepSecs"]) else None,
        "sleep_score": pick(latest_recovery, "sleepScore"),
        "sleep_quality": pick(latest_recovery, "sleepQuality"),
        "avg_sleeping_hr": pick(latest_recovery, "avgSleepingHR"),
        "spo2": pick(latest_recovery, "spO2"),
        "fatigue": pick(latest_recovery, "fatigue"),
        "soreness": pick(latest_recovery, "soreness"),
        "stress": pick(latest_recovery, "stress"),
        "mood": pick(latest_recovery, "mood"),
        "motivation": pick(latest_recovery, "motivation"),
        "injury": pick(latest_recovery, "injury"),
    }

    training_state = {
        "date": safe(latest_state.get("date")),
        "ctl": pick(latest_state, "ctl"),
        "atl": pick(latest_state, "atl"),
        "ramp_rate": pick(latest_state, "rampRate"),
        "ctl_load": pick(latest_state, "ctlLoad"),
        "atl_load": pick(latest_state, "atlLoad"),
        "ride_eftp": pick(latest_state, "Ride_eftp"),
        "run_eftp": pick(latest_state, "Run_eftp"),
        "atl_ctl_ratio": round(float(latest_state["atl"]) / float(latest_state["ctl"]), 2)
        if "atl" in latest_state.index and pd.notna(latest_state["atl"]) and "ctl" in latest_state.index and pd.notna(latest_state["ctl"]) and float(latest_state["ctl"]) != 0 else None,
    }

    recovery_trend = summarize_recovery_trends(recovery_df, today_date)
    data_status = {
        "today_date": today_date.isoformat(),
        "recovery_date": iso_date_only(recovery.get("date")),
        "training_state_date": iso_date_only(training_state.get("date")),
        "recovery_is_fresh": iso_date_only(recovery.get("date")) == today_date.isoformat(),
        "training_state_is_fresh": iso_date_only(training_state.get("date")) == today_date.isoformat(),
    }

    return recovery, training_state, recovery_trend, recovery_df, data_status


def normalize_activities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "start_date_local" in df.columns:
        df["start_date_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")
        df = df[df["start_date_local"].notna()]
        df = df.sort_values("start_date_local")
    return df


def classify_activity(row) -> dict:
    activity_type = pick(row, "type")
    load = pick(row, "icu_training_load", "training_load")
    avg_hr = pick(row, "average_heartrate")
    moving_time = pick(row, "moving_time")
    distance = pick(row, "distance")
    avg_speed = pick(row, "average_speed")
    threshold_pace = pick(row, "threshold_pace")

    distance_km = float(distance) / 1000 if distance is not None else None
    duration_min = float(moving_time) / 60 if moving_time is not None else None
    speed_ratio = None
    if avg_speed is not None and threshold_pace is not None and float(threshold_pace) > 0:
        speed_ratio = float(avg_speed) / float(threshold_pace)

    label = "other_activity"
    reason = []

    if activity_type in RUN_TYPES:
        if load is not None and float(load) <= 20 and (avg_hr is None or float(avg_hr) < 150) and (speed_ratio is None or speed_ratio < 0.82):
            label = "recovery_run"
            reason.append("负荷较轻，心率不高，速度明显低于阈值配速")
        elif (load is not None and float(load) >= 55) or (avg_hr is not None and float(avg_hr) >= 160) or (speed_ratio is not None and speed_ratio >= 0.88):
            label = "moderate_to_hard_run"
            reason.append("心率/负荷/配速至少一项偏高，不应视作恢复跑")
        else:
            label = "easy_aerobic_run"
            reason.append("整体更像常规有氧，而不是恢复跑")

        if duration_min is not None and duration_min < 28 and avg_hr is not None and float(avg_hr) >= 152:
            if label == "recovery_run":
                label = "moderate_to_hard_run"
            reason.append("距离虽短但平均心率不低")

    elif activity_type == "Workout":
        label = "strength_or_workout"
        reason.append("非跑步活动，需要结合主观疲劳解释")
    elif activity_type in RIDE_TYPES:
        label = "ride"
        reason.append("骑行活动，按骑行负荷解释")
    else:
        reason.append("非核心跑步类型，暂按其他活动处理")

    return {
        "label": label,
        "speed_vs_threshold_ratio": round(speed_ratio, 3) if speed_ratio is not None else None,
        "reason": "；".join(reason) if reason else None,
        "distance_km": round(distance_km, 2) if distance_km is not None else None,
        "duration_min": round(duration_min, 1) if duration_min is not None else None,
    }


def choose_primary_activity(candidates: pd.DataFrame):
    if candidates.empty:
        return None, {}

    work = candidates.copy()
    work["type_value"] = work["type"].fillna("") if "type" in work.columns else ""
    work["training_load_num"] = pd.to_numeric(
        work["icu_training_load"] if "icu_training_load" in work.columns else work.get("training_load"),
        errors="coerce",
    ).fillna(-1)
    work["moving_time_num"] = pd.to_numeric(work["moving_time"] if "moving_time" in work.columns else None, errors="coerce").fillna(-1)

    work["is_primary_type"] = work["type_value"].isin(PRIMARY_TRAINING_TYPES).astype(int)
    work["is_run_type"] = work["type_value"].isin(RUN_TYPES).astype(int)
    work["is_non_primary_type"] = work["type_value"].isin(NON_PRIMARY_TYPES).astype(int)

    chosen = work.sort_values(
        ["is_primary_type", "is_run_type", "training_load_num", "moving_time_num", "start_date_local"],
        ascending=[True, True, True, True, True],
    ).iloc[-1]

    meta = {
        "selection_mode": "primary_activity_of_day",
        "candidate_count": int(len(work)),
        "candidate_types": sorted([str(x) for x in work["type_value"].dropna().unique().tolist()]),
        "selected_type": safe(chosen.get("type")),
        "selected_training_load": to_float(chosen.get("training_load_num")),
        "selected_moving_time_sec": to_float(chosen.get("moving_time_num")),
        "selection_rule": (
            "先选主训练类型（Run/Workout/Ride），再优先 Run，之后按训练负荷、时长、开始时间排序。"
        ),
    }
    return chosen, meta


def summarize_activity_trends(df: pd.DataFrame, today_date):
    if df.empty:
        return {}
    start_date = today_date - timedelta(days=6)
    recent = df[df["start_date_local"].dt.date >= start_date].copy()
    if recent.empty:
        return {}

    out = {
        "window_start": start_date.isoformat(),
        "window_end": today_date.isoformat(),
        "activity_count": int(len(recent)),
    }

    if "icu_training_load" in recent.columns:
        s = recent["icu_training_load"].dropna()
        out["total_training_load"] = round(float(s.sum()), 1) if not s.empty else None
        out["avg_training_load"] = round(float(s.mean()), 1) if not s.empty else None

    if "type" in recent.columns:
        run_mask = recent["type"].isin(list(RUN_TYPES))
        out["run_count"] = int(run_mask.sum())
        if "distance" in recent.columns:
            d = recent.loc[run_mask, "distance"].dropna()
            out["run_distance_km"] = round(float(d.sum()) / 1000, 2) if not d.empty else 0.0

    return out


def prepare_activity(df: pd.DataFrame, today_date):
    df = normalize_activities(df)
    if df.empty:
        return {}, {}, {}, pd.DataFrame()

    yesterday = today_date - timedelta(days=1)
    chosen = None
    selection_meta = {}

    if "start_date_local" in df.columns:
        y_df = df[df["start_date_local"].dt.date == yesterday].copy()
        if not y_df.empty:
            chosen, selection_meta = choose_primary_activity(y_df)
            selection_meta["target_date"] = yesterday.isoformat()

    if chosen is None:
        chosen = df.iloc[-1]
        selection_meta = {
            "selection_mode": "latest_activity_fallback",
            "candidate_count": 1,
            "target_date": yesterday.isoformat(),
            "selection_rule": "昨天没有活动，回退到最近一条活动。",
        }

    classification = classify_activity(chosen)

    activity = {
        "id": pick(chosen, "id"),
        "start_date_local": safe(chosen.get("start_date_local")),
        "name": pick(chosen, "name"),
        "type": pick(chosen, "type"),
        "moving_time_sec": pick(chosen, "moving_time"),
        "elapsed_time_sec": pick(chosen, "elapsed_time"),
        "distance_m": pick(chosen, "distance"),
        "distance_km": round(float(chosen["distance"]) / 1000, 2)
        if "distance" in chosen.index and pd.notna(chosen["distance"]) else None,
        "average_heartrate": pick(chosen, "average_heartrate"),
        "max_heartrate": pick(chosen, "max_heartrate"),
        "average_speed": pick(chosen, "average_speed"),
        "max_speed": pick(chosen, "max_speed"),
        "total_elevation_gain": pick(chosen, "total_elevation_gain"),
        "calories": pick(chosen, "calories"),
        "average_cadence": pick(chosen, "average_cadence"),
        "pace": pick(chosen, "pace"),
        "threshold_pace": pick(chosen, "threshold_pace"),
        "icu_training_load": pick(chosen, "icu_training_load", "training_load"),
        "icu_intensity": pick(chosen, "icu_intensity"),
        "icu_fitness": pick(chosen, "icu_fitness"),
        "icu_fatigue": pick(chosen, "icu_fatigue"),
        "icu_eftp": pick(chosen, "icu_eftp"),
        "icu_average_watts": pick(chosen, "icu_average_watts", "average_watts"),
        "icu_normalized_watts": pick(chosen, "icu_normalized_watts", "normalized_watts"),
        "selection_meta": selection_meta,
        "classification": classification,
    }

    recent_7d = summarize_activity_trends(df, today_date)
    return activity, recent_7d, selection_meta, df


def build_health_signals(recovery, training_state, recovery_trend, activity, data_status):
    signals = []

    if not data_status.get("recovery_is_fresh"):
        signals.append("今天晨间恢复数据可能还未同步完成，当前恢复判断可能仍沿用前一日数据")

    sleep_hours = recovery.get("sleep_hours")
    if sleep_hours is not None and sleep_hours < 6:
        signals.append("昨晚睡眠时长偏短，优先按恢复不足处理")

    avg_sleep = recovery_trend.get("avg_sleep_hours")
    if sleep_hours is not None and avg_sleep is not None and sleep_hours < avg_sleep - 0.8:
        signals.append("昨晚睡眠明显低于近7天均值")

    hrv = recovery.get("hrv")
    avg_hrv = recovery_trend.get("avg_hrv")
    if hrv is not None and avg_hrv is not None and hrv < avg_hrv * 0.92:
        signals.append("HRV 低于近7天均值，恢复可能偏弱")

    resting_hr = recovery.get("resting_hr")
    avg_resting = recovery_trend.get("avg_resting_hr")
    if resting_hr is not None and avg_resting is not None and resting_hr >= avg_resting + 3:
        signals.append("静息心率高于近7天均值，需留意疲劳或压力")

    ratio = training_state.get("atl_ctl_ratio")
    if ratio is not None and ratio > 1.2:
        signals.append("ATL/CTL 偏高，近期负荷堆积需要留意")

    spo2 = recovery.get("spo2")
    if spo2 is not None and spo2 < 94:
        signals.append("血氧偏低，需继续观察是否只是单日波动")

    injury = recovery.get("injury")
    if injury is not None:
        try:
            if float(injury) > 0:
                signals.append("有 injury 标记，需优先排除局部问题")
        except Exception:
            pass

    cls = activity.get("classification", {}).get("label")
    if cls == "moderate_to_hard_run":
        signals.append("昨天主训练并非恢复跑，今天不宜机械加码")

    return signals


def build_conclusion_and_suggestion(recovery, training_state, recovery_trend, activity, activity_trend):
    score = 0

    sleep_hours = recovery.get("sleep_hours")
    if sleep_hours is not None:
        if sleep_hours >= 7:
            score += 2
        elif sleep_hours >= 6.2:
            score += 1
        else:
            score -= 2

    sleep_score = recovery.get("sleep_score")
    if sleep_score is not None:
        if sleep_score >= 85:
            score += 2
        elif sleep_score >= 75:
            score += 1
        elif sleep_score < 55:
            score -= 2
        elif sleep_score < 65:
            score -= 1

    hrv = recovery.get("hrv")
    avg_hrv = recovery_trend.get("avg_hrv")
    if hrv is not None:
        if avg_hrv is not None:
            if hrv < avg_hrv * 0.92:
                score -= 2
            elif hrv > avg_hrv * 1.05:
                score += 1
        else:
            score += 1

    resting_hr = recovery.get("resting_hr")
    avg_resting = recovery_trend.get("avg_resting_hr")
    if resting_hr is not None and avg_resting is not None:
        if resting_hr >= avg_resting + 3:
            score -= 1

    for metric_name in ["fatigue", "soreness", "stress"]:
        x = recovery.get(metric_name)
        if x is not None:
            try:
                if float(x) >= 7:
                    score -= 2
                elif float(x) >= 5:
                    score -= 1
            except Exception:
                pass

    ratio = training_state.get("atl_ctl_ratio")
    if ratio is not None:
        if ratio > 1.3:
            score -= 2
        elif ratio > 1.15:
            score -= 1

    activity_class = activity.get("classification", {}).get("label")
    if activity_class == "moderate_to_hard_run":
        score -= 1

    recent_load = activity_trend.get("total_training_load")
    if recent_load is not None and recent_load >= 240:
        score -= 1

    if score >= 3:
        conclusion = "恢复较好，今天可以正常训练；是否加码仍要看主观感觉和当天安排。"
        suggestion = "优先做正常训练。若中午有重腿力量，下午跑步保持轻松；若下午想做质量跑，中午力量只做上肢或核心。"
    elif score >= 1:
        conclusion = "恢复尚可，今天适合常规有氧或中等训练，不建议激进加码。"
        suggestion = "以轻松跑或常规有氧为主。只有在主观状态也不错时，才考虑中等质量训练。"
    elif score >= -1:
        conclusion = "恢复一般，今天更适合轻松有氧或降强度。"
        suggestion = "避免双强度，同一天只保留一个主要刺激；更稳的是轻松跑、上肢力量或低负荷日。"
    else:
        conclusion = "恢复偏弱，今天不适合高强度，优先恢复。"
        suggestion = "更适合休息、走路、拉伸或非常轻的恢复活动。若有不适或疲劳延续，连续观察。"

    return conclusion, suggestion, score


def build_report(now, conclusion, suggestion, recovery, training_state, recovery_trend, activity, activity_trend, signals, data_status):
    lines = [
        "# Daily Coach Report",
        "",
        f"- Generated: {now.isoformat()}",
        f"- Conclusion: {conclusion}",
        f"- Today suggestion: {suggestion}",
        "",
        "## Data Status",
    ]
    for k, v in data_status.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Recovery"]
    for k, v in recovery.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## 7-Day Recovery Trend"]
    for k, v in recovery_trend.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Training State"]
    for k, v in training_state.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Yesterday Primary Activity"]
    for k, v in activity.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## 7-Day Activity Trend"]
    for k, v in activity_trend.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Signals To Watch"]
    if signals:
        for s in signals:
            lines.append(f"- {s}")
    else:
        lines.append("- No strong warning signal from the available data.")

    return "\n".join(lines) + "\n"


def build_payload_and_report(now: datetime):
    wellness_df, activities_df, now = load_data()
    recovery, training_state, recovery_trend, _, data_status = prepare_wellness(wellness_df, now.date())
    activity, activity_trend, _, _ = prepare_activity(activities_df, now.date())

    conclusion, suggestion, score = build_conclusion_and_suggestion(
        recovery,
        training_state,
        recovery_trend,
        activity,
        activity_trend,
    )
    signals = build_health_signals(recovery, training_state, recovery_trend, activity, data_status)

    payload = {
        "generated_at": now.isoformat(),
        "timezone": "Australia/Sydney",
        "source": "Intervals.icu API",
        "data_status": data_status,
        "conclusion": conclusion,
        "today_training_suggestion": suggestion,
        "internal_score": score,
        "recovery": recovery,
        "recovery_trend_7d": recovery_trend,
        "training_state": training_state,
        "yesterday_activity": activity,
        "activity_trend_7d": activity_trend,
        "health_signals_to_watch": signals,
        "analysis_order": [
            "direct_conclusion",
            "recovery",
            "yesterday_activity",
            "training_state",
            "today_training_suggestion",
            "health_signals_to_watch",
        ],
    }

    report_md = build_report(
        now,
        conclusion,
        suggestion,
        recovery,
        training_state,
        recovery_trend,
        activity,
        activity_trend,
        signals,
        data_status,
    )
    return payload, report_md, data_status


def write_outputs(payload: dict, report_md: str):
    (DATA_DIR / "daily_coach_input.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (DATA_DIR / "daily_report.md").write_text(report_md, encoding="utf-8")


def strtobool_env(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def main():
    max_retries = int(os.getenv("MAX_RECOVERY_RETRIES", "2"))
    retry_sleep_seconds = int(os.getenv("RETRY_SLEEP_SECONDS", "900"))
    retry_if_not_fresh = strtobool_env("RETRY_IF_NOT_FRESH", "1")

    attempt = 0
    last_payload = None
    last_report_md = None
    last_status = None

    while True:
        now = datetime.now(TZ)
        payload, report_md, data_status = build_payload_and_report(now)
        last_payload = payload
        last_report_md = report_md
        last_status = data_status

        recovery_fresh = bool(data_status.get("recovery_is_fresh"))
        training_fresh = bool(data_status.get("training_state_is_fresh"))
        print(
            f"Attempt {attempt + 1}: today={data_status.get('today_date')} "
            f"recovery_date={data_status.get('recovery_date')} "
            f"training_state_date={data_status.get('training_state_date')} "
            f"recovery_is_fresh={recovery_fresh} "
            f"training_state_is_fresh={training_fresh}"
        )

        if not retry_if_not_fresh:
            print("Retry disabled. Writing current outputs immediately.")
            break

        if recovery_fresh and training_fresh:
            print("Today's data is fresh. No retry needed.")
            break

        if attempt >= max_retries:
            print("Reached retry limit. Writing latest available outputs.")
            break

        attempt += 1
        print(
            f"Today's data is not fully fresh yet. Sleeping for {retry_sleep_seconds} seconds "
            f"before retry {attempt + 1}."
        )
        import time
        time.sleep(retry_sleep_seconds)

    write_outputs(last_payload, last_report_md)
    print("Auto pipeline finished.")


if __name__ == "__main__":
    main()
