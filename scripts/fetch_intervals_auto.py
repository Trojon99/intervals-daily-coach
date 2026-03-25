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


def load_data():
    now = datetime.now(TZ)
    today = now.date()

    wellness_oldest = today - timedelta(days=21)
    wellness_newest = today + timedelta(days=1)

    activities_oldest = today - timedelta(days=7)
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


def prepare_wellness(df: pd.DataFrame, today_date):
    if df.empty:
        return {}, {}

    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
        df = df[df["date"].dt.date <= today_date]
        df = df.sort_values("date")

    if df.empty:
        return {}, {}

    # Training-state metrics can live on a placeholder row for today/tomorrow.
    # Recovery metrics should come from the most recent row that actually has
    # recovery data, not simply the last row in the file.
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
    }

    return recovery, training_state


def prepare_activity(df: pd.DataFrame, today_date):
    if df.empty:
        return {}

    df = df.copy()
    if "start_date_local" in df.columns:
        df["start_date_local"] = pd.to_datetime(df["start_date_local"], errors="coerce")
        df = df.sort_values("start_date_local")

    yesterday = today_date - timedelta(days=1)
    chosen = None

    if "start_date_local" in df.columns:
        y_df = df[df["start_date_local"].dt.date == yesterday]
        if not y_df.empty:
            chosen = y_df.iloc[-1]

    if chosen is None:
        chosen = df.iloc[-1]

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
        "icu_training_load": pick(chosen, "icu_training_load", "training_load"),
        "icu_fitness": pick(chosen, "icu_fitness"),
        "icu_fatigue": pick(chosen, "icu_fatigue"),
        "icu_eftp": pick(chosen, "icu_eftp"),
        "icu_average_watts": pick(chosen, "icu_average_watts", "average_watts"),
        "icu_normalized_watts": pick(chosen, "icu_normalized_watts", "normalized_watts"),
    }

    return activity


def build_conclusion(recovery, training_state, activity):
    score = 0

    sleep_hours = recovery.get("sleep_hours")
    if sleep_hours is not None:
        if sleep_hours >= 7:
            score += 2
        elif sleep_hours >= 6:
            score += 1
        else:
            score -= 2

    sleep_score = recovery.get("sleep_score")
    if sleep_score is not None:
        if sleep_score >= 80:
            score += 2
        elif sleep_score >= 70:
            score += 1
        elif sleep_score < 60:
            score -= 2

    hrv = recovery.get("hrv")
    if hrv is not None and hrv > 0:
        score += 1

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

    atl = training_state.get("atl")
    ctl = training_state.get("ctl")
    if atl is not None and ctl is not None:
        try:
            if float(atl) > float(ctl) * 1.25:
                score -= 2
            elif float(atl) > float(ctl) * 1.10:
                score -= 1
        except Exception:
            pass

    load = activity.get("icu_training_load")
    if load is not None:
        try:
            if float(load) >= 70:
                score -= 1
        except Exception:
            pass

    if score >= 3:
        return "恢复较好，可正常训练；是否加码仍要看主观感觉。"
    if score >= 1:
        return "恢复尚可，适合常规训练，不建议激进加码。"
    if score >= -1:
        return "恢复一般，今天更适合轻松有氧或降强度。"
    return "恢复偏弱，今天不适合高强度，优先恢复。"


def build_report(now, recovery, training_state, activity, conclusion):
    lines = [
        "# Daily Coach Report",
        "",
        f"- Generated: {now.isoformat()}",
        f"- Conclusion: {conclusion}",
        "",
        "## Recovery",
    ]
    for k, v in recovery.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Training State"]
    for k, v in training_state.items():
        lines.append(f"- {k}: {v}")

    lines += ["", "## Yesterday Activity"]
    for k, v in activity.items():
        lines.append(f"- {k}: {v}")

    return "\n".join(lines) + "\n"


def main():
    wellness_df, activities_df, now = load_data()
    recovery, training_state = prepare_wellness(wellness_df, now.date())
    activity = prepare_activity(activities_df, now.date())

    conclusion = build_conclusion(recovery, training_state, activity)

    payload = {
        "generated_at": now.isoformat(),
        "timezone": "Australia/Sydney",
        "source": "Intervals.icu API",
        "conclusion": conclusion,
        "recovery": recovery,
        "training_state": training_state,
        "yesterday_activity": activity,
        "analysis_order": [
            "direct_conclusion",
            "recovery",
            "yesterday_activity",
            "training_state",
            "today_training_suggestion",
            "health_signals_to_watch",
        ],
    }

    (DATA_DIR / "daily_coach_input.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    report_md = build_report(now, recovery, training_state, activity, conclusion)
    (DATA_DIR / "daily_report.md").write_text(report_md, encoding="utf-8")

    print("Auto pipeline finished.")


if __name__ == "__main__":
    main()
