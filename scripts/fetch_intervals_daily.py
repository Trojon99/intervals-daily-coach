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

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_KEY = require_env("INTERVALS_API_KEY")


def get_text(url: str, params: dict | None = None) -> str:
    resp = requests.get(
        url,
        auth=(AUTH_USER, API_KEY),
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.text


def read_csv_text(text: str) -> pd.DataFrame:
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(StringIO(text))


def safe_value(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    return v


def last_non_null(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    return safe_value(s.iloc[-1])


def column_or_none(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else pd.Series(dtype="object")


def build_wellness_url() -> str:
    return f"{BASE_URL}/athlete/0/wellness.csv"


def build_activities_url() -> str:
    return f"{BASE_URL}/athlete/0/activities.csv"


def main():
    now = datetime.now(TZ)
    today = now.date()

    wellness_oldest = today - timedelta(days=21)
    wellness_newest = today + timedelta(days=1)

    activities_oldest = today - timedelta(days=7)
    activities_newest = today + timedelta(days=1)

    # 官方论坛给出的 wellness 可选字段清单里包含这些列
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
        "soreness",
        "fatigue",
        "stress",
        "mood",
        "motivation",
        "spO2",
        "injury",
        "ctl",
        "atl",
        "rampRate",
        "ctlLoad",
        "atlLoad",
        "Ride_eftp",
        "Run_eftp"
    ])

    wellness_text = get_text(
        build_wellness_url(),
        params={
            "oldest": wellness_oldest.isoformat(),
            "newest": wellness_newest.isoformat(),
            "cols": wellness_cols,
        },
    )

    activities_text = get_text(
        build_activities_url(),
        params={
            "oldest": activities_oldest.isoformat(),
            "newest": activities_newest.isoformat(),
        },
    )

    # 保存原始文件
    (RAW_DIR / "wellness.csv").write_text(wellness_text, encoding="utf-8")
    (RAW_DIR / "activities.csv").write_text(activities_text, encoding="utf-8")

    wellness = read_csv_text(wellness_text)
    activities = read_csv_text(activities_text)

    # 记录列名，便于之后核对字段
    (DATA_DIR / "wellness_columns.json").write_text(
        json.dumps(list(wellness.columns), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (DATA_DIR / "activities_columns.json").write_text(
        json.dumps(list(activities.columns), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---------- 处理 wellness ----------
    latest_wellness = {}
    if not wellness.empty:
        # date 列在论坛里说明是默认自带的
        if "date" in wellness.columns:
            wellness["date"] = pd.to_datetime(wellness["date"], errors="coerce")
            wellness = wellness.sort_values("date")

        w = wellness.iloc[-1]

        latest_wellness = {
            "date": safe_value(w.get("date")),
            "weight": safe_value(w.get("weight")),
            "resting_hr": safe_value(w.get("restingHR")),
            "hrv": safe_value(w.get("hrv")),
            "hrv_sdnn": safe_value(w.get("hrvSDNN")),
            "readiness": safe_value(w.get("readiness")),
            "sleep_secs": safe_value(w.get("sleepSecs")),
            "sleep_hours": round(float(w["sleepSecs"]) / 3600, 2) if "sleepSecs" in w and pd.notna(w["sleepSecs"]) else None,
            "sleep_score": safe_value(w.get("sleepScore")),
            "sleep_quality": safe_value(w.get("sleepQuality")),
            "avg_sleeping_hr": safe_value(w.get("avgSleepingHR")),
            "spO2": safe_value(w.get("spO2")),
            "soreness": safe_value(w.get("soreness")),
            "fatigue": safe_value(w.get("fatigue")),
            "stress": safe_value(w.get("stress")),
            "mood": safe_value(w.get("mood")),
            "motivation": safe_value(w.get("motivation")),
            "injury": safe_value(w.get("injury")),
            "ctl": safe_value(w.get("ctl")),
            "atl": safe_value(w.get("atl")),
            "ramp_rate": safe_value(w.get("rampRate")),
            "ctl_load": safe_value(w.get("ctlLoad")),
            "atl_load": safe_value(w.get("atlLoad")),
            "ride_eftp": safe_value(w.get("Ride_eftp")),
            "run_eftp": safe_value(w.get("Run_eftp")),
        }

    # ---------- 处理 activities ----------
    latest_activity = {}
    if not activities.empty:
        if "start_date_local" in activities.columns:
            activities["start_date_local"] = pd.to_datetime(
                activities["start_date_local"], errors="coerce"
            )
            activities = activities.sort_values("start_date_local")

        # 优先找“昨天”的活动；没有就取最近一次
        target_date = pd.Timestamp(today - timedelta(days=1))
        chosen = None

        if "start_date_local" in activities.columns:
            same_day = activities[
                activities["start_date_local"].dt.date == target_date.date()
            ]
            if not same_day.empty:
                chosen = same_day.iloc[-1]

        if chosen is None:
            chosen = activities.iloc[-1]

        latest_activity = {
            "id": safe_value(chosen.get("id")),
            "start_date_local": safe_value(chosen.get("start_date_local")),
            "name": safe_value(chosen.get("name")),
            "type": safe_value(chosen.get("type")),
            "moving_time_sec": safe_value(chosen.get("moving_time")),
            "elapsed_time_sec": safe_value(chosen.get("elapsed_time")),
            "distance_m": safe_value(chosen.get("distance")),
            "distance_km": round(float(chosen["distance"]) / 1000, 2) if "distance" in chosen and pd.notna(chosen["distance"]) else None,
            "total_elevation_gain": safe_value(chosen.get("total_elevation_gain")),
            "average_heartrate": safe_value(chosen.get("average_heartrate")),
            "max_heartrate": safe_value(chosen.get("max_heartrate")),
            "average_speed": safe_value(chosen.get("average_speed")),
            "max_speed": safe_value(chosen.get("max_speed")),
            "calories": safe_value(chosen.get("calories")),
            "icu_training_load": safe_value(chosen.get("icu_training_load")),
            "icu_fitness": safe_value(chosen.get("icu_fitness")),
            "icu_fatigue": safe_value(chosen.get("icu_fatigue")),
            "icu_eftp": safe_value(chosen.get("icu_eftp")),
            "icu_average_watts": safe_value(chosen.get("icu_average_watts")),
            "icu_normalized_watts": safe_value(chosen.get("icu_normalized_watts")),
            "icu_efficiency": safe_value(chosen.get("icu_efficiency")),
        }

    # ---------- 生成给教练分析的输入 ----------
    coach_input = {
        "generated_at": now.isoformat(),
        "timezone": "Australia/Sydney",
        "source": "Intervals.icu API",
        "recovery": latest_wellness,
        "yesterday_activity": latest_activity,
        "notes_for_analysis": {
            "priority_order": [
                "sleep/recovery",
                "training_state",
                "yesterday_activity",
                "subjective_feeling"
            ],
            "important_signals": [
                "sleep_hours",
                "sleep_score",
                "sleep_quality",
                "avg_sleeping_hr",
                "resting_hr",
                "hrv",
                "readiness",
                "fatigue",
                "soreness",
                "stress",
                "ctl",
                "atl",
                "ramp_rate",
                "icu_training_load"
            ]
        }
    }

    (DATA_DIR / "daily_coach_input.json").write_text(
        json.dumps(coach_input, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("Saved files:")
    print("- data/raw/wellness.csv")
    print("- data/raw/activities.csv")
    print("- data/wellness_columns.json")
    print("- data/activities_columns.json")
    print("- data/daily_coach_input.json")


if __name__ == "__main__":
    main()