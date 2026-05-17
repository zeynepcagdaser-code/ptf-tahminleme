from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
BEFORE_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_dataset.csv"
AFTER_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_dataset.csv"

CUTOFF_HOUR = 13
TARGET_OFFSET_DAYS = 2
D_PLUS_1_OFFSET_DAYS = 1

LAG_COLUMNS = [
    "ptf",
    "smf",
    "real_time_consumption",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "unlicensed_generation_total",
    "gop_fiyattan_bagimsiz_alis",
    "gop_fiyattan_bagimsiz_satis",
    "price_independent_buy_sell_ratio",
]
CUTOFF_LAGS = [1, 24, 168]
SAME_HOUR_SAFE_LAGS = [72, 168]


@dataclass(frozen=True)
class DPlus2DatasetSummary:
    before_rows: int
    after_rows: int
    before_columns: int
    after_columns: int
    first_forecast_issue_date: str
    last_forecast_issue_date: str
    before_path: str
    after_path: str


def build_d_plus_2_forecast_datasets() -> tuple[pd.DataFrame, pd.DataFrame, DPlus2DatasetSummary]:
    data = _read_final_hourly_dataset()
    by_datetime = data.set_index("datetime").sort_index()
    available_dates = sorted(data["date"].drop_duplicates())

    before_rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    after_rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for issue_date in available_dates:
        target_date = issue_date + pd.Timedelta(days=TARGET_OFFSET_DAYS)
        d_plus_1_date = issue_date + pd.Timedelta(days=D_PLUS_1_OFFSET_DAYS)
        cutoff_datetime = pd.Timestamp.combine(issue_date.date(), time(CUTOFF_HOUR, 0))

        if cutoff_datetime not in by_datetime.index:
            continue

        d_plus_1_profile = _d_plus_1_profile(by_datetime, d_plus_1_date)
        for target_hour in range(24):
            target_datetime = pd.Timestamp.combine(target_date.date(), time(target_hour, 0))
            if target_datetime not in by_datetime.index:
                continue

            base_row = _base_row(by_datetime, issue_date, target_date, target_hour, target_datetime, cutoff_datetime)
            if base_row is None:
                continue

            before_rows.append(base_row.copy())
            if d_plus_1_profile is not None:
                after_row = base_row.copy()
                after_row.update(_d_plus_1_features(d_plus_1_profile, target_hour))
                after_rows.append(after_row)

    before = pd.DataFrame(before_rows).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    after = pd.DataFrame(after_rows).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    BEFORE_14_PATH.parent.mkdir(parents=True, exist_ok=True)
    before.to_csv(BEFORE_14_PATH, index=False)
    after.to_csv(AFTER_14_PATH, index=False)

    summary = DPlus2DatasetSummary(
        before_rows=len(before),
        after_rows=len(after),
        before_columns=len(before.columns),
        after_columns=len(after.columns),
        first_forecast_issue_date=str(before["forecast_issue_date"].min()) if not before.empty else "",
        last_forecast_issue_date=str(before["forecast_issue_date"].max()) if not before.empty else "",
        before_path=str(BEFORE_14_PATH),
        after_path=str(AFTER_14_PATH),
    )
    return before, after, summary


def _read_final_hourly_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Final hourly dataset bulunamadi: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required = {
        "datetime",
        "ptf",
        "smf",
        "real_time_consumption",
        "wind_generation",
        "solar_generation",
        "hydro_dam_generation",
        "unlicensed_generation_total",
        "gop_fiyattan_bagimsiz_alis",
        "gop_fiyattan_bagimsiz_satis",
        "price_independent_buy_sell_ratio",
        "load_forecast_plan",
        "grf_tl",
        "usd_try",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"D+2 dataset icin eksik kolonlar: {sorted(missing)}")

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    data["date"] = data["datetime"].dt.normalize()
    data["hour"] = data["datetime"].dt.hour

    for column in required - {"datetime"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _base_row(
    by_datetime: pd.DataFrame,
    issue_date: pd.Timestamp,
    target_date: pd.Timestamp,
    target_hour: int,
    target_datetime: pd.Timestamp,
    cutoff_datetime: pd.Timestamp,
) -> dict[str, float | int | str | pd.Timestamp] | None:
    target = by_datetime.loc[target_datetime, "ptf"]
    if pd.isna(target):
        return None

    row: dict[str, float | int | str | pd.Timestamp] = {
        "forecast_issue_date": issue_date.date().isoformat(),
        "target_date": target_date.date().isoformat(),
        "target_hour": target_hour,
        "target_datetime": target_datetime,
        "ptf_target": float(target),
    }

    for column in LAG_COLUMNS:
        for lag in CUTOFF_LAGS:
            lookup_time = cutoff_datetime - pd.Timedelta(hours=lag)
            row[f"{column}_cutoff_lag_{lag}"] = _value_at(by_datetime, lookup_time, column)

        for lag in SAME_HOUR_SAFE_LAGS:
            lookup_time = target_datetime - pd.Timedelta(hours=lag)
            if lookup_time <= cutoff_datetime:
                row[f"{column}_same_hour_lag_{lag}"] = _value_at(by_datetime, lookup_time, column)
            else:
                row[f"{column}_same_hour_lag_{lag}"] = np.nan

    row["grf_tl_issue_day"] = _last_known_value(by_datetime, cutoff_datetime, "grf_tl")
    row["usd_try_issue_day"] = _last_known_value(by_datetime, cutoff_datetime, "usd_try")
    row["load_forecast_plan_target_hour"] = _load_forecast_value(by_datetime, target_datetime, cutoff_datetime)

    row.update(_target_calendar_features(target_datetime))
    return row


def _d_plus_1_profile(by_datetime: pd.DataFrame, d_plus_1_date: pd.Timestamp) -> pd.Series | None:
    start = pd.Timestamp.combine(d_plus_1_date.date(), time(0, 0))
    hours = [start + pd.Timedelta(hours=hour) for hour in range(24)]
    if not set(hours).issubset(by_datetime.index):
        return None
    profile = by_datetime.loc[hours, "ptf"].reset_index(drop=True)
    if profile.isna().any() or len(profile) != 24:
        return None
    return profile


def _d_plus_1_features(profile: pd.Series, target_hour: int) -> dict[str, float]:
    features = {f"d_plus_1_ptf_hour_{hour:02d}": float(profile.iloc[hour]) for hour in range(24)}
    evening_peak = profile.iloc[[18, 19, 20, 21, 22]]
    daytime = profile.iloc[list(range(8, 18))]
    features.update(
        {
            "d_plus_1_ptf_same_target_hour": float(profile.iloc[target_hour]),
            "d_plus_1_ptf_daily_mean": float(profile.mean()),
            "d_plus_1_ptf_daily_max": float(profile.max()),
            "d_plus_1_ptf_daily_min": float(profile.min()),
            "d_plus_1_ptf_daily_volatility": float(profile.std()),
            "d_plus_1_ptf_evening_peak_mean": float(evening_peak.mean()),
            "d_plus_1_ptf_daytime_mean": float(daytime.mean()),
        }
    )
    return features


def _load_forecast_value(
    by_datetime: pd.DataFrame,
    target_datetime: pd.Timestamp,
    cutoff_datetime: pd.Timestamp,
) -> float:
    value = _value_at(by_datetime, target_datetime, "load_forecast_plan")
    if not pd.isna(value):
        return float(value)
    return _last_known_value(by_datetime, cutoff_datetime, "load_forecast_plan")


def _target_calendar_features(target_datetime: pd.Timestamp) -> dict[str, float | int]:
    hour = target_datetime.hour
    day_of_week = target_datetime.dayofweek
    month = target_datetime.month
    return {
        "target_hour_calendar": hour,
        "target_day_of_week": day_of_week,
        "target_month": month,
        "target_is_weekend": int(day_of_week in {5, 6}),
        "target_hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "target_hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "target_dow_sin": float(np.sin(2 * np.pi * day_of_week / 7)),
        "target_dow_cos": float(np.cos(2 * np.pi * day_of_week / 7)),
    }


def _value_at(by_datetime: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if timestamp not in by_datetime.index:
        return np.nan
    value = by_datetime.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def _last_known_value(by_datetime: pd.DataFrame, cutoff_datetime: pd.Timestamp, column: str) -> float:
    available = by_datetime.loc[by_datetime.index <= cutoff_datetime, column].dropna()
    if available.empty:
        return np.nan
    return float(available.iloc[-1])
