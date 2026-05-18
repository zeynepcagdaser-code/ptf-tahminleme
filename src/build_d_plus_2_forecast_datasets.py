from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
BEFORE_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_dataset.csv"
AFTER_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_dataset.csv"

CUTOFF_HOUR = 14
BEFORE_TARGET_OFFSET_DAYS = 2
AFTER_TARGET_OFFSET_DAYS = 1

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
        target_date = issue_date + pd.Timedelta(days=BEFORE_TARGET_OFFSET_DAYS)
        cutoff_datetime = pd.Timestamp.combine(issue_date.date(), time(CUTOFF_HOUR, 0))

        if cutoff_datetime not in by_datetime.index:
            continue

        for target_hour in range(24):
            target_datetime = pd.Timestamp.combine(target_date.date(), time(target_hour, 0))
            if target_datetime not in by_datetime.index:
                continue

            base_row = _base_row(
                by_datetime,
                issue_date,
                target_date,
                target_hour,
                target_datetime,
                cutoff_datetime,
            )
            if base_row is None:
                continue

            before_rows.append(base_row)

    for issue_date in available_dates:
        target_date = issue_date + pd.Timedelta(days=AFTER_TARGET_OFFSET_DAYS)
        cutoff_datetime = pd.Timestamp.combine(issue_date.date(), time(CUTOFF_HOUR, 0))

        if cutoff_datetime not in by_datetime.index:
            continue

        issue_day_profile = _issue_day_partial_ptf_profile(by_datetime, issue_date, CUTOFF_HOUR)
        if issue_day_profile is None:
            continue

        for target_hour in range(24):
            target_datetime = pd.Timestamp.combine(target_date.date(), time(target_hour, 0))
            if target_datetime not in by_datetime.index:
                continue

            base_row = _base_row(
                by_datetime,
                issue_date,
                target_date,
                target_hour,
                target_datetime,
                cutoff_datetime,
            )
            if base_row is None:
                continue

            after_row = base_row.copy()
            after_row.update(_issue_day_partial_ptf_features(issue_day_profile, target_hour))
            after_rows.append(after_row)

    before = (
        pd.DataFrame(before_rows)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["ptf_target"])
        .reset_index(drop=True)
    )
    after = (
        pd.DataFrame(after_rows)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["ptf_target"])
        .reset_index(drop=True)
    )

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
    row["load_forecast_plan_at_cutoff"] = _load_forecast_value(by_datetime, cutoff_datetime)
    row.update(_ptf_cutoff_window_stats(by_datetime, cutoff_datetime))
    row.update(_derived_cutoff_features(row))
    row.update(_target_calendar_features(target_datetime))
    return row


def _issue_day_partial_ptf_profile(
    by_datetime: pd.DataFrame,
    issue_date: pd.Timestamp,
    cutoff_hour: int,
) -> pd.Series | None:
    hours = [
        pd.Timestamp.combine(issue_date.date(), time(hour, 0))
        for hour in range(cutoff_hour)
    ]
    if not hours or not set(hours).issubset(by_datetime.index):
        return None

    profile = by_datetime.loc[hours, "ptf"].reset_index(drop=True)
    if profile.isna().any() or len(profile) != len(hours):
        return None
    return profile


def _issue_day_partial_ptf_features(profile: pd.Series, target_hour: int) -> dict[str, float]:
    features = {
        f"issue_day_ptf_hour_{hour:02d}": float(profile.iloc[hour])
        for hour in range(len(profile))
    }

    if profile.empty:
        return features

    night_hours = [hour for hour in range(0, 6) if hour < len(profile)]
    morning_hours = [hour for hour in range(6, 12) if hour < len(profile)]
    midday_hours = [hour for hour in range(12, len(profile))]

    same_hour_value = float(profile.iloc[target_hour]) if target_hour < len(profile) else np.nan
    features.update(
        {
            "issue_day_ptf_same_target_hour": same_hour_value,
            "issue_day_ptf_partial_mean": float(profile.mean()),
            "issue_day_ptf_partial_max": float(profile.max()),
            "issue_day_ptf_partial_min": float(profile.min()),
            "issue_day_ptf_partial_volatility": float(profile.std()),
            "issue_day_ptf_night_mean": float(profile.iloc[night_hours].mean()) if night_hours else np.nan,
            "issue_day_ptf_morning_mean": float(profile.iloc[morning_hours].mean()) if morning_hours else np.nan,
            "issue_day_ptf_midday_mean": float(profile.iloc[midday_hours].mean()) if midday_hours else np.nan,
            "issue_day_ptf_partial_range": float(profile.max() - profile.min()),
        }
    )
    return features


def _ptf_cutoff_window_stats(
    by_datetime: pd.DataFrame,
    cutoff_datetime: pd.Timestamp,
) -> dict[str, float]:
    window_start = cutoff_datetime - pd.Timedelta(hours=23)
    window = by_datetime.loc[
        (by_datetime.index >= window_start) & (by_datetime.index <= cutoff_datetime),
        "ptf",
    ].dropna()
    if len(window) < 6:
        return {}

    return {
        "ptf_cutoff_window_mean_24h": float(window.mean()),
        "ptf_cutoff_window_std_24h": float(window.std()),
        "ptf_cutoff_window_max_24h": float(window.max()),
        "ptf_cutoff_window_min_24h": float(window.min()),
        "ptf_cutoff_window_range_24h": float(window.max() - window.min()),
    }


def _derived_cutoff_features(row: dict[str, float | int | str | pd.Timestamp]) -> dict[str, float]:
    features: dict[str, float] = {}

    ptf_1 = _numeric_or_nan(row.get("ptf_cutoff_lag_1"))
    ptf_24 = _numeric_or_nan(row.get("ptf_cutoff_lag_24"))
    ptf_168 = _numeric_or_nan(row.get("ptf_cutoff_lag_168"))
    smf_1 = _numeric_or_nan(row.get("smf_cutoff_lag_1"))
    load_fc = _numeric_or_nan(row.get("load_forecast_plan_at_cutoff"))
    consumption_1 = _numeric_or_nan(row.get("real_time_consumption_cutoff_lag_1"))
    wind_1 = _numeric_or_nan(row.get("wind_generation_cutoff_lag_1"))
    solar_1 = _numeric_or_nan(row.get("solar_generation_cutoff_lag_1"))
    hydro_1 = _numeric_or_nan(row.get("hydro_dam_generation_cutoff_lag_1"))

    if not np.isnan(ptf_1) and not np.isnan(ptf_24):
        features["ptf_momentum_cutoff_1_24"] = ptf_1 - ptf_24
        features["ptf_ratio_cutoff_1_24"] = ptf_1 / max(ptf_24, 1.0)
    if not np.isnan(ptf_24) and not np.isnan(ptf_168):
        features["ptf_ratio_cutoff_24_168"] = ptf_24 / max(ptf_168, 1.0)
    if not np.isnan(smf_1) and not np.isnan(ptf_1):
        features["smf_ptf_spread_cutoff_lag_1"] = smf_1 - ptf_1
    if not np.isnan(load_fc) and not np.isnan(consumption_1):
        features["load_forecast_to_consumption_ratio"] = load_fc / max(consumption_1, 1.0)

    renewable_parts = [value for value in (wind_1, solar_1, hydro_1) if not np.isnan(value)]
    if renewable_parts:
        features["renewable_generation_cutoff_lag_1"] = float(sum(renewable_parts))
        if not np.isnan(consumption_1):
            features["renewable_share_of_consumption"] = features["renewable_generation_cutoff_lag_1"] / max(
                consumption_1,
                1.0,
            )

    ptf_same_168 = _numeric_or_nan(row.get("ptf_same_hour_lag_168"))
    if not np.isnan(ptf_same_168) and not np.isnan(ptf_24):
        features["ptf_same_hour_vs_recent_day"] = ptf_same_168 - ptf_24

    return features


def _numeric_or_nan(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _load_forecast_value(by_datetime: pd.DataFrame, cutoff_datetime: pd.Timestamp) -> float:
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
