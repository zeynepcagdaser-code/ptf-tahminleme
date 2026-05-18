from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_12h_dataset.csv"

FORECAST_HORIZON_HOURS = 12
PTF_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
AUX_LAGS = [1, 24, 168]
PTF_SERIES_COLUMNS = ["ptf", "ptf_kesinlesmis"]

FEATURE_COLUMNS = [
    "ptf",
    "ptf_kesinlesmis",
    "smf",
    "real_time_consumption",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "gop_fiyattan_bagimsiz_alis",
    "gop_fiyattan_bagimsiz_satis",
    "price_independent_buy_sell_ratio",
    "load_forecast_plan",
    "grf_tl",
    "usd_try",
]

TARGET_COLUMN = "ptf_target"


@dataclass(frozen=True)
class Forecast12hDatasetSummary:
    rows: int
    issue_start: str
    issue_end: str
    output_path: str


def build_12h_forecast_dataset() -> tuple[pd.DataFrame, Forecast12hDatasetSummary]:
    hourly = _read_hourly_dataset()
    by_datetime = hourly.set_index("datetime").sort_index()

    rows: list[dict] = []
    for issue_datetime in by_datetime.index:
        if pd.isna(by_datetime.loc[issue_datetime, "ptf"]):
            continue

        base_features = _features_at_cutoff(by_datetime, issue_datetime)
        if base_features is None:
            continue

        for horizon in range(1, FORECAST_HORIZON_HOURS + 1):
            target_datetime = issue_datetime + pd.Timedelta(hours=horizon)
            if target_datetime not in by_datetime.index:
                continue

            target_ptf = by_datetime.loc[target_datetime, "ptf_kesinlesmis"]
            if pd.isna(target_ptf):
                target_ptf = by_datetime.loc[target_datetime, "ptf"]
            if pd.isna(target_ptf):
                continue

            row = {
                "issue_datetime": issue_datetime,
                "target_datetime": target_datetime,
                "forecast_horizon": horizon,
                TARGET_COLUMN: float(target_ptf),
                **base_features,
                **_target_calendar_features(target_datetime),
                "load_forecast_plan_target_hour": _load_forecast_at(
                    by_datetime, target_datetime, issue_datetime
                ),
            }
            rows.append(row)

    dataset = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)

    summary = Forecast12hDatasetSummary(
        rows=len(dataset),
        issue_start=str(dataset["issue_datetime"].min()) if not dataset.empty else "",
        issue_end=str(dataset["issue_datetime"].max()) if not dataset.empty else "",
        output_path=str(OUTPUT_PATH),
    )
    return dataset, summary


def _read_hourly_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Saatlik veri seti bulunamadi: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def _features_at_cutoff(by_datetime: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, float] | None:
    features: dict[str, float] = {}

    for column in FEATURE_COLUMNS:
        if column in {"load_forecast_plan", "grf_tl", "usd_try"}:
            continue
        lags = PTF_LAGS if column in PTF_SERIES_COLUMNS else AUX_LAGS
        for lag in lags:
            lookup = cutoff - pd.Timedelta(hours=lag)
            features[f"{column}_lag_{lag}"] = _value_at(by_datetime, lookup, column)

    window_start = cutoff - pd.Timedelta(hours=23)
    interim_window = by_datetime.loc[
        (by_datetime.index >= window_start) & (by_datetime.index <= cutoff),
        "ptf",
    ].dropna()
    if len(interim_window) >= 6:
        features["ptf_interim_window_mean_24h"] = float(interim_window.mean())
        features["ptf_interim_window_std_24h"] = float(interim_window.std())

    ptf_1 = features.get("ptf_lag_1", np.nan)
    ptf_24 = features.get("ptf_lag_24", np.nan)
    ptf_k_1 = features.get("ptf_kesinlesmis_lag_1", np.nan)
    if not pd.isna(ptf_1) and not pd.isna(ptf_24):
        features["ptf_interim_momentum_1_24"] = ptf_1 - ptf_24
    if not pd.isna(ptf_1) and not pd.isna(ptf_k_1):
        features["interim_vs_kesinlesmis_spread_lag_1"] = ptf_1 - ptf_k_1

    features["grf_tl_at_cutoff"] = _last_known(by_datetime, cutoff, "grf_tl")
    features["usd_try_at_cutoff"] = _last_known(by_datetime, cutoff, "usd_try")
    features["issue_hour"] = int(cutoff.hour)
    features["issue_day_of_week"] = int(cutoff.dayofweek)
    features["issue_is_weekend"] = int(cutoff.dayofweek in {5, 6})

    if pd.isna(features.get("ptf_lag_1")):
        return None
    return features


def _load_forecast_at(
    by_datetime: pd.DataFrame,
    target_datetime: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> float:
    value = _value_at(by_datetime, target_datetime, "load_forecast_plan")
    if not pd.isna(value):
        return float(value)
    return _last_known(by_datetime, cutoff, "load_forecast_plan")


def _target_calendar_features(target_datetime: pd.Timestamp) -> dict[str, float | int]:
    hour = target_datetime.hour
    dow = target_datetime.dayofweek
    return {
        "target_hour": hour,
        "target_day_of_week": dow,
        "target_month": target_datetime.month,
        "target_is_weekend": int(dow in {5, 6}),
        "target_hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "target_hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "target_dow_sin": float(np.sin(2 * np.pi * dow / 7)),
        "target_dow_cos": float(np.cos(2 * np.pi * dow / 7)),
    }


def _value_at(by_datetime: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    if timestamp not in by_datetime.index:
        return np.nan
    value = by_datetime.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def _last_known(by_datetime: pd.DataFrame, cutoff: pd.Timestamp, column: str) -> float:
    series = by_datetime.loc[by_datetime.index <= cutoff, column].dropna()
    if series.empty:
        return np.nan
    return float(series.iloc[-1])
