from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "final_feature_dataset.csv"
OUTPUT_XLSX_PATH = PROJECT_ROOT / "data" / "processed" / "final_feature_dataset.xlsx"
REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineering_report.json"

FEATURE_SOURCE_COLUMNS = [
    "ptf",
    "smf",
    "grf_tl",
    "usd_try",
    "load_forecast_plan",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "real_time_consumption",
    "price_independent_buy_sell_ratio",
]

LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLLING_WINDOWS = [3, 6, 12, 24, 72, 168]
ROLLING_STATS = ["mean", "std", "min", "max"]
CHANGE_PERIODS = LAG_PERIODS
VOLATILITY_WINDOWS = ROLLING_WINDOWS
EPSILON = 1e-9


@dataclass(frozen=True)
class FeatureEngineeringReport:
    initial_row_count: int
    initial_column_count: int
    final_row_count: int
    final_column_count: int
    generated_lag_count: int
    generated_rolling_feature_count: int
    generated_change_feature_count: int
    generated_volatility_feature_count: int
    generated_time_feature_count: int
    dropped_nan_row_count: int
    output_csv_path: str
    output_xlsx_path: str


def run_final_dataset_feature_engineering() -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    dataset = _read_final_dataset()
    initial_row_count = len(dataset)
    initial_column_count = len(dataset.columns)

    lag_features = _build_lag_features(dataset)
    rolling_features = _build_rolling_features(dataset)
    change_features = _build_change_features(dataset)
    volatility_features = _build_volatility_features(dataset)
    time_features = _build_time_features(dataset)

    feature_dataset = pd.concat(
        [dataset, lag_features, rolling_features, change_features, volatility_features, time_features],
        axis=1,
    )
    feature_dataset = feature_dataset.replace([np.inf, -np.inf], np.nan)

    before_drop = len(feature_dataset)
    feature_dataset = feature_dataset.dropna().reset_index(drop=True)
    dropped_nan_row_count = before_drop - len(feature_dataset)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_dataset.to_csv(OUTPUT_CSV_PATH, index=False)
    feature_dataset.to_excel(OUTPUT_XLSX_PATH, index=False)

    report = FeatureEngineeringReport(
        initial_row_count=initial_row_count,
        initial_column_count=initial_column_count,
        final_row_count=len(feature_dataset),
        final_column_count=len(feature_dataset.columns),
        generated_lag_count=len(lag_features.columns),
        generated_rolling_feature_count=len(rolling_features.columns),
        generated_change_feature_count=len(change_features.columns),
        generated_volatility_feature_count=len(volatility_features.columns),
        generated_time_feature_count=len(time_features.columns),
        dropped_nan_row_count=dropped_nan_row_count,
        output_csv_path=str(OUTPUT_CSV_PATH),
        output_xlsx_path=str(OUTPUT_XLSX_PATH),
    )
    REPORT_PATH.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return feature_dataset, report


def _read_final_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Final saatlik veri seti bulunamadi: {INPUT_PATH}")

    dataset = pd.read_csv(INPUT_PATH)
    missing_columns = [column for column in FEATURE_SOURCE_COLUMNS if column not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Feature engineering icin eksik kolonlar: {missing_columns}")

    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
    dataset = dataset.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce").dt.date.astype(str)
    dataset["hour"] = pd.to_numeric(dataset["hour"], errors="coerce").astype(int)

    for column in FEATURE_SOURCE_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    return dataset


def _build_lag_features(dataset: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in FEATURE_SOURCE_COLUMNS:
        series = dataset[column]
        for lag in LAG_PERIODS:
            features[f"{column}_lag_{lag}"] = series.shift(lag)
    return pd.DataFrame(features, index=dataset.index)


def _build_rolling_features(dataset: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in FEATURE_SOURCE_COLUMNS:
        shifted = dataset[column].shift(1)
        for window in ROLLING_WINDOWS:
            rolling = shifted.rolling(window=window, min_periods=window)
            features[f"{column}_rolling_mean_{window}"] = rolling.mean()
            features[f"{column}_rolling_std_{window}"] = rolling.std()
            features[f"{column}_rolling_min_{window}"] = rolling.min()
            features[f"{column}_rolling_max_{window}"] = rolling.max()
    return pd.DataFrame(features, index=dataset.index)


def _build_change_features(dataset: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in FEATURE_SOURCE_COLUMNS:
        previous_value = dataset[column].shift(1)
        for period in CHANGE_PERIODS:
            older_value = dataset[column].shift(period + 1)
            delta = previous_value - older_value
            pct_change = _safe_divide(delta, older_value.abs())
            momentum = previous_value - previous_value.rolling(window=period, min_periods=period).mean()

            features[f"{column}_delta_{period}"] = delta
            features[f"{column}_pct_change_{period}"] = pct_change
            features[f"{column}_momentum_{period}"] = momentum
    return pd.DataFrame(features, index=dataset.index)


def _build_volatility_features(dataset: pd.DataFrame) -> pd.DataFrame:
    features: dict[str, pd.Series] = {}
    for column in FEATURE_SOURCE_COLUMNS:
        shifted = dataset[column].shift(1)
        for window in VOLATILITY_WINDOWS:
            rolling = shifted.rolling(window=window, min_periods=window)
            mean = rolling.mean()
            std = rolling.std()
            variance = rolling.var()

            features[f"{column}_volatility_{window}"] = _safe_divide(std, mean.abs())
            features[f"{column}_variance_{window}"] = variance
            features[f"{column}_zscore_{window}"] = _safe_divide(shifted - mean, std)
    return pd.DataFrame(features, index=dataset.index)


def _build_time_features(dataset: pd.DataFrame) -> pd.DataFrame:
    datetime_series = dataset["datetime"]
    hour = pd.to_numeric(dataset["hour"], errors="coerce")
    day_of_week = datetime_series.dt.dayofweek
    month = datetime_series.dt.month
    day = datetime_series.dt.day

    features = pd.DataFrame(index=dataset.index)
    features["day_of_week"] = day_of_week
    features["month"] = month
    features["day"] = day
    features["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    return features


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.mask(denominator.abs() < EPSILON)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
