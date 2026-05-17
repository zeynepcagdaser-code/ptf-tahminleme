from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_24h_dataset.csv"

FORECAST_HORIZON_HOURS = 24


def build_24h_forecast_dataset() -> tuple[pd.DataFrame, dict[str, int | str]]:
    data = _read_final_hourly_dataset()
    forecast = pd.DataFrame(index=data.index)

    forecast["datetime"] = data["datetime"]
    forecast["target_datetime"] = data["datetime"] + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    forecast["ptf_target"] = data["ptf"].shift(-FORECAST_HORIZON_HOURS)

    _add_lags(forecast, data, "ptf", [1, 24, 168])
    _add_lags(forecast, data, "smf", [1, 24, 168])
    _add_lags(forecast, data, "real_time_consumption", [24, 168])
    _add_lags(forecast, data, "wind_generation", [24, 168])
    _add_lags(forecast, data, "solar_generation", [24, 168])
    _add_lags(forecast, data, "hydro_dam_generation", [24, 168])
    _add_lags(forecast, data, "gop_fiyattan_bagimsiz_alis", [24])
    _add_lags(forecast, data, "gop_fiyattan_bagimsiz_satis", [24])
    _add_lags(forecast, data, "price_independent_buy_sell_ratio", [24])

    forecast["load_forecast_plan"] = data["load_forecast_plan"]
    forecast["grf_tl"] = data["grf_tl"]
    forecast["usd_try"] = data["usd_try"]

    _add_target_time_features(forecast)

    before_drop = len(forecast)
    forecast = forecast.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    dropped_rows = before_drop - len(forecast)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "input_path": str(INPUT_PATH),
        "output_path": str(OUTPUT_PATH),
        "input_rows": len(data),
        "output_rows": len(forecast),
        "dropped_rows": dropped_rows,
        "feature_count": len(_feature_columns(forecast)),
    }
    return forecast, summary


def _read_final_hourly_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Final hourly dataset bulunamadi: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    required_columns = {
        "datetime",
        "ptf",
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
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"24h forecast dataset icin eksik kolonlar: {sorted(missing)}")

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    for column in required_columns - {"datetime"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def _add_lags(target: pd.DataFrame, source: pd.DataFrame, column: str, lags: list[int]) -> None:
    for lag in lags:
        target[f"{column}_lag_{lag}"] = source[column].shift(lag)


def _add_target_time_features(forecast: pd.DataFrame) -> None:
    target_dt = pd.to_datetime(forecast["target_datetime"], errors="coerce")
    hour = target_dt.dt.hour
    day_of_week = target_dt.dt.dayofweek
    month = target_dt.dt.month

    forecast["hour"] = hour
    forecast["day_of_week"] = day_of_week
    forecast["month"] = month
    forecast["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    forecast["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    forecast["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    forecast["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    forecast["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in {"datetime", "target_datetime", "ptf_target"}]
