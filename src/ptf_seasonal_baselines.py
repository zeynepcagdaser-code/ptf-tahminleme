from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_kesin_hourly(hourly_data: pd.DataFrame) -> pd.DataFrame:
    hourly = hourly_data.set_index("datetime").sort_index()
    if "ptf_kesinlesmis" not in hourly.columns:
        raise ValueError("ptf_kesinlesmis kolonu gerekli")
    hourly["kesin_rolling_24h"] = hourly["ptf_kesinlesmis"].rolling(window=24, min_periods=1).mean()
    hourly["kesin_rolling_168h"] = hourly["ptf_kesinlesmis"].rolling(window=168, min_periods=1).mean()
    return hourly


def kesin_value_at(hourly: pd.DataFrame, timestamp: pd.Timestamp, column: str = "ptf_kesinlesmis") -> float:
    if timestamp not in hourly.index:
        return np.nan
    value = hourly.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if not pd.isna(value) else np.nan


def seasonal_predictions_for_target(
    hourly: pd.DataFrame,
    issue_dt: pd.Timestamp,
    target_dt: pd.Timestamp,
) -> dict[str, float]:
    """Hedef saat icin kesinlesmis PTF mevsimsel tahminleri (sizinti yok)."""
    y_dt = target_dt - pd.Timedelta(hours=24)
    w_dt = target_dt - pd.Timedelta(hours=168)

    kesin_yesterday = kesin_value_at(hourly, y_dt)
    kesin_week = kesin_value_at(hourly, w_dt)
    rolling_24h = kesin_value_at(hourly, issue_dt, "kesin_rolling_24h")
    rolling_168h = kesin_value_at(hourly, issue_dt, "kesin_rolling_168h")

    last_24h = hourly.loc[
        (hourly.index > issue_dt - pd.Timedelta(hours=24)) & (hourly.index <= issue_dt),
        "ptf_kesinlesmis",
    ]
    kesin_last_24h_mean = float(last_24h.mean()) if len(last_24h) else np.nan

    parts: list[float] = []
    weights: list[float] = []
    for value, weight in (
        (kesin_yesterday, 0.50),
        (kesin_week, 0.30),
        (rolling_168h, 0.15),
        (rolling_24h, 0.05),
    ):
        if not np.isnan(value):
            parts.append(value)
            weights.append(weight)

    seasonal_blend = (
        float(sum(p * w for p, w in zip(parts, weights)) / sum(weights)) if parts else np.nan
    )

    return {
        "pred_kesin_same_hour_yesterday": kesin_yesterday,
        "pred_kesin_same_hour_last_week": kesin_week,
        "pred_kesin_last_24h_mean": kesin_last_24h_mean,
        "pred_kesin_rolling_24h": rolling_24h,
        "pred_kesin_rolling_168h": rolling_168h,
        "pred_seasonal_blend": seasonal_blend,
        "kesin_seasonal_anchor": kesin_yesterday,
    }
