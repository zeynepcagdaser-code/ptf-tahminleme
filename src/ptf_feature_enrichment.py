from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1.0


def enrich_forecast_features(data: pd.DataFrame) -> pd.DataFrame:
    """12h forecast dataset uzerine model icin turetilmis ozellikler."""
    out = data.copy()

    out["issue_hour_sin"] = np.sin(2 * np.pi * out["issue_hour"] / 24)
    out["issue_hour_cos"] = np.cos(2 * np.pi * out["issue_hour"] / 24)
    out["issue_dow_sin"] = np.sin(2 * np.pi * out["issue_day_of_week"] / 7)
    out["issue_dow_cos"] = np.cos(2 * np.pi * out["issue_day_of_week"] / 7)
    out["issue_month"] = out["issue_datetime"].dt.month

    out["hour_weekend_interaction"] = out["issue_hour"] * out["issue_is_weekend"]
    out["target_hour_weekend_interaction"] = out["target_hour"] * out["target_is_weekend"]
    out["horizon_squared"] = out["forecast_horizon"] ** 2

    if "ptf_lag_1" in out.columns and "ptf_lag_24" in out.columns:
        out["ptf_momentum_1_24"] = out["ptf_lag_1"] - out["ptf_lag_24"]
        out["ptf_ratio_1_24"] = out["ptf_lag_1"] / (out["ptf_lag_24"].abs() + EPSILON)
    if "ptf_kesinlesmis_lag_1" in out.columns and "ptf_kesinlesmis_lag_24" in out.columns:
        out["kesin_momentum_1_24"] = out["ptf_kesinlesmis_lag_1"] - out["ptf_kesinlesmis_lag_24"]

    if "kesin_seasonal_anchor" in out.columns and "ptf_lag_1" in out.columns:
        out["interim_vs_seasonal_anchor"] = out["ptf_lag_1"] - out["kesin_seasonal_anchor"]

    if "ptf_interim_window_mean_24h" in out.columns:
        out["ptf_interim_window_cv_24h"] = out["ptf_interim_window_std_24h"] / (
            out["ptf_interim_window_mean_24h"].abs() + EPSILON
        )

    if "kesin_seasonal_anchor" in out.columns and "kesin_at_target_week_ago" in out.columns:
        out["seasonal_yesterday_vs_week"] = (
            out["kesin_seasonal_anchor"] - out["kesin_at_target_week_ago"]
        )

    return out


def recency_sample_weights(issue_times: pd.Series, half_life_days: float = 45.0) -> np.ndarray:
    """Son doneme daha yuksek agirlik."""
    times = pd.to_datetime(issue_times, errors="coerce")
    max_time = times.max()
    age_days = (max_time - times).dt.total_seconds() / 86400.0
    weights = np.power(0.5, age_days / half_life_days)
    return weights.fillna(1.0).to_numpy(dtype=float)
