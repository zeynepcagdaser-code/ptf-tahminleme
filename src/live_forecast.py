from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.build_final_ensemble import (
    ENSEMBLE_SOURCE_COLS,
    _apply_ensemble,
    _apply_weight_caps,
    _hybrid_prediction,
    _predict_with_strategy,
)
from src.config import PROJECT_ROOT
from src.ptf_seasonal_baselines import prepare_kesin_hourly, seasonal_predictions_for_target


HOURLY_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
CATBOOST_LIVE_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_live_forecast.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_metrics.json"
LIVE_BUNDLE_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_live_bundle.csv"


def build_live_forecast_bundle() -> pd.DataFrame:
    hourly_raw = pd.read_csv(HOURLY_PATH)
    hourly_raw["datetime"] = pd.to_datetime(hourly_raw["datetime"], errors="coerce")
    hourly_raw = hourly_raw.dropna(subset=["datetime"]).sort_values("datetime")

    hourly_idx = prepare_kesin_hourly(hourly_raw)
    ptf_known = hourly_idx["ptf"].dropna() if "ptf" in hourly_idx.columns else pd.Series(dtype=float)
    if ptf_known.empty:
        return pd.DataFrame()

    cutoff = ptf_known.index.max()
    interim_at_cutoff = float(ptf_known.iloc[-1])

    catboost_live = _read_catboost_live(cutoff)
    params = _load_ensemble_params()

    rows: list[dict] = []
    for horizon in range(1, 13):
        target_dt = cutoff + pd.Timedelta(hours=horizon)
        seasonal = seasonal_predictions_for_target(hourly_idx, cutoff, target_dt)
        seasonal_ptf = float(seasonal.get("pred_seasonal_blend", np.nan))
        catboost = _catboost_price(catboost_live, horizon)

        pred_row = {
            "pred_kesin_same_hour_yesterday": seasonal["pred_kesin_same_hour_yesterday"],
            "pred_kesin_same_hour_last_week": seasonal["pred_kesin_same_hour_last_week"],
            "pred_kesin_rolling_24h": seasonal["pred_kesin_rolling_24h"],
            "pred_kesin_rolling_168h": seasonal["pred_kesin_rolling_168h"],
            "pred_seasonal_blend": seasonal_ptf,
            "pred_catboost": catboost if catboost is not None else np.nan,
        }
        row_df = pd.DataFrame([pred_row])

        primary = params["primary_model"].get(horizon, "blend")
        panel = _predict_with_strategy(
            row_df,
            horizon,
            primary,
            params["weights"].get(horizon, {}),
            params["biases"].get(horizon, 0.0),
            params["blend_weights"].get(horizon, 1.0),
            params["biases"].get(horizon, 0.0),
        )[0]

        stacked = _apply_ensemble(
            row_df,
            [c for c in ENSEMBLE_SOURCE_COLS if c in row_df.columns],
            _apply_weight_caps(params["weights"].get(horizon, {})),
            params["biases"].get(horizon, 0.0),
        )[0]

        bounds = params["clip_bounds"].get(horizon, {"lower": 0, "upper": 5000})
        panel = float(np.clip(panel, bounds["lower"], bounds["upper"]))
        if np.isnan(panel):
            panel = catboost if catboost is not None else seasonal_ptf

        rows.append(
            {
                "issue_datetime": cutoff,
                "target_datetime": target_dt,
                "forecast_horizon": horizon,
                "interim_ptf": interim_at_cutoff,
                "seasonal_ptf": seasonal_ptf,
                "naive_ptf": seasonal_ptf,
                "catboost_ptf": catboost,
                "ensemble_ptf": panel,
                "panel_ptf": panel,
                "stacked_ptf": stacked,
                "primary_model": primary,
            }
        )

    bundle = pd.DataFrame(rows)
    LIVE_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_csv(LIVE_BUNDLE_PATH, index=False)
    return bundle


def _read_catboost_live(cutoff: pd.Timestamp) -> pd.DataFrame:
    if not CATBOOST_LIVE_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(CATBOOST_LIVE_PATH)
    for col in ("issue_datetime", "target_datetime"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    latest = data[data["issue_datetime"] == cutoff] if "issue_datetime" in data.columns else data
    return latest.sort_values("forecast_horizon")


def _catboost_price(catboost_live: pd.DataFrame, horizon: int) -> float | None:
    if catboost_live.empty:
        return None
    row = catboost_live[catboost_live["forecast_horizon"] == horizon]
    if row.empty:
        return None
    return max(0.0, float(row["predicted_ptf"].iloc[0]))


def _load_ensemble_params() -> dict:
    empty: dict = {
        "weights": {},
        "blend_weights": {},
        "biases": {},
        "clip_bounds": {},
        "primary_model": {},
    }
    if not FINAL_METRICS_PATH.exists():
        return empty

    payload = json.loads(FINAL_METRICS_PATH.read_text(encoding="utf-8"))
    empty["weights"] = {int(k): v for k, v in payload.get("ensemble_weights", {}).items()}
    empty["blend_weights"] = {int(k): float(v) for k, v in payload.get("blend_weights", {}).items()}
    empty["biases"] = {int(k): float(v) for k, v in payload.get("bias_corrections", {}).items()}
    empty["clip_bounds"] = {int(k): v for k, v in payload.get("clip_bounds", {}).items()}
    empty["primary_model"] = {int(k): str(v) for k, v in payload.get("primary_model_by_horizon", {}).items()}
    return empty
