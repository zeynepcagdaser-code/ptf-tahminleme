"""Streamlit panel — aktif tahmin kaynağı (5y best öncelikli)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.best_ptf_config import (
    BEST_HORIZON_METRICS_PATH,
    BEST_LIVE_PATH,
    BEST_METRICS_PATH,
    BEST_PREDICTIONS_PATH,
    CATBOOST_BEST_METRICS_PATH,
    CATBOOST_BEST_PRED_PATH,
    HOURLY_BEST_PATH,
    PROCESSED,
)
from src.config import PROJECT_ROOT


LEGACY_METRICS_PATH = PROCESSED / "ptf_12h_final_metrics.json"
LEGACY_PREDICTIONS_PATH = PROCESSED / "ptf_12h_final_predictions.csv"
LEGACY_HORIZON_PATH = PROCESSED / "ptf_12h_final_horizon_metrics.csv"
LEGACY_LIVE_PATH = PROCESSED / "ptf_12h_live_forecast.csv"
LEGACY_LIVE_BUNDLE_PATH = PROCESSED / "ptf_12h_live_bundle.csv"
LEGACY_DATASET_PATH = PROCESSED / "final_hourly_dataset.csv"
DASHBOARD_DATA_PATH = PROJECT_ROOT / "app_data" / "dashboard_data.json"
CLOUD_PREDICTIONS_SAMPLE = PROJECT_ROOT / "app_data" / "ptf_12h_best_predictions_sample.csv"
LEADERBOARD_PATH = PROCESSED / "ptf_12h_best_leaderboard.json"


@dataclass(frozen=True)
class PanelDataSources:
    label: str
    metrics_path: Path
    predictions_path: Path
    horizon_metrics_path: Path
    live_forecast_path: Path
    hourly_dataset_path: Path
    is_best_5y: bool


def get_active_panel_sources() -> PanelDataSources:
    """5y CatBoost best varsa onu kullan, yoksa legacy final ensemble."""
    if CATBOOST_BEST_PRED_PATH.exists() and CATBOOST_BEST_METRICS_PATH.exists():
        return PanelDataSources(
            label="CatBoost 5Y (2020–2024, MAE ~291)",
            metrics_path=CATBOOST_BEST_METRICS_PATH,
            predictions_path=CATBOOST_BEST_PRED_PATH,
            horizon_metrics_path=BEST_HORIZON_METRICS_PATH
            if BEST_HORIZON_METRICS_PATH.exists()
            else CATBOOST_BEST_PRED_PATH,
            live_forecast_path=BEST_LIVE_PATH,
            hourly_dataset_path=HOURLY_BEST_PATH,
            is_best_5y=True,
        )
    if BEST_PREDICTIONS_PATH.exists() and BEST_METRICS_PATH.exists():
        return PanelDataSources(
            label="Ensemble 5Y",
            metrics_path=BEST_METRICS_PATH,
            predictions_path=BEST_PREDICTIONS_PATH,
            horizon_metrics_path=BEST_HORIZON_METRICS_PATH,
            live_forecast_path=BEST_LIVE_PATH,
            hourly_dataset_path=HOURLY_BEST_PATH,
            is_best_5y=True,
        )
    return PanelDataSources(
        label="Final Ensemble (legacy)",
        metrics_path=LEGACY_METRICS_PATH,
        predictions_path=LEGACY_PREDICTIONS_PATH,
        horizon_metrics_path=LEGACY_HORIZON_PATH,
        live_forecast_path=LEGACY_LIVE_PATH,
        hourly_dataset_path=LEGACY_DATASET_PATH,
        is_best_5y=False,
    )


def load_active_metrics() -> dict:
    src = get_active_panel_sources()
    if not src.metrics_path.exists():
        return {}
    metrics = json.loads(src.metrics_path.read_text(encoding="utf-8"))
    if "baseline_comparison" not in metrics:
        board = load_leaderboard()
        base = board.get("ensemble_metrics", {}).get("baseline_comparison", {})
        if base:
            metrics["baseline_comparison"] = base
        elif board.get("catboost_mae"):
            metrics["baseline_comparison"] = {
                "mae": board.get("ensemble_mae", 564),
                "rmse": 0,
                "smape": 0,
                "r2": board.get("ensemble_r2", 0),
            }
    metrics.setdefault("model", "catboost_5y" if src.is_best_5y else "final_ensemble")
    metrics["panel_label"] = src.label
    return metrics


def load_leaderboard() -> dict:
    if not LEADERBOARD_PATH.exists():
        return {}
    try:
        return json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
