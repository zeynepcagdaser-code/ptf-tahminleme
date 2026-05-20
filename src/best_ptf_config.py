from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT
from src.dl_5y_config import HOURLY_5Y_PATH


PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "data" / "models"

HOURLY_BEST_PATH = HOURLY_5Y_PATH
FORECAST_BEST_PATH = PROCESSED / "forecast_12h_dataset_best.csv"
CATBOOST_BEST_PRED_PATH = PROCESSED / "ptf_12h_predictions_best.csv"
CATBOOST_BEST_METRICS_PATH = PROCESSED / "ptf_12h_metrics_best.json"
CATBOOST_BEST_MODEL_PATH = MODELS / "ptf_12h_horizon_models_best.pkl"

BEST_PREDICTIONS_PATH = PROCESSED / "ptf_12h_best_predictions.csv"
BEST_METRICS_PATH = PROCESSED / "ptf_12h_best_metrics.json"
BEST_HORIZON_METRICS_PATH = PROCESSED / "ptf_12h_best_horizon_metrics.csv"
BEST_COMPARISON_PATH = PROCESSED / "ptf_12h_best_model_comparison.csv"
BEST_LIVE_PATH = PROCESSED / "ptf_12h_best_live_forecast.csv"

HYBRID_BEST_PRED_PATH = PROCESSED / "ptf_12h_best_hybrid_predictions.csv"
HYBRID_BEST_METRICS_PATH = PROCESSED / "ptf_12h_best_hybrid_metrics.json"
