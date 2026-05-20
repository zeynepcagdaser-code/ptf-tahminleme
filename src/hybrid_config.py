from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT


PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "data" / "models"

FORECAST_12H_PATH = PROCESSED / "forecast_12h_dataset.csv"
HOURLY_PATH = PROCESSED / "final_hourly_dataset.csv"
CATBOOST_PRED_PATH = PROCESSED / "ptf_12h_predictions.csv"

SPIKE_MODEL_PATH = MODELS / "ptf_12h_spike_classifier.pkl"
SPIKE_PRED_PATH = PROCESSED / "ptf_12h_spike_predictions.csv"
SPIKE_METRICS_PATH = PROCESSED / "ptf_12h_spike_metrics.json"

HYBRID_GRU_MODEL_PATH = MODELS / "ptf_12h_hybrid_gru.pt"
HYBRID_GRU_SCALERS_PATH = MODELS / "ptf_12h_hybrid_gru_scalers.json"
HYBRID_GRU_PRED_PATH = PROCESSED / "ptf_12h_hybrid_gru_predictions.csv"
HYBRID_GRU_METRICS_PATH = PROCESSED / "ptf_12h_hybrid_gru_metrics.json"
HYBRID_GRU_HISTORY_PATH = PROCESSED / "ptf_12h_hybrid_gru_history.csv"

HYBRID_PRED_PATH = PROCESSED / "ptf_12h_hybrid_predictions.csv"
HYBRID_METRICS_PATH = PROCESSED / "ptf_12h_hybrid_metrics.json"
HYBRID_HORIZON_METRICS_PATH = PROCESSED / "ptf_12h_hybrid_horizon_metrics.csv"
HYBRID_CALIBRATED_METRICS_PATH = PROCESSED / "ptf_12h_hybrid_calibrated_metrics.json"
HYBRID_CALIBRATION_CHART_PATH = PROCESSED / "ptf_12h_hybrid_calibration_chart.png"

HORIZON = 12
EPSILON = 1.0
TARGET_COLUMN = "ptf_target"
