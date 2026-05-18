from __future__ import annotations

import pandas as pd

from src.config import PROJECT_ROOT
from src.model_splits import project_relative_path


CATBOOST_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_metrics_comparison.csv"
LSTM_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_lstm_metrics_comparison.csv"
ALL_MODELS_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_all_models_metrics.csv"


def write_combined_metrics_file() -> str:
    frames: list[pd.DataFrame] = []

    if CATBOOST_METRICS_PATH.exists():
        catboost = pd.read_csv(CATBOOST_METRICS_PATH)
        if "model" not in catboost.columns:
            catboost.insert(0, "model", "catboost")
        frames.append(catboost)

    if LSTM_METRICS_PATH.exists():
        lstm = pd.read_csv(LSTM_METRICS_PATH)
        if "model" not in lstm.columns:
            lstm.insert(0, "model", "lstm")
        frames.append(lstm)

    if not frames:
        return ""

    combined = pd.concat(frames, ignore_index=True)
    ALL_MODELS_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ALL_MODELS_METRICS_PATH, index=False)
    return project_relative_path(ALL_MODELS_METRICS_PATH)
