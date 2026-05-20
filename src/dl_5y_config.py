from __future__ import annotations

from datetime import date

from src.config import PROJECT_ROOT, TIMEZONE, get_settings


START_DATE_5Y = date(2020, 1, 1)


def end_date_5y() -> date:
    return get_settings().end_date


RAW_5Y_DIR = PROJECT_ROOT / "data" / "raw" / "epias_5y"
PROCESSED_5Y_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_5Y_DIR = PROJECT_ROOT / "data" / "models" / "dl_5y"

HOURLY_5Y_PATH = PROCESSED_5Y_DIR / "final_hourly_dataset_5y.csv"
SEQUENCE_NPZ_PATH = PROCESSED_5Y_DIR / "forecast_12h_sequence_dataset_5y.npz"
TABULAR_5Y_PATH = PROCESSED_5Y_DIR / "forecast_12h_tabular_dataset_5y.csv"
QUALITY_REPORT_PATH = PROCESSED_5Y_DIR / "data_quality_report_5y.json"
FEATURE_SUMMARY_PATH = PROCESSED_5Y_DIR / "feature_summary_5y.csv"
SCALERS_5Y_PATH = MODELS_5Y_DIR / "scalers_5y.pkl"
DL_METRICS_5Y_PATH = PROCESSED_5Y_DIR / "dl_models_metrics_5y.json"
DL_COMPARISON_5Y_PATH = PROCESSED_5Y_DIR / "dl_models_comparison_5y.csv"

INPUT_WINDOW_5Y = 168
OUTPUT_HORIZON_5Y = 12
TARGET_COLUMN_5Y = "ptf_target_12h"  # tabular meta; sequence y = 12 future prices
PRICE_COLUMN_5Y = "ptf_price"  # kesinlesmis > interim

TRAIN_RATIO_5Y = 0.70
VAL_RATIO_5Y = 0.15  # of train+val block before test

EPSILON_5Y = 1.0
