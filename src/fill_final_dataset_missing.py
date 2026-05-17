from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


MISSING_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset_missing_report.csv"

HOURLY_COLUMNS = [
    "gop_fiyattan_bagimsiz_alis",
    "gop_fiyattan_bagimsiz_satis",
    "load_forecast_plan",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "real_time_consumption",
    "unlicensed_generation_total",
    "smf",
]

DAILY_BROADCAST_COLUMNS = ["grf_tl", "usd_try"]
RATIO_COLUMN = "price_independent_buy_sell_ratio"


def fill_final_dataset_missing(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filled = dataset.copy()
    filled["datetime"] = pd.to_datetime(filled["datetime"], errors="coerce")
    filled = filled.sort_values("datetime").reset_index(drop=True)
    before_missing = filled.isna().sum()

    indexed = filled.set_index("datetime")

    for column in HOURLY_COLUMNS:
        if column not in indexed.columns or column == "ptf":
            continue
        indexed[column] = pd.to_numeric(indexed[column], errors="coerce")
        indexed[column] = indexed[column].interpolate(method="time", limit_direction="both")
        indexed[column] = indexed[column].ffill().bfill()
        indexed[column] = _fill_with_median(indexed[column])

    for column in DAILY_BROADCAST_COLUMNS:
        if column not in indexed.columns:
            continue
        indexed[column] = pd.to_numeric(indexed[column], errors="coerce").ffill().bfill()

    if RATIO_COLUMN in indexed.columns:
        indexed[RATIO_COLUMN] = pd.to_numeric(indexed[RATIO_COLUMN], errors="coerce")
        indexed[RATIO_COLUMN] = indexed[RATIO_COLUMN].replace([np.inf, -np.inf], np.nan)
        indexed[RATIO_COLUMN] = _fill_with_median(indexed[RATIO_COLUMN])

    filled = indexed.reset_index()
    after_missing = filled.isna().sum()

    report = pd.DataFrame(
        {
            "column": filled.columns,
            "missing_count_before": [int(before_missing.get(column, 0)) for column in filled.columns],
            "missing_ratio_before": [
                float(before_missing.get(column, 0) / len(filled)) if len(filled) else 0.0
                for column in filled.columns
            ],
            "missing_count_after": [int(after_missing.get(column, 0)) for column in filled.columns],
            "missing_ratio_after": [
                float(after_missing.get(column, 0) / len(filled)) if len(filled) else 0.0
                for column in filled.columns
            ],
        }
    )
    MISSING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(MISSING_REPORT_PATH, index=False)
    return filled, report


def _fill_with_median(series: pd.Series) -> pd.Series:
    median = series.median(skipna=True)
    if pd.isna(median):
        return series
    return series.fillna(median)
