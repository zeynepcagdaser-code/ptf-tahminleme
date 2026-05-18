from __future__ import annotations

from src.build_12h_forecast_dataset import INPUT_PATH, build_12h_forecast_dataset
from src.build_final_hourly_dataset import build_final_hourly_dataset
from src.fill_final_dataset_missing import fill_final_dataset_missing
from src.train_ptf_12h_forecast import train_ptf_12h_forecast


def run_full_ptf_pipeline(*, fetch_live: bool = False) -> dict:
    if fetch_live:
        from src.fetch_final_selected_features import run_fetch_final_selected_features

        run_fetch_final_selected_features()

    raw_dataset, _ = build_final_hourly_dataset()
    final_dataset, _ = fill_final_dataset_missing(raw_dataset)

    INPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(INPUT_PATH, index=False)

    _, dataset_summary = build_12h_forecast_dataset()
    training_summary = train_ptf_12h_forecast()

    return {
        "dataset_rows": dataset_summary.rows,
        "dataset_path": dataset_summary.output_path,
        "mae": training_summary.overall_mae,
        "rmse": training_summary.overall_rmse,
        "smape": training_summary.overall_smape,
        "r2": training_summary.overall_r2,
        "predictions_path": training_summary.predictions_path,
        "metrics_path": training_summary.metrics_path,
        "horizon_metrics_path": training_summary.horizon_metrics_path,
        "live_forecast_path": training_summary.live_forecast_path,
    }
