#!/usr/bin/env python3
"""Update dashboard snapshot with latest ensemble predictions and metrics."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FINAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_metrics.json"
LIVE_BUNDLE_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_live_bundle.csv"
DASHBOARD_DATA_PATH = PROJECT_ROOT / "app_data" / "dashboard_data.json"


def update_dashboard_snapshot() -> None:
    if not FINAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Final metrics not found: {FINAL_METRICS_PATH}")

    with open(FINAL_METRICS_PATH, encoding="utf-8") as f:
        final_metrics = json.load(f)

    if not LIVE_BUNDLE_PATH.exists():
        from src.live_forecast import build_live_forecast_bundle

        build_live_forecast_bundle()

    live_bundle = pd.read_csv(LIVE_BUNDLE_PATH)
    for col in ("issue_datetime", "target_datetime"):
        if col in live_bundle.columns:
            live_bundle[col] = pd.to_datetime(live_bundle[col], errors="coerce")

    cutoff = live_bundle["issue_datetime"].max() if not live_bundle.empty else None

    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S +03"),
        "cutoff_datetime": str(cutoff) if cutoff is not None else "",
        "model_type": "ensemble",
        "metrics": {
            "MAE": final_metrics["MAE"],
            "RMSE": final_metrics["RMSE"],
            "SMAPE": final_metrics["SMAPE"],
            "R2": final_metrics["R2"],
            "baseline_mae": final_metrics["baseline_comparison"]["mae"],
            "baseline_rmse": final_metrics["baseline_comparison"]["rmse"],
            "baseline_smape": final_metrics["baseline_comparison"]["smape"],
            "baseline_r2": final_metrics["baseline_comparison"]["r2"],
            "mae_improvement_pct": (
                (final_metrics["baseline_comparison"]["mae"] - final_metrics["MAE"])
                / final_metrics["baseline_comparison"]["mae"]
                * 100
            ),
            "ensemble_weights": final_metrics.get("ensemble_weights"),
            "bias_corrections": final_metrics.get("bias_corrections"),
        },
        "live_forecast": live_bundle.to_dict(orient="records"),
    }

    DASHBOARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, default=str)

    print("Dashboard data updated successfully")
    print(f"Live forecast cutoff: {cutoff}")
    print(f"Live forecast rows: {len(live_bundle)}")
    print(f"Output: {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    update_dashboard_snapshot()
