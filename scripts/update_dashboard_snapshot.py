#!/usr/bin/env python3
"""Update dashboard snapshot with latest ensemble predictions and metrics."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FINAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_metrics.json"
FINAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_predictions.csv"
DASHBOARD_DATA_PATH = PROJECT_ROOT / "app_data" / "dashboard_data.json"


def update_dashboard_snapshot():
    """Generate dashboard_data.json from final ensemble predictions and metrics."""
    
    # Load final metrics
    if not FINAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Final metrics not found: {FINAL_METRICS_PATH}")
    
    with open(FINAL_METRICS_PATH) as f:
        final_metrics = json.load(f)
    
    # Load final predictions
    if not FINAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Final predictions not found: {FINAL_PREDICTIONS_PATH}")
    
    final_preds = pd.read_csv(FINAL_PREDICTIONS_PATH)
    final_preds["issue_datetime"] = pd.to_datetime(final_preds["issue_datetime"], errors="coerce")
    
    # Get latest issue datetime for live forecast
    latest_issue = final_preds["issue_datetime"].max()
    live_forecast = final_preds[final_preds["issue_datetime"] == latest_issue].sort_values("forecast_horizon")
    
    # Create dashboard data
    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S +03"),
        "cutoff_datetime": str(latest_issue),
        "model_type": "ensemble",
        "metrics": {
            "MAE": final_metrics["MAE"],
            "RMSE": final_metrics["RMSE"],
            "SMAPE": final_metrics["SMAPE"],
            "R2": final_metrics["R2"],
            "rows": len(final_preds),
            "baseline_mae": final_metrics["baseline_comparison"]["mae"],
            "baseline_rmse": final_metrics["baseline_comparison"]["rmse"],
            "baseline_smape": final_metrics["baseline_comparison"]["smape"],
            "baseline_r2": final_metrics["baseline_comparison"]["r2"],
            "mae_improvement_pct": (
                (final_metrics["baseline_comparison"]["mae"] - final_metrics["MAE"])
                / final_metrics["baseline_comparison"]["mae"]
                * 100
            ),
            "ensemble_weights": final_metrics["ensemble_weights"],
            "bias_corrections": final_metrics["bias_corrections"],
        },
        "live_forecast": live_forecast[["issue_datetime", "target_datetime", "forecast_horizon", "final_predicted_ptf"]]
        .rename(columns={"final_predicted_ptf": "predicted_ptf"})
        .to_dict(orient="records"),
    }
    
    # Save dashboard data
    DASHBOARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_DATA_PATH, "w") as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"Dashboard data updated successfully")
    print(f"Live forecast cutoff: {latest_issue}")
    print(f"Live forecast rows: {len(live_forecast)}")
    print(f"Output: {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    update_dashboard_snapshot()
