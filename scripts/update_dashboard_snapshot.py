#!/usr/bin/env python3
"""Dashboard snapshot — 5y best veya legacy final ensemble."""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.panel_data import (
    DASHBOARD_DATA_PATH,
    get_active_panel_sources,
    load_active_metrics,
    load_leaderboard,
)


def _live_rows_from_best() -> pd.DataFrame:
    from src.best_ptf_config import BEST_LIVE_PATH, FORECAST_BEST_PATH, CATBOOST_BEST_MODEL_PATH
    import src.build_12h_forecast_dataset as b12
    import src.train_ptf_12h_forecast as tr
    from src.build_best_ptf_pipeline import BRIDGE_HOURLY
    from src.train_ptf_12h_forecast import predict_live_next_12h

    if not (FORECAST_BEST_PATH.exists() and CATBOOST_BEST_MODEL_PATH.exists() and BRIDGE_HOURLY.exists()):
        return pd.DataFrame()

    b12.INPUT_PATH = BRIDGE_HOURLY
    tr.DATA_PATH = FORECAST_BEST_PATH
    tr.MODEL_BUNDLE_PATH = CATBOOST_BEST_MODEL_PATH
    tr.LIVE_FORECAST_PATH = BEST_LIVE_PATH
    live = predict_live_next_12h()
    if not live.empty:
        live.to_csv(BEST_LIVE_PATH, index=False)
        return live
    if BEST_LIVE_PATH.exists():
        return pd.read_csv(BEST_LIVE_PATH)
    return pd.DataFrame()


def _live_rows_from_legacy() -> pd.DataFrame:
    from src.live_forecast import build_live_forecast_bundle

    bundle_path = PROJECT_ROOT / "data" / "processed" / "ptf_12h_live_bundle.csv"
    if not bundle_path.exists():
        build_live_forecast_bundle()
    if bundle_path.exists():
        return pd.read_csv(bundle_path)
    return pd.DataFrame()


def update_dashboard_snapshot(*, refresh_live: bool = False) -> None:
    src = get_active_panel_sources()
    metrics = load_active_metrics()
    if not metrics:
        raise FileNotFoundError(f"Metrik dosyası yok: {src.metrics_path}")

    if refresh_live and src.is_best_5y:
        live_bundle = _live_rows_from_best()
    elif refresh_live:
        live_bundle = _live_rows_from_legacy()
    else:
        live_path = src.live_forecast_path
        if live_path.exists():
            live_bundle = pd.read_csv(live_path)
        else:
            live_bundle = _live_rows_from_best() if src.is_best_5y else _live_rows_from_legacy()

    for col in ("issue_datetime", "target_datetime"):
        if col in live_bundle.columns:
            live_bundle[col] = pd.to_datetime(live_bundle[col], errors="coerce")

    cutoff = live_bundle["issue_datetime"].max() if not live_bundle.empty else None
    board = load_leaderboard()

    baseline = metrics.get("baseline_comparison", {})
    if not baseline and board:
        baseline = {
            "mae": board.get("ensemble_metrics", {}).get("baseline_comparison", {}).get("mae", 564),
        }

    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S +03"),
        "cutoff_datetime": str(cutoff) if cutoff is not None else "",
        "model_type": metrics.get("model", src.label),
        "active_source": src.label,
        "is_best_5y": src.is_best_5y,
        "metrics": {
            "MAE": metrics.get("MAE", 0),
            "RMSE": metrics.get("RMSE", 0),
            "SMAPE": metrics.get("SMAPE", 0),
            "R2": metrics.get("R2", 0),
            "baseline_mae": baseline.get("mae", board.get("catboost_mae", 0)),
            "baseline_rmse": baseline.get("rmse", 0),
            "baseline_smape": baseline.get("smape", 0),
            "baseline_r2": baseline.get("r2", 0),
            "mae_improvement_pct": (
                (baseline.get("mae", 564) - metrics.get("MAE", 0)) / baseline.get("mae", 564) * 100
                if baseline.get("mae")
                else 0
            ),
            "ensemble_weights": metrics.get("ensemble_weights"),
            "bias_corrections": metrics.get("bias_corrections"),
        },
        "leaderboard": board,
        "live_forecast": live_bundle.to_dict(orient="records"),
        "primary_model_by_horizon": metrics.get("primary_model_by_horizon", {}),
        "predictions_path": str(src.predictions_path),
    }

    DASHBOARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DATA_PATH.write_text(
        json.dumps(dashboard_data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    sample_path = PROJECT_ROOT / "app_data" / "ptf_12h_best_predictions_sample.csv"
    if src.predictions_path.exists():
        preds = pd.read_csv(src.predictions_path)
        if "target_datetime" in preds.columns:
            preds["target_datetime"] = pd.to_datetime(preds["target_datetime"], errors="coerce")
            end = preds["target_datetime"].max()
            window = preds[preds["target_datetime"] >= end - pd.Timedelta(days=45)]
            window.to_csv(sample_path, index=False)
            print(f"  Cloud örnek tahmin: {len(window)} satır -> {sample_path.name}")

    print("Dashboard güncellendi")
    print(f"  Kaynak: {src.label}")
    print(f"  MAE: {metrics.get('MAE', 0):.1f}")
    print(f"  Canlı kesit: {cutoff}")
    print(f"  Çıktı: {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-live", action="store_true", help="Canlı 12h tahmini yeniden üret")
    args = parser.parse_args()
    update_dashboard_snapshot(refresh_live=args.refresh_live)
