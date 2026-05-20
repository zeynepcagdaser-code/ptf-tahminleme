from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.best_ptf_config import (
    BEST_COMPARISON_PATH,
    BEST_HORIZON_METRICS_PATH,
    BEST_LIVE_PATH,
    BEST_METRICS_PATH,
    BEST_PREDICTIONS_PATH,
    CATBOOST_BEST_METRICS_PATH,
    CATBOOST_BEST_MODEL_PATH,
    CATBOOST_BEST_PRED_PATH,
    FORECAST_BEST_PATH,
    HOURLY_BEST_PATH,
    HYBRID_BEST_METRICS_PATH,
    HYBRID_BEST_PRED_PATH,
    PROCESSED,
)
from src.build_12h_forecast_dataset import build_12h_forecast_dataset
from src.build_final_ensemble import build_final_ensemble
from src.build_hybrid_forecast import build_hybrid_forecast
from src.dl_5y_config import HOURLY_5Y_PATH, SEQUENCE_NPZ_PATH
from src.hybrid_volatility_calibration import build_and_calibrate_hybrid, evaluate_calibration, save_calibration_artifacts
from src.model_splits import chronological_train_val_test_split
from src.ptf_feature_enrichment import enrich_forecast_features
from src.ptf_seasonal_baselines import prepare_kesin_hourly, seasonal_predictions_for_target
from src.train_ptf_12h_forecast import train_ptf_12h_forecast
from src.train_spike_classifier import train_spike_classifier
from src.train_lstm_hybrid import train_lstm_hybrid


BRIDGE_HOURLY = PROCESSED / "final_hourly_dataset_best_bridge.csv"


@dataclass(frozen=True)
class BestPtfSummary:
    catboost_mae: float
    ensemble_mae: float
    hybrid_mae: float
    best_track: str
    best_mae: float
    metrics_path: str
    predictions_path: str


def _prepare_hourly_bridge() -> Path:
    if not HOURLY_BEST_PATH.exists():
        raise FileNotFoundError(f"5y saatlik veri yok: {HOURLY_BEST_PATH}")

    df = pd.read_csv(HOURLY_BEST_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    if "ptf" not in df.columns:
        df["ptf"] = pd.to_numeric(df.get("ptf_price"), errors="coerce")
    price = pd.to_numeric(df.get("ptf_kesinlesmis", df.get("ptf_price", df.get("ptf"))), errors="coerce")
    df["ptf_kesinlesmis"] = price
    df["ptf"] = pd.to_numeric(df.get("ptf", price), errors="coerce").fillna(price)

    for col in (
        "smf",
        "real_time_consumption",
        "wind_generation",
        "solar_generation",
        "hydro_dam_generation",
        "gop_fiyattan_bagimsiz_alis",
        "gop_fiyattan_bagimsiz_satis",
        "price_independent_buy_sell_ratio",
        "load_forecast_plan",
        "grf_tl",
        "usd_try",
    ):
        if col not in df.columns:
            df[col] = np.nan

    BRIDGE_HOURLY.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BRIDGE_HOURLY, index=False)
    return BRIDGE_HOURLY


def _patch_paths_for_best() -> dict[str, Path]:
    import src.build_12h_forecast_dataset as b12
    import src.train_ptf_12h_forecast as tr
    import src.build_final_ensemble as ens
    import src.build_hybrid_forecast as hyb
    import src.hybrid_config as hc
    import src.train_spike_classifier as sp
    import src.train_lstm_hybrid as gru

    bridge = _prepare_hourly_bridge()
    originals = {
        "b12_in": b12.INPUT_PATH,
        "b12_out": b12.OUTPUT_PATH,
        "tr_data": tr.DATA_PATH,
        "tr_pred": tr.PREDICTIONS_PATH,
        "tr_met": tr.METRICS_PATH,
        "tr_model": tr.MODEL_BUNDLE_PATH,
        "ens_fc": ens.FORECAST_DATA_PATH,
        "ens_hr": ens.HOURLY_DATA_PATH,
        "ens_cb": ens.CATBOOST_PREDICTIONS_PATH,
        "ens_out": ens.FINAL_PREDICTIONS_PATH,
        "ens_met": ens.FINAL_METRICS_PATH,
        "ens_hor": ens.FINAL_HORIZON_METRICS_PATH,
        "ens_cmp": ens.FINAL_COMPARISON_PATH,
        "hyb_fc": hc.FORECAST_12H_PATH,
        "hyb_hr": hc.HOURLY_PATH,
        "hyb_cb": hc.CATBOOST_PRED_PATH,
        "hyb_out": hc.HYBRID_PRED_PATH,
        "hyb_met": hc.HYBRID_METRICS_PATH,
        "hyb_spike": hc.SPIKE_PRED_PATH,
        "hyb_gru": hc.HYBRID_GRU_PRED_PATH,
    }

    b12.INPUT_PATH = bridge
    b12.OUTPUT_PATH = FORECAST_BEST_PATH
    tr.DATA_PATH = FORECAST_BEST_PATH
    tr.PREDICTIONS_PATH = CATBOOST_BEST_PRED_PATH
    tr.METRICS_PATH = CATBOOST_BEST_METRICS_PATH
    tr.MODEL_BUNDLE_PATH = CATBOOST_BEST_MODEL_PATH
    ens.FORECAST_DATA_PATH = FORECAST_BEST_PATH
    ens.HOURLY_DATA_PATH = bridge
    ens.CATBOOST_PREDICTIONS_PATH = CATBOOST_BEST_PRED_PATH
    ens.FINAL_PREDICTIONS_PATH = BEST_PREDICTIONS_PATH
    ens.FINAL_METRICS_PATH = BEST_METRICS_PATH
    ens.FINAL_HORIZON_METRICS_PATH = BEST_HORIZON_METRICS_PATH
    ens.FINAL_COMPARISON_PATH = BEST_COMPARISON_PATH
    hc.FORECAST_12H_PATH = FORECAST_BEST_PATH
    hc.HOURLY_PATH = bridge
    hc.CATBOOST_PRED_PATH = CATBOOST_BEST_PRED_PATH
    hc.HYBRID_PRED_PATH = HYBRID_BEST_PRED_PATH
    hc.HYBRID_METRICS_PATH = HYBRID_BEST_METRICS_PATH
    hc.SPIKE_PRED_PATH = PROCESSED / "ptf_12h_spike_predictions_best.csv"
    hc.HYBRID_GRU_PRED_PATH = PROCESSED / "ptf_12h_hybrid_gru_predictions_best.csv"

    return originals


def _restore_paths(originals: dict[str, Path]) -> None:
    import src.build_12h_forecast_dataset as b12
    import src.train_ptf_12h_forecast as tr
    import src.build_final_ensemble as ens
    import src.hybrid_config as hc

    b12.INPUT_PATH = originals["b12_in"]
    b12.OUTPUT_PATH = originals["b12_out"]
    tr.DATA_PATH = originals["tr_data"]
    tr.PREDICTIONS_PATH = originals["tr_pred"]
    tr.METRICS_PATH = originals["tr_met"]
    tr.MODEL_BUNDLE_PATH = originals["tr_model"]
    ens.FORECAST_DATA_PATH = originals["ens_fc"]
    ens.HOURLY_DATA_PATH = originals["ens_hr"]
    ens.CATBOOST_PREDICTIONS_PATH = originals["ens_cb"]
    ens.FINAL_PREDICTIONS_PATH = originals["ens_out"]
    ens.FINAL_METRICS_PATH = originals["ens_met"]
    ens.FINAL_HORIZON_METRICS_PATH = originals["ens_hor"]
    ens.FINAL_COMPARISON_PATH = originals["ens_cmp"]
    hc.FORECAST_12H_PATH = originals["hyb_fc"]
    hc.HOURLY_PATH = originals["hyb_hr"]
    hc.CATBOOST_PRED_PATH = originals["hyb_cb"]
    hc.HYBRID_PRED_PATH = originals["hyb_out"]
    hc.HYBRID_METRICS_PATH = originals["hyb_met"]
    hc.SPIKE_PRED_PATH = originals["hyb_spike"]
    hc.HYBRID_GRU_PRED_PATH = originals["hyb_gru"]


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.maximum(y_true.astype(float), 0.0)
    y_pred = np.maximum(y_pred.astype(float), 0.0)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred) + 1e-6, 1e-6)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "R2": r2}


def _build_live_best() -> None:
    import src.train_ptf_12h_forecast as tr
    from src.train_ptf_12h_forecast import _feature_columns, _inverse_transform_prediction

    if not CATBOOST_BEST_MODEL_PATH.exists():
        return
    with CATBOOST_BEST_MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)
    data = enrich_forecast_features(pd.read_csv(FORECAST_BEST_PATH))
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"])
    data["target_datetime"] = pd.to_datetime(data["target_datetime"])
    cutoff = data["issue_datetime"].max()
    feature_columns = _feature_columns(data)

    rows = []
    for horizon in range(1, 13):
        model = bundle.get(horizon)
        if model is None:
            continue
        sub = data[(data["issue_datetime"] == cutoff) & (data["forecast_horizon"] == horizon)]
        if sub.empty:
            continue
        raw = model.predict(sub[feature_columns])
        pred = _inverse_transform_prediction(sub, raw)
        for i, (_, r) in enumerate(sub.iterrows()):
            rows.append(
                {
                    "issue_datetime": cutoff,
                    "target_datetime": r["target_datetime"],
                    "forecast_horizon": horizon,
                    "predicted_ptf": float(pred[i]),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(BEST_LIVE_PATH, index=False)


def run_best_ptf_pipeline(*, include_hybrid: bool = True, include_spike_gru: bool = True) -> BestPtfSummary:
    originals = _patch_paths_for_best()
    try:
        print("=== 1/5 12h forecast dataset (5y veri) ===")
        _, ds_summary = build_12h_forecast_dataset()
        print(f"   {ds_summary.rows:,} ornek -> {ds_summary.output_path}")

        print("=== 2/5 CatBoost (12 ufuk, tam tuning) ===")
        cb = train_ptf_12h_forecast()
        print(f"   CatBoost MAE={cb.overall_mae:,.1f} R2={cb.overall_r2:.3f}")

        print("=== 3/5 Final ensemble ===")
        ens = build_final_ensemble()
        print(f"   Ensemble MAE={ens.overall_mae:,.1f} R2={ens.overall_r2:.3f}")

        hybrid_mae = float("nan")
        if include_hybrid:
            print("=== 4/5 Spike + GRU + Hybrid ===")
            if include_spike_gru:
                try:
                    train_spike_classifier()
                except Exception as exc:
                    print(f"   spike atlandi: {exc}")
                try:
                    train_lstm_hybrid()
                except Exception as exc:
                    print(f"   gru atlandi: {exc}")
            try:
                hyb = build_hybrid_forecast()
                hybrid_mae = hyb.overall_mae
                print(f"   Hybrid (kalibre) MAE={hyb.overall_mae:,.1f} R2={hyb.overall_r2:.3f}")
            except Exception as exc:
                print(f"   hybrid atlandi: {exc}")

        print("=== 5/5 Canli tahmin ozeti ===")
        try:
            from src.train_ptf_12h_forecast import predict_live_next_12h

            tr.DATA_PATH = FORECAST_BEST_PATH
            tr.MODEL_BUNDLE_PATH = CATBOOST_BEST_MODEL_PATH
            tr.LIVE_FORECAST_PATH = BEST_LIVE_PATH
            predict_live_next_12h()
            print(f"   -> {BEST_LIVE_PATH}")
        except Exception as exc:
            print(f"   live atlandi: {exc}")

        candidates = {
            "catboost": cb.overall_mae,
            "ensemble": ens.overall_mae,
        }
        if not np.isnan(hybrid_mae):
            candidates["hybrid"] = hybrid_mae

        best_track = min(candidates, key=candidates.get)
        best_mae = candidates[best_track]

        ens_metrics = {}
        if BEST_METRICS_PATH.exists():
            try:
                ens_metrics = json.loads(BEST_METRICS_PATH.read_text(encoding="utf-8"))
            except Exception:
                ens_metrics = {"MAE": ens.overall_mae, "R2": ens.overall_r2}

        payload = {
            "best_track": best_track,
            "best_mae": best_mae,
            "catboost_mae": cb.overall_mae,
            "catboost_r2": cb.overall_r2,
            "ensemble_mae": ens.overall_mae,
            "ensemble_r2": ens.overall_r2,
            "hybrid_mae": hybrid_mae,
            "ensemble_metrics": ens_metrics,
            "data_years": "2020-2024 (43800h)",
            "predictions_path": str(BEST_PREDICTIONS_PATH),
        }
        payload_path = PROCESSED / "ptf_12h_best_leaderboard.json"
        payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if best_track == "ensemble":
            final_pred = BEST_PREDICTIONS_PATH
        elif best_track == "hybrid" and HYBRID_BEST_PRED_PATH.exists():
            final_pred = HYBRID_BEST_PRED_PATH
        else:
            final_pred = CATBOOST_BEST_PRED_PATH

        selected = PROCESSED / "ptf_12h_best_selected_predictions.csv"
        shutil.copy2(final_pred, selected)

        best_metrics = {
            "MAE": cb.overall_mae if best_track == "catboost" else ens.overall_mae,
            "RMSE": cb.overall_rmse if best_track == "catboost" else ens.overall_rmse,
            "SMAPE": cb.overall_smape if best_track == "catboost" else ens.overall_smape,
            "R2": cb.overall_r2 if best_track == "catboost" else ens.overall_r2,
            "model": f"catboost_5y_{best_track}",
            "predictions_path": str(selected),
        }
        BEST_METRICS_PATH.write_text(json.dumps(best_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            from scripts.update_dashboard_snapshot import update_dashboard_snapshot

            update_dashboard_snapshot(refresh_live=True)
        except Exception as exc:
            print(f"Dashboard snapshot atlandi: {exc}")

        return BestPtfSummary(
            catboost_mae=cb.overall_mae,
            ensemble_mae=ens.overall_mae,
            hybrid_mae=hybrid_mae,
            best_track=best_track,
            best_mae=best_mae,
            metrics_path=str(PROCESSED / "ptf_12h_best_leaderboard.json"),
            predictions_path=str(final_pred),
        )
    finally:
        _restore_paths(originals)
