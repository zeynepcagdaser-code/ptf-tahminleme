from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.hybrid_config import (
    CATBOOST_PRED_PATH,
    FORECAST_12H_PATH,
    HOURLY_PATH,
    HYBRID_CALIBRATED_METRICS_PATH,
    HYBRID_HORIZON_METRICS_PATH,
    HYBRID_METRICS_PATH,
    HYBRID_PRED_PATH,
    HYBRID_GRU_PRED_PATH,
    SPIKE_PRED_PATH,
    TARGET_COLUMN,
)
from src.hybrid_volatility_calibration import (
    attach_volatility_features,
    build_and_calibrate_hybrid,
    evaluate_calibration,
    save_calibration_artifacts,
)
from src.model_splits import chronological_train_val_test_split, project_relative_path
from src.ptf_seasonal_baselines import prepare_kesin_hourly, seasonal_predictions_for_target


ID_COLS = ["issue_datetime", "target_datetime", "forecast_horizon", TARGET_COLUMN]
EPSILON = 1.0
TREND_PENALTY_LAMBDA = 0.20


@dataclass(frozen=True)
class HybridForecastSummary:
    overall_mae: float
    overall_rmse: float
    overall_smape: float
    overall_r2: float
    trend_correlation: float
    spike_hours_mae: float
    non_spike_hours_mae: float
    direction_accuracy: float
    raw_mae: float
    predictions_path: str
    metrics_path: str
    horizon_metrics_path: str
    calibration_chart_path: str
    calibrated_metrics_path: str


def build_hybrid_forecast() -> HybridForecastSummary:
    base = _read_forecast()
    hourly = prepare_kesin_hourly(_read_hourly())

    merged = base.merge(
        _read_catboost(),
        on=["issue_datetime", "target_datetime", "forecast_horizon"],
        how="left",
    )
    merged = merged.merge(
        _read_spike(),
        on=["issue_datetime", "target_datetime", "forecast_horizon"],
        how="left",
    )
    merged = merged.merge(
        _read_gru(),
        on=["issue_datetime", "target_datetime", "forecast_horizon"],
        how="left",
    )

    seasonal_rows = []
    for _, row in merged.iterrows():
        s = seasonal_predictions_for_target(hourly, row["issue_datetime"], row["target_datetime"])
        seasonal_rows.append(s)
    seasonal_df = pd.DataFrame(seasonal_rows)
    merged["pred_seasonal"] = seasonal_df["pred_seasonal_blend"].to_numpy()
    merged["spike_probability"] = merged["spike_probability"].fillna(0.0).clip(0.0, 1.0)
    merged["pred_catboost"] = merged["pred_catboost"].fillna(merged["pred_seasonal"])
    merged["pred_gru"] = merged["pred_gru"].fillna(merged["pred_catboost"])

    merged = attach_volatility_features(merged, hourly)

    blend_weights: dict[int, dict[str, float]] = {}
    for horizon in range(1, 13):
        hdata = merged[merged["forecast_horizon"] == horizon].sort_values("issue_datetime")
        _, val_df, _ = chronological_train_val_test_split(hdata)
        blend_weights[horizon] = _fit_blend_weights(val_df)

    final_df = build_and_calibrate_hybrid(merged, blend_weights)
    final_df["absolute_error"] = np.abs(final_df[TARGET_COLUMN] - final_df["hybrid_predicted_ptf"])
    final_df["absolute_error_raw"] = np.abs(final_df[TARGET_COLUMN] - final_df["hybrid_raw_predicted_ptf"])

    cal_metrics = evaluate_calibration(final_df)
    chart_path = save_calibration_artifacts(final_df, cal_metrics)
    cal = cal_metrics["calibrated"]
    raw = cal_metrics["raw"]

    _, _, test_df = chronological_train_val_test_split(final_df.sort_values("issue_datetime"))
    eval_df = test_df if len(test_df) else final_df
    horizon_metrics = _horizon_metrics(eval_df)
    horizon_metrics.to_csv(HYBRID_HORIZON_METRICS_PATH, index=False)
    final_df.to_csv(HYBRID_PRED_PATH, index=False)

    payload = {
        **cal,
        "blend_weights": blend_weights,
        "calibration": cal_metrics,
        "calibration_chart": str(chart_path),
    }
    HYBRID_METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return HybridForecastSummary(
        overall_mae=cal["MAE"],
        overall_rmse=cal["RMSE"],
        overall_smape=cal["SMAPE"],
        overall_r2=cal["R2"],
        trend_correlation=cal["trend_correlation"],
        spike_hours_mae=cal["spike_hours_mae"],
        non_spike_hours_mae=cal["non_spike_hours_mae"],
        direction_accuracy=cal["direction_accuracy"],
        raw_mae=raw["MAE"],
        predictions_path=project_relative_path(HYBRID_PRED_PATH),
        metrics_path=project_relative_path(HYBRID_METRICS_PATH),
        horizon_metrics_path=project_relative_path(HYBRID_HORIZON_METRICS_PATH),
        calibration_chart_path=project_relative_path(chart_path),
        calibrated_metrics_path=project_relative_path(HYBRID_CALIBRATED_METRICS_PATH),
    )


def _fit_blend_weights(val_df: pd.DataFrame) -> dict[str, float]:
    y = val_df[TARGET_COLUMN].to_numpy()
    cb = val_df["pred_catboost"].to_numpy()
    seas = val_df["pred_seasonal"].to_numpy()
    gru = val_df["pred_gru"].to_numpy()

    best_w = {"catboost": 0.45, "seasonal": 0.25, "gru": 0.30}
    best_score = float("inf")

    for w_cb in np.linspace(0.2, 0.6, 9):
        for w_gru in np.linspace(0.1, 0.5, 9):
            w_seas = 1.0 - w_cb - w_gru
            if w_seas < 0.05:
                continue
            pred = w_cb * cb + w_seas * seas + w_gru * gru
            score = _composite_objective(y, pred)
            if score < best_score:
                best_score = score
                best_w = {"catboost": float(w_cb), "seasonal": float(w_seas), "gru": float(w_gru)}

    return best_w


def _composite_objective(y: np.ndarray, pred: np.ndarray) -> float:
    mae = mean_absolute_error(y, pred)
    trend_pen = max(0.0, 1.0 - _trend_correlation(y, pred))
    return mae + TREND_PENALTY_LAMBDA * trend_pen * max(float(np.std(y)), 50.0)


def _trend_correlation(y: np.ndarray, pred: np.ndarray) -> float:
    dy = np.diff(y.astype(float))
    dp = np.diff(pred.astype(float))
    if len(dy) < 3 or np.std(dy) < 1e-6 or np.std(dp) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(dy, dp)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _horizon_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h, g in df.groupby("forecast_horizon"):
        m = _metrics(g[TARGET_COLUMN].to_numpy(), g["hybrid_predicted_ptf"].to_numpy())
        m["direction_accuracy"] = _direction_accuracy_grouped(g, "hybrid_predicted_ptf")
        m["trend_correlation"] = _trend_correlation(g[TARGET_COLUMN].to_numpy(), g["hybrid_predicted_ptf"].to_numpy())
        rows.append({"forecast_horizon": int(h), **m})
    return pd.DataFrame(rows).sort_values("forecast_horizon")


def _direction_accuracy_grouped(df: pd.DataFrame, pred_col: str) -> float:
    from src.hybrid_volatility_calibration import direction_accuracy

    return direction_accuracy(df, pred_col)


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum((np.abs(actual) + np.abs(pred)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "SMAPE": float(np.mean(np.abs(actual - pred) / denom) * 100),
        "R2": float(r2_score(actual, pred)),
    }


def _read_forecast() -> pd.DataFrame:
    data = pd.read_csv(FORECAST_12H_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN])


def _read_hourly() -> pd.DataFrame:
    data = pd.read_csv(HOURLY_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"])


def _read_catboost() -> pd.DataFrame:
    data = pd.read_csv(CATBOOST_PRED_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data[["issue_datetime", "target_datetime", "forecast_horizon", "predicted_ptf"]].rename(
        columns={"predicted_ptf": "pred_catboost"}
    )


def _read_spike() -> pd.DataFrame:
    data = pd.read_csv(SPIKE_PRED_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    cols = ["issue_datetime", "target_datetime", "forecast_horizon", "spike_probability"]
    if "is_spike" in data.columns:
        cols.append("is_spike")
    return data[cols]


def _read_gru() -> pd.DataFrame:
    data = pd.read_csv(HYBRID_GRU_PRED_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data[
        ["issue_datetime", "target_datetime", "forecast_horizon", "pred_gru"]
    ]
