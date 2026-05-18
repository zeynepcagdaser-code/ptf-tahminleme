from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ptf_seasonal_baselines import prepare_kesin_hourly, seasonal_predictions_for_target

from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path


FORECAST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_12h_dataset.csv"
HOURLY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
CATBOOST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_predictions.csv"

FINAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_predictions.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_metrics.json"
FINAL_HORIZON_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_horizon_metrics.csv"
FINAL_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_model_comparison.csv"

BASELINE_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_metrics.json"

TARGET_COLUMN = "ptf_target"
ID_COLUMNS = ["issue_datetime", "target_datetime", "forecast_horizon", TARGET_COLUMN]
EPSILON = 1.0

# CatBoost + kesinlesmis mevsimsel kaynaklar (yinelenen kolon yok)
ENSEMBLE_SOURCE_COLS = (
    "pred_catboost",
    "pred_seasonal_blend",
    "pred_kesin_same_hour_yesterday",
    "pred_kesin_same_hour_last_week",
    "pred_kesin_rolling_168h",
)


@dataclass(frozen=True)
class FinalEnsembleSummary:
    overall_mae: float
    overall_rmse: float
    overall_smape: float
    overall_r2: float
    baseline_mae: float
    baseline_rmse: float
    baseline_smape: float
    baseline_r2: float
    mae_improvement: float
    rmse_improvement: float
    smape_improvement: float
    r2_improvement: float
    predictions_path: str
    metrics_path: str
    horizon_metrics_path: str
    comparison_path: str


def build_final_ensemble() -> FinalEnsembleSummary:
    # Load baseline metrics
    baseline_metrics = _load_baseline_metrics()
    
    # Load data
    forecast_data = _read_forecast_data()
    hourly_data = _read_hourly_data()
    catboost_preds = _read_catboost_predictions()
    
    # Create simple baseline predictions
    simple_preds = _create_simple_baselines(forecast_data, hourly_data)
    
    # Merge all predictions
    all_preds = forecast_data[ID_COLUMNS].copy()
    if "ptf_lag_1" in forecast_data.columns:
        all_preds["pred_persistence"] = forecast_data["ptf_lag_1"]
    if "kesin_seasonal_anchor" in forecast_data.columns:
        all_preds["pred_dataset_seasonal"] = forecast_data["kesin_seasonal_anchor"]
    all_preds = all_preds.merge(catboost_preds, on=ID_COLUMNS[:3], how="left")
    all_preds = all_preds.merge(simple_preds, on=ID_COLUMNS[:3], how="left")
    
    ensemble_weights: dict[int, dict[str, float]] = {}
    blend_weights: dict[int, float] = {}
    bias_corrections: dict[int, float] = {}
    clip_bounds: dict[int, dict[str, float]] = {}

    for horizon in range(1, 13):
        print(f"\n=== Optimizing ensemble for horizon {horizon}h ===")
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        horizon_data = horizon_data.sort_values("issue_datetime").reset_index(drop=True)

        _, val_df, _ = chronological_train_val_test_split(horizon_data)
        pred_cols = [c for c in ENSEMBLE_SOURCE_COLS if c in horizon_data.columns]

        weights, bias = _fit_stacked_ensemble(val_df, pred_cols)
        catboost_w, blend_bias = _fit_catboost_seasonal_blend(val_df)
        ensemble_weights[horizon] = weights
        blend_weights[horizon] = catboost_w
        bias_corrections[horizon] = blend_bias

        val_hybrid = np.nan_to_num(_hybrid_prediction(val_df, catboost_w, blend_bias), nan=0.0)
        val_stacked = np.nan_to_num(_apply_ensemble(val_df, pred_cols, weights, bias), nan=0.0)
        y_val = val_df[TARGET_COLUMN].to_numpy()
        use_hybrid = mean_absolute_error(y_val, val_hybrid) <= mean_absolute_error(
            y_val, val_stacked
        )
        ensemble_weights[horizon]["_use_hybrid"] = 1.0 if use_hybrid else 0.0

        val_preds = val_hybrid if use_hybrid else val_stacked
        val_preds = val_preds[~np.isnan(val_preds)]
        if len(val_preds) > 0:
            clip_bounds[horizon] = {
                "lower": max(0.0, float(np.percentile(val_preds, 1))),
                "upper": float(np.percentile(val_preds, 99)),
            }
        else:
            clip_bounds[horizon] = {"lower": 0.0, "upper": 5000.0}

        print(f"  CatBoost blend weight: {catboost_w:.2f}")
        print(f"  Use hybrid blend: {use_hybrid}")

    final_predictions = []
    for horizon in range(1, 13):
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        pred_cols = [c for c in ENSEMBLE_SOURCE_COLS if c in horizon_data.columns]

        if ensemble_weights[horizon].get("_use_hybrid", 0) >= 0.5:
            ensemble_pred = _hybrid_prediction(
                horizon_data, blend_weights[horizon], bias_corrections[horizon]
            )
        else:
            ensemble_pred = _apply_ensemble(
                horizon_data, pred_cols, ensemble_weights[horizon], bias_corrections[horizon]
            )

        seasonal = horizon_data["pred_seasonal_blend"].fillna(horizon_data["pred_catboost"])
        catboost = horizon_data["pred_catboost"].fillna(seasonal)
        ensemble_pred = np.where(np.isnan(ensemble_pred), catboost, ensemble_pred)

        ensemble_pred = np.clip(
            ensemble_pred,
            clip_bounds[horizon]["lower"],
            clip_bounds[horizon]["upper"],
        )

        horizon_data["final_predicted_ptf"] = ensemble_pred
        horizon_data["absolute_error"] = np.abs(
            horizon_data[TARGET_COLUMN] - horizon_data["final_predicted_ptf"]
        )
        final_predictions.append(horizon_data)
    
    final_df = pd.concat(final_predictions, ignore_index=True)
    
    # Remove any rows with NaN predictions
    final_df = final_df.dropna(subset=["final_predicted_ptf"])
    
    # Save final predictions
    FINAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_PREDICTIONS_PATH, index=False)
    
    # Calculate metrics
    overall = _metrics(
        final_df[TARGET_COLUMN].to_numpy(),
        final_df["final_predicted_ptf"].to_numpy(),
    )
    horizon_metrics = _horizon_metrics(final_df)
    horizon_metrics.to_csv(FINAL_HORIZON_METRICS_PATH, index=False)
    
    # Create comparison with baseline
    comparison = _create_comparison(final_df, catboost_preds, baseline_metrics)
    comparison.to_csv(FINAL_COMPARISON_PATH, index=False)
    
    # Save metrics
    payload = {
        **overall,
        "baseline_comparison": {
            "mae": baseline_metrics.get("MAE"),
            "rmse": baseline_metrics.get("RMSE"),
            "smape": baseline_metrics.get("SMAPE"),
            "r2": baseline_metrics.get("R2"),
        },
        "ensemble_weights": ensemble_weights,
        "blend_weights": blend_weights,
        "bias_corrections": bias_corrections,
        "clip_bounds": clip_bounds,
    }
    FINAL_METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Calculate improvements
    mae_improvement = (baseline_metrics.get("MAE", 0) - overall["MAE"]) / baseline_metrics.get("MAE", 1) * 100
    rmse_improvement = (baseline_metrics.get("RMSE", 0) - overall["RMSE"]) / baseline_metrics.get("RMSE", 1) * 100
    smape_improvement = (baseline_metrics.get("SMAPE", 0) - overall["SMAPE"]) / baseline_metrics.get("SMAPE", 1) * 100
    r2_improvement = overall["R2"] - baseline_metrics.get("R2", 0)
    
    return FinalEnsembleSummary(
        overall_mae=overall["MAE"],
        overall_rmse=overall["RMSE"],
        overall_smape=overall["SMAPE"],
        overall_r2=overall["R2"],
        baseline_mae=baseline_metrics.get("MAE", 0),
        baseline_rmse=baseline_metrics.get("RMSE", 0),
        baseline_smape=baseline_metrics.get("SMAPE", 0),
        baseline_r2=baseline_metrics.get("R2", 0),
        mae_improvement=mae_improvement,
        rmse_improvement=rmse_improvement,
        smape_improvement=smape_improvement,
        r2_improvement=r2_improvement,
        predictions_path=project_relative_path(FINAL_PREDICTIONS_PATH),
        metrics_path=project_relative_path(FINAL_METRICS_PATH),
        horizon_metrics_path=project_relative_path(FINAL_HORIZON_METRICS_PATH),
        comparison_path=project_relative_path(FINAL_COMPARISON_PATH),
    )


def _load_baseline_metrics() -> dict[str, float]:
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            return json.load(f)
    return {"MAE": 0, "RMSE": 0, "SMAPE": 0, "R2": 0}


def _read_forecast_data() -> pd.DataFrame:
    if not FORECAST_DATA_PATH.exists():
        raise FileNotFoundError(f"Forecast dataset not found: {FORECAST_DATA_PATH}")
    data = pd.read_csv(FORECAST_DATA_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN]).reset_index(drop=True)


def _read_hourly_data() -> pd.DataFrame:
    if not HOURLY_DATA_PATH.exists():
        raise FileNotFoundError(f"Hourly dataset not found: {HOURLY_DATA_PATH}")
    data = pd.read_csv(HOURLY_DATA_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def _read_catboost_predictions() -> pd.DataFrame:
    if not CATBOOST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"CatBoost predictions not found: {CATBOOST_PREDICTIONS_PATH}")
    data = pd.read_csv(CATBOOST_PREDICTIONS_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data[["issue_datetime", "target_datetime", "forecast_horizon", "predicted_ptf"]].rename(
        columns={"predicted_ptf": "pred_catboost"}
    )


def _create_simple_baselines(forecast_data: pd.DataFrame, hourly_data: pd.DataFrame) -> pd.DataFrame:
    """Kesinlesmis PTF mevsimsel baseline'lari (hedef saate gore)."""
    hourly = prepare_kesin_hourly(hourly_data)

    baselines = []
    for _, row in forecast_data.iterrows():
        issue_dt = row["issue_datetime"]
        target_dt = row["target_datetime"]
        seasonal = seasonal_predictions_for_target(hourly, issue_dt, target_dt)

        baselines.append(
            {
                "issue_datetime": issue_dt,
                "target_datetime": target_dt,
                "forecast_horizon": row["forecast_horizon"],
                **seasonal,
                # Geriye uyumluluk (eski kolon adlari)
                "pred_same_hour_yesterday": seasonal["pred_kesin_same_hour_yesterday"],
                "pred_same_hour_last_week": seasonal["pred_kesin_same_hour_last_week"],
                "pred_last_24h_mean": seasonal["pred_kesin_last_24h_mean"],
                "pred_rolling_24h": seasonal["pred_kesin_rolling_24h"],
                "pred_rolling_168h": seasonal["pred_kesin_rolling_168h"],
            }
        )

    return pd.DataFrame(baselines)


def _fit_stacked_ensemble(
    val_df: pd.DataFrame, pred_cols: list[str]
) -> tuple[dict[str, float], float]:
    """Non-negative stacking (Ridge benzeri, pozitif katsayilar)."""
    available_cols = [col for col in pred_cols if val_df[col].notna().any()]
    weights_dict = {col: 0.0 for col in pred_cols}

    if not available_cols:
        return weights_dict, 0.0

    if len(available_cols) == 1:
        col = available_cols[0]
        weights_dict[col] = 1.0
        bias = float((val_df[TARGET_COLUMN] - val_df[col]).mean())
        return weights_dict, bias

    X = val_df[available_cols].fillna(val_df[available_cols].median()).to_numpy()
    y = val_df[TARGET_COLUMN].to_numpy()

    reg = LinearRegression(positive=True, fit_intercept=True)
    reg.fit(X, y)

    coef_sum = float(reg.coef_.sum())
    if coef_sum > 1e-6:
        normalized = reg.coef_ / coef_sum
        for col, weight in zip(available_cols, normalized):
            weights_dict[col] = float(weight)
        bias = float(reg.intercept_)
    else:
        weights_dict[available_cols[0]] = 1.0
        bias = float((y - X[:, 0]).mean())

    return weights_dict, bias


def _apply_ensemble(df: pd.DataFrame, pred_cols: list[str], weights: dict[str, float], bias: float) -> np.ndarray:
    """Stacked ensemble: intercept + agirlikli toplam."""
    X = df[pred_cols].fillna(df[pred_cols].median()).to_numpy()
    weight_array = np.array([weights.get(col, 0.0) for col in pred_cols])
    return np.dot(X, weight_array) + bias


def _fit_catboost_seasonal_blend(val_df: pd.DataFrame) -> tuple[float, float]:
    """CatBoost ile mevsimsel blend arasinda en iyi agirligi sec."""
    y = val_df[TARGET_COLUMN].to_numpy()
    catboost = val_df["pred_catboost"].fillna(val_df["pred_seasonal_blend"]).to_numpy()
    seasonal = val_df["pred_seasonal_blend"].fillna(val_df["pred_catboost"]).to_numpy()

    best_w = 1.0
    best_mae = float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        pred = w * catboost + (1.0 - w) * seasonal
        mae = mean_absolute_error(y, pred)
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)

    blend_pred = best_w * catboost + (1.0 - best_w) * seasonal
    bias = float(np.mean(y - blend_pred))
    return best_w, bias


def _hybrid_prediction(df: pd.DataFrame, catboost_weight: float, bias: float) -> np.ndarray:
    catboost = df["pred_catboost"].fillna(df["pred_seasonal_blend"]).to_numpy()
    seasonal = df["pred_seasonal_blend"].fillna(df["pred_catboost"]).to_numpy()
    return catboost_weight * catboost + (1.0 - catboost_weight) * seasonal + bias


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }


def _horizon_metrics(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for horizon, group in prediction_frame.groupby("forecast_horizon"):
        metrics = _metrics(group[TARGET_COLUMN].to_numpy(), group["final_predicted_ptf"].to_numpy())
        rows.append({"forecast_horizon": int(horizon), **metrics})
    return pd.DataFrame(rows).sort_values("forecast_horizon")


def _create_comparison(final_df: pd.DataFrame, catboost_preds: pd.DataFrame, baseline_metrics: dict) -> pd.DataFrame:
    """Create comparison between final ensemble and CatBoost baseline."""
    comparison = final_df[["issue_datetime", "target_datetime", "forecast_horizon", TARGET_COLUMN, "final_predicted_ptf"]].copy()
    comparison = comparison.merge(
        catboost_preds[["issue_datetime", "target_datetime", "forecast_horizon", "pred_catboost"]],
        on=["issue_datetime", "target_datetime", "forecast_horizon"],
        how="left",
    )
    comparison["catboost_error"] = np.abs(comparison[TARGET_COLUMN] - comparison["pred_catboost"])
    comparison["ensemble_error"] = np.abs(comparison[TARGET_COLUMN] - comparison["final_predicted_ptf"])
    comparison["ensemble_better"] = comparison["ensemble_error"] < comparison["catboost_error"]
    
    return comparison
