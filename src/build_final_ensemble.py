from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    all_preds = all_preds.merge(catboost_preds, on=ID_COLUMNS[:3], how="left")
    all_preds = all_preds.merge(simple_preds, on=ID_COLUMNS[:3], how="left")
    
    # Optimize ensemble weights per horizon
    ensemble_weights = {}
    bias_corrections = {}
    clip_bounds = {}
    
    for horizon in range(1, 13):
        print(f"\n=== Optimizing ensemble for horizon {horizon}h ===")
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        horizon_data = horizon_data.sort_values("issue_datetime").reset_index(drop=True)
        
        # Split into train/val/test
        train_df, val_df, test_df = chronological_train_val_test_split(horizon_data)
        
        # Get prediction columns
        pred_cols = [col for col in all_preds.columns if col.startswith("pred_")]
        
        # Optimize weights on validation set
        weights, bias = _optimize_ensemble_weights(val_df, pred_cols)
        ensemble_weights[horizon] = weights
        bias_corrections[horizon] = bias
        
        # Calculate clip bounds from validation predictions
        val_preds = _apply_ensemble(val_df, pred_cols, weights, bias)
        val_preds = val_preds[~np.isnan(val_preds)]  # Remove NaN for percentile calculation
        if len(val_preds) > 0:
            clip_bounds[horizon] = {
                "lower": max(0, np.percentile(val_preds, 1)),  # Clip to at least 0
                "upper": np.percentile(val_preds, 99),  # Clip extreme high values
            }
        else:
            # Fallback bounds if no valid predictions
            clip_bounds[horizon] = {"lower": 0, "upper": 5000}
        
        print(f"  Weights: {weights}")
        print(f"  Bias correction: {bias:.2f}")
        print(f"  Clip bounds: [{clip_bounds[horizon]['lower']:.2f}, {clip_bounds[horizon]['upper']:.2f}]")
    
    # Apply ensemble to all data
    final_predictions = []
    for horizon in range(1, 13):
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        pred_cols = [col for col in all_preds.columns if col.startswith("pred_")]
        
        # Apply ensemble
        ensemble_pred = _apply_ensemble(horizon_data, pred_cols, ensemble_weights[horizon], bias_corrections[horizon])
        
        # Replace NaN with CatBoost prediction as fallback
        catboost_fallback = horizon_data["pred_catboost"].fillna(horizon_data[TARGET_COLUMN].mean())
        ensemble_pred = np.where(np.isnan(ensemble_pred), catboost_fallback, ensemble_pred)
        
        # Apply clipping
        ensemble_pred = np.clip(
            ensemble_pred,
            clip_bounds[horizon]["lower"],
            clip_bounds[horizon]["upper"],
        )
        
        horizon_data["final_predicted_ptf"] = ensemble_pred
        horizon_data["absolute_error"] = np.abs(horizon_data[TARGET_COLUMN] - horizon_data["final_predicted_ptf"])
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
    """Create simple baseline predictions using hourly data."""
    hourly = hourly_data.set_index("datetime").sort_index()
    
    # Calculate rolling statistics
    hourly["rolling_24h_mean"] = hourly["ptf"].rolling(window=24, min_periods=1).mean()
    hourly["rolling_168h_mean"] = hourly["ptf"].rolling(window=168, min_periods=1).mean()
    
    # Create baseline predictions for each forecast row
    baselines = []
    for _, row in forecast_data.iterrows():
        issue_dt = row["issue_datetime"]
        
        # Same hour yesterday
        same_hour_yesterday_dt = issue_dt - pd.Timedelta(hours=24)
        pred_same_hour_yesterday = _get_value_at(hourly, same_hour_yesterday_dt, "ptf")
        
        # Same hour last week
        same_hour_last_week_dt = issue_dt - pd.Timedelta(hours=168)
        pred_same_hour_last_week = _get_value_at(hourly, same_hour_last_week_dt, "ptf")
        
        # Last 24h mean
        last_24h_start = issue_dt - pd.Timedelta(hours=24)
        last_24h_end = issue_dt
        last_24h_data = hourly.loc[(hourly.index >= last_24h_start) & (hourly.index < last_24h_end), "ptf"]
        pred_last_24h_mean = last_24h_data.mean() if len(last_24h_data) > 0 else np.nan
        
        # Rolling 24h mean
        pred_rolling_24h = _get_value_at(hourly, issue_dt, "rolling_24h_mean")
        
        # Rolling 168h mean
        pred_rolling_168h = _get_value_at(hourly, issue_dt, "rolling_168h_mean")
        
        baselines.append({
            "issue_datetime": issue_dt,
            "target_datetime": row["target_datetime"],
            "forecast_horizon": row["forecast_horizon"],
            "pred_same_hour_yesterday": pred_same_hour_yesterday,
            "pred_same_hour_last_week": pred_same_hour_last_week,
            "pred_last_24h_mean": pred_last_24h_mean,
            "pred_rolling_24h": pred_rolling_24h,
            "pred_rolling_168h": pred_rolling_168h,
        })
    
    return pd.DataFrame(baselines)


def _get_value_at(df: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    """Get value at specific timestamp, return NaN if not found."""
    try:
        if timestamp in df.index:
            value = df.loc[timestamp, column]
            return float(value) if not pd.isna(value) else np.nan
    except (KeyError, TypeError):
        pass
    return np.nan


def _optimize_ensemble_weights(val_df: pd.DataFrame, pred_cols: list[str]) -> tuple[dict[str, float], float]:
    """Optimize ensemble weights to minimize MAE on validation set."""
    
    # Get available prediction columns (some might be NaN)
    available_cols = [col for col in pred_cols if val_df[col].notna().any()]
    
    if len(available_cols) == 0:
        # Fallback to equal weights if no predictions available
        return {col: 1.0 / len(pred_cols) for col in pred_cols}, 0.0
    
    if len(available_cols) == 1:
        # If only one prediction source, use it with weight 1
        weights = {col: 0.0 for col in pred_cols}
        weights[available_cols[0]] = 1.0
        bias = (val_df[TARGET_COLUMN] - val_df[available_cols[0]]).mean()
        return weights, bias
    
    # Prepare data for optimization
    X = val_df[available_cols].fillna(val_df[available_cols].mean()).to_numpy()
    y = val_df[TARGET_COLUMN].to_numpy()
    
    # Objective function: minimize MAE
    def objective(weights):
        weights = np.array(weights)
        ensemble_pred = np.dot(X, weights)
        mae = mean_absolute_error(y, ensemble_pred)
        return mae
    
    # Constraints: weights sum to 1, non-negative
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(available_cols))]
    
    # Initial guess: equal weights
    initial_weights = np.ones(len(available_cols)) / len(available_cols)
    
    # Optimize
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints, method="SLSQP")
    
    optimal_weights = result.x
    bias = (y - np.dot(X, optimal_weights)).mean()
    
    # Map back to all prediction columns
    weights_dict = {col: 0.0 for col in pred_cols}
    for i, col in enumerate(available_cols):
        weights_dict[col] = optimal_weights[i]
    
    return weights_dict, bias


def _apply_ensemble(df: pd.DataFrame, pred_cols: list[str], weights: dict[str, float], bias: float) -> np.ndarray:
    """Apply ensemble weights to predictions."""
    X = df[pred_cols].fillna(df[pred_cols].mean()).to_numpy()
    weight_array = np.array([weights[col] for col in pred_cols])
    ensemble_pred = np.dot(X, weight_array) - bias
    return ensemble_pred


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
