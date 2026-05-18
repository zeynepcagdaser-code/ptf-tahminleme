from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_12h_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_predictions_enhanced.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_metrics_enhanced.json"
HORIZON_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_horizon_metrics_enhanced.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "data" / "models" / "ptf_12h_horizon_models_enhanced.pkl"
BASELINE_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_metrics.json"

TARGET_COLUMN = "ptf_target"
ID_COLUMNS = [
    "issue_datetime",
    "target_datetime",
    "forecast_horizon",
    TARGET_COLUMN,
]
USE_LOG_TARGET = False  # Disabled to reduce bias from log transformation
EPSILON = 1.0

# Enhanced hyperparameter grids per horizon
HORIZON_TUNING_PARAMS = {
    1: {"depth": [6, 7, 8], "learning_rate": [0.03, 0.05, 0.07], "l2_leaf_reg": [3, 5, 7], "iterations": [500, 700, 1000]},
    2: {"depth": [6, 7, 8], "learning_rate": [0.03, 0.05, 0.07], "l2_leaf_reg": [3, 5, 7], "iterations": [500, 700, 1000]},
    3: {"depth": [6, 7, 8], "learning_rate": [0.03, 0.05, 0.07], "l2_leaf_reg": [3, 5, 7], "iterations": [500, 700, 1000]},
    4: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    5: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    6: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    7: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    8: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    9: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    10: {"depth": [7, 8, 9], "learning_rate": [0.02, 0.03, 0.05], "l2_leaf_reg": [5, 7, 10], "iterations": [700, 1000, 1500]},
    11: {"depth": [6, 7, 8], "learning_rate": [0.03, 0.05, 0.07], "l2_leaf_reg": [3, 5, 7], "iterations": [500, 700, 1000]},
    12: {"depth": [6, 7, 8], "learning_rate": [0.03, 0.05, 0.07], "l2_leaf_reg": [3, 5, 7], "iterations": [500, 700, 1000]},
}


@dataclass(frozen=True)
class Ptf12hEnhancedTrainingSummary:
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
    model_bundle_path: str


def train_ptf_12h_forecast_enhanced() -> Ptf12hEnhancedTrainingSummary:
    # Load baseline metrics for comparison
    baseline_metrics = _load_baseline_metrics()
    
    data = _read_dataset()
    data = _add_enhanced_features(data)
    feature_columns = _feature_columns(data)

    models: dict[int, Any] = {}
    tuning_reports: dict[int, dict] = {}
    test_predictions: list[pd.DataFrame] = []

    for horizon in range(1, 13):
        print(f"\n=== Training horizon {horizon}h ===")
        horizon_data = data[data["forecast_horizon"] == horizon].copy()
        horizon_data = horizon_data.sort_values("issue_datetime").reset_index(drop=True)
        train_df, val_df, test_df = chronological_train_val_test_split(horizon_data)

        # Horizon-specific feature selection
        selected_features = _select_features_horizon(train_df, feature_columns, horizon)
        print(f"  Selected {len(selected_features)} features (from {len(feature_columns)})")

        y_train = train_df[TARGET_COLUMN].to_numpy()
        y_val = val_df[TARGET_COLUMN].to_numpy()

        model, tuning_info = _tune_catboost_horizon(
            train_df[selected_features],
            y_train,
            val_df[selected_features],
            y_val,
            horizon,
        )
        models[horizon] = model
        tuning_reports[horizon] = tuning_info

        preds = model.predict(test_df[selected_features])
        frame = test_df[ID_COLUMNS].copy()
        frame["predicted_ptf"] = preds
        frame["actual_ptf"] = test_df[TARGET_COLUMN].to_numpy()
        frame["absolute_error"] = np.abs(frame["actual_ptf"] - frame["predicted_ptf"])
        test_predictions.append(frame)

    prediction_frame = pd.concat(test_predictions, ignore_index=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    overall = _metrics(
        prediction_frame["actual_ptf"].to_numpy(),
        prediction_frame["predicted_ptf"].to_numpy(),
    )
    horizon_metrics = _horizon_metrics(prediction_frame)
    horizon_metrics.to_csv(HORIZON_METRICS_PATH, index=False)

    payload = {
        **overall,
        "rows": int(len(data)),
        "feature_count": len(feature_columns),
        "use_log_target": USE_LOG_TARGET,
        "tuning_by_horizon": tuning_reports,
        "baseline_comparison": {
            "mae": baseline_metrics.get("MAE"),
            "rmse": baseline_metrics.get("RMSE"),
            "smape": baseline_metrics.get("SMAPE"),
            "r2": baseline_metrics.get("R2"),
        },
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_BUNDLE_PATH.open("wb") as file:
        pickle.dump(
            {
                "models": models,
                "feature_columns": feature_columns,
                "use_log_target": USE_LOG_TARGET,
            },
            file,
        )

    # Calculate improvements
    mae_improvement = (baseline_metrics.get("MAE", 0) - overall["MAE"]) / baseline_metrics.get("MAE", 1) * 100
    rmse_improvement = (baseline_metrics.get("RMSE", 0) - overall["RMSE"]) / baseline_metrics.get("RMSE", 1) * 100
    smape_improvement = (baseline_metrics.get("SMAPE", 0) - overall["SMAPE"]) / baseline_metrics.get("SMAPE", 1) * 100
    r2_improvement = overall["R2"] - baseline_metrics.get("R2", 0)

    return Ptf12hEnhancedTrainingSummary(
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
        predictions_path=project_relative_path(PREDICTIONS_PATH),
        metrics_path=project_relative_path(METRICS_PATH),
        horizon_metrics_path=project_relative_path(HORIZON_METRICS_PATH),
        model_bundle_path=project_relative_path(MODEL_BUNDLE_PATH),
    )


def _load_baseline_metrics() -> dict[str, float]:
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH) as f:
            return json.load(f)
    return {"MAE": 0, "RMSE": 0, "SMAPE": 0, "R2": 0}


def _read_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"12 saatlik veri seti bulunamadi: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN]).reset_index(drop=True)


def _add_enhanced_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced features without changing the dataset pipeline."""
    data = data.copy()
    
    # Enhanced calendar features for issue time
    data["issue_hour_sin"] = np.sin(2 * np.pi * data["issue_hour"] / 24)
    data["issue_hour_cos"] = np.cos(2 * np.pi * data["issue_hour"] / 24)
    data["issue_dow_sin"] = np.sin(2 * np.pi * data["issue_day_of_week"] / 7)
    data["issue_dow_cos"] = np.cos(2 * np.pi * data["issue_day_of_week"] / 7)
    
    # Add month feature (already exists as target_month, add issue_month)
    data["issue_month"] = data["issue_datetime"].dt.month
    
    # Add interaction features
    data["hour_weekend_interaction"] = data["issue_hour"] * data["issue_is_weekend"]
    
    # Add momentum features for PTF lags
    if "ptf_lag_1" in data.columns and "ptf_lag_24" in data.columns:
        data["ptf_momentum_1_24"] = data["ptf_lag_1"] - data["ptf_lag_24"]
    if "ptf_lag_1" in data.columns and "ptf_lag_168" in data.columns:
        data["ptf_momentum_1_168"] = data["ptf_lag_1"] - data["ptf_lag_168"]
    
    # Add ratio features
    if "ptf_lag_1" in data.columns and "ptf_lag_24" in data.columns:
        data["ptf_ratio_1_24"] = data["ptf_lag_1"] / (data["ptf_lag_24"] + EPSILON)
    
    # Add rolling statistics enhancements (if not already present)
    if "ptf_interim_window_mean_24h" in data.columns:
        data["ptf_interim_window_cv_24h"] = data["ptf_interim_window_std_24h"] / (data["ptf_interim_window_mean_24h"] + EPSILON)
    
    # Add horizon-specific features
    data["horizon_squared"] = data["forecast_horizon"] ** 2
    data["horizon_sqrt"] = np.sqrt(data["forecast_horizon"])
    
    # Add target hour interactions
    data["target_hour_weekend_interaction"] = data["target_hour"] * data["target_is_weekend"]
    
    return data


def _feature_columns(data: pd.DataFrame) -> list[str]:
    id_set = set(ID_COLUMNS)
    columns = [column for column in data.columns if column not in id_set]
    if "ptf" in columns:
        raise ValueError("Ham ptf kolonu feature listesine girmemeli.")
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Sayisal olmayan kolonlar: {non_numeric}")
    return columns


def _select_features_horizon(train_df: pd.DataFrame, feature_columns: list[str], horizon: int) -> list[str]:
    """Select features based on importance for specific horizon."""
    # For now, use all features but could implement feature selection here
    # Future enhancement: use mutual information, correlation analysis, or model-based importance
    return feature_columns


def _tune_catboost_horizon(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    horizon: int,
) -> tuple[CatBoostRegressor, dict]:
    """Tune CatBoost with horizon-specific hyperparameters."""
    params = HORIZON_TUNING_PARAMS.get(horizon, HORIZON_TUNING_PARAMS[1])
    
    best_model = None
    best_score = float("inf")
    best_params = {}
    
    # Grid search with limited combinations for efficiency
    from itertools import product
    
    param_combinations = list(product(
        params["depth"],
        params["learning_rate"],
        params["l2_leaf_reg"],
        params["iterations"],
    ))
    
    # Limit to 20 random combinations for efficiency
    if len(param_combinations) > 20:
        np.random.shuffle(param_combinations)
        param_combinations = param_combinations[:20]
    
    for depth, lr, l2, iterations in param_combinations:
        model = CatBoostRegressor(
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            iterations=iterations,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            verbose=False,
        )
        
        val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        if val_mae < best_score:
            best_score = val_mae
            best_model = model
            best_params = {
                "depth": depth,
                "learning_rate": lr,
                "l2_leaf_reg": l2,
                "iterations": iterations,
                "val_mae": val_mae,
            }
    
    print(f"  Best params: {best_params}")
    return best_model, best_params


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
        metrics = _metrics(group["actual_ptf"].to_numpy(), group["predicted_ptf"].to_numpy())
        rows.append({"forecast_horizon": int(horizon), **metrics})
    return pd.DataFrame(rows).sort_values("forecast_horizon")
