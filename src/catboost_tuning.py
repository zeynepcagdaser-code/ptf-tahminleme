from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error


DEFAULT_PARAM_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 5, "min_data_in_leaf": 40},
    {"depth": 7, "learning_rate": 0.025, "l2_leaf_reg": 6, "min_data_in_leaf": 35},
    {"depth": 8, "learning_rate": 0.02, "l2_leaf_reg": 8, "min_data_in_leaf": 30},
    {"depth": 6, "learning_rate": 0.02, "l2_leaf_reg": 10, "min_data_in_leaf": 50},
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 4, "min_data_in_leaf": 25},
    {"depth": 8, "learning_rate": 0.015, "l2_leaf_reg": 12, "min_data_in_leaf": 45},
)


def fit_tuned_catboost_regressor(
    train_features: pd.DataFrame,
    train_target: pd.Series | np.ndarray,
    val_features: pd.DataFrame,
    val_target: pd.Series | np.ndarray,
    *,
    loss_function: str = "RMSE",
    random_seed: int = 42,
    verbose: int = 0,
    param_candidates: tuple[dict[str, Any], ...] | None = None,
) -> tuple[CatBoostRegressor, dict[str, Any]]:
    candidates = param_candidates or DEFAULT_PARAM_CANDIDATES
    best_model: CatBoostRegressor | None = None
    best_params: dict[str, Any] = {}
    best_rmse = float("inf")

    for params in candidates:
        model = CatBoostRegressor(
            iterations=2500,
            loss_function=loss_function,
            random_seed=random_seed,
            subsample=0.85,
            rsm=0.85,
            bootstrap_type="Bernoulli",
            od_type="Iter",
            od_wait=80,
            verbose=verbose,
            **params,
        )
        model.fit(
            train_features,
            train_target,
            eval_set=(val_features, val_target),
            use_best_model=True,
            early_stopping_rounds=100,
        )
        predictions = model.predict(val_features)
        rmse = float(np.sqrt(mean_squared_error(val_target, predictions)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("Hiperparametre ayari basarisiz oldu.")

    return best_model, {"best_validation_rmse": best_rmse, **best_params}


def predict_with_optional_log_transform(
    model: CatBoostRegressor,
    features: pd.DataFrame,
    *,
    use_log_target: bool,
) -> np.ndarray:
    predictions = model.predict(features)
    if use_log_target:
        predictions = np.expm1(predictions)
    return np.clip(predictions, 0, None)
