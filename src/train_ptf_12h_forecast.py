from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.build_12h_forecast_dataset import SEASONAL_ANCHOR_COLUMN
from src.catboost_tuning import (
    fit_tuned_catboost_regressor,
    predict_with_optional_log_transform,
)
from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path
from src.ptf_feature_enrichment import enrich_forecast_features, recency_sample_weights


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_12h_dataset.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_metrics.json"
HORIZON_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_horizon_metrics.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "data" / "models" / "ptf_12h_horizon_models.pkl"
LIVE_FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_live_forecast.csv"

TARGET_COLUMN = "ptf_target"
RESIDUAL_TARGET_COLUMN = "ptf_target_residual"
DELTA_TARGET_COLUMN = "ptf_target_delta"
INTERIM_ANCHOR_COLUMN = "interim_anchor"
ID_COLUMNS = [
    "issue_datetime",
    "target_datetime",
    "forecast_horizon",
    TARGET_COLUMN,
    RESIDUAL_TARGET_COLUMN,
    DELTA_TARGET_COLUMN,
    SEASONAL_ANCHOR_COLUMN,
    INTERIM_ANCHOR_COLUMN,
]
USE_RESIDUAL_TARGET = True
USE_LOG_TARGET = False
EPSILON = 1.0

TUNING_CANDIDATES = (
    {"depth": 7, "learning_rate": 0.03, "l2_leaf_reg": 6, "min_data_in_leaf": 20},
    {"depth": 8, "learning_rate": 0.022, "l2_leaf_reg": 9, "min_data_in_leaf": 18},
    {"depth": 8, "learning_rate": 0.018, "l2_leaf_reg": 11, "min_data_in_leaf": 22},
)


@dataclass(frozen=True)
class Ptf12hTrainingSummary:
    overall_mae: float
    overall_rmse: float
    overall_smape: float
    overall_r2: float
    predictions_path: str
    metrics_path: str
    horizon_metrics_path: str
    model_bundle_path: str
    live_forecast_path: str


def train_ptf_12h_forecast() -> Ptf12hTrainingSummary:
    data = enrich_forecast_features(_read_dataset())
    feature_columns = _feature_columns(data)

    models: dict[int, object] = {}
    tuning_reports: dict[int, dict] = {}
    test_predictions: list[pd.DataFrame] = []

    for horizon in range(1, 13):
        horizon_data = data[data["forecast_horizon"] == horizon].copy()
        horizon_data = horizon_data.sort_values("issue_datetime").reset_index(drop=True)
        train_df, val_df, test_df = chronological_train_val_test_split(horizon_data)

        train_target_col = _training_target_column(train_df)
        y_train = _transform_target(train_df[train_target_col])
        y_val = _transform_target(val_df[train_target_col])
        train_weights = recency_sample_weights(train_df["issue_datetime"])

        model, tuning_info = fit_tuned_catboost_regressor(
            train_df[feature_columns],
            y_train,
            val_df[feature_columns],
            y_val,
            verbose=0,
            param_candidates=TUNING_CANDIDATES,
            sample_weight=train_weights,
        )
        models[horizon] = model
        tuning_reports[horizon] = tuning_info

        residual_preds = predict_with_optional_log_transform(
            model, test_df[feature_columns], use_log_target=USE_LOG_TARGET
        )
        preds = _to_kesinlesmis_prediction(test_df, residual_preds)
        frame = test_df[[c for c in ID_COLUMNS if c in test_df.columns]].copy()
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
        "use_residual_target": USE_RESIDUAL_TARGET,
        "tuning_by_horizon": tuning_reports,
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_BUNDLE_PATH.open("wb") as file:
        pickle.dump(
            {
                "models": models,
                "feature_columns": feature_columns,
                "use_log_target": USE_LOG_TARGET,
                "use_residual_target": USE_RESIDUAL_TARGET,
            },
            file,
        )

    live_forecast = predict_live_next_12h(models, feature_columns)
    live_forecast.to_csv(LIVE_FORECAST_PATH, index=False)

    return Ptf12hTrainingSummary(
        overall_mae=overall["MAE"],
        overall_rmse=overall["RMSE"],
        overall_smape=overall["SMAPE"],
        overall_r2=overall["R2"],
        predictions_path=project_relative_path(PREDICTIONS_PATH),
        metrics_path=project_relative_path(METRICS_PATH),
        horizon_metrics_path=project_relative_path(HORIZON_METRICS_PATH),
        model_bundle_path=project_relative_path(MODEL_BUNDLE_PATH),
        live_forecast_path=project_relative_path(LIVE_FORECAST_PATH),
    )


def predict_live_next_12h(
    models: dict[int, object] | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    from src.build_12h_forecast_dataset import (
        FORECAST_HORIZON_HOURS,
        _features_at_cutoff,
        _load_forecast_at,
        _read_hourly_dataset,
        _target_calendar_features,
        _value_at,
    )

    if models is None or feature_columns is None:
        with MODEL_BUNDLE_PATH.open("rb") as file:
            bundle = pickle.load(file)
        models = bundle["models"]
        feature_columns = bundle["feature_columns"]
        use_log = bundle.get("use_log_target", USE_LOG_TARGET)
    else:
        use_log = USE_LOG_TARGET

    hourly = _read_hourly_dataset()
    by_datetime = hourly.set_index("datetime").sort_index()
    ptf_known = by_datetime["ptf"].dropna()
    if ptf_known.empty:
        raise ValueError("Kesinlesmemis PTF (I-MCP) saati bulunamadi.")

    cutoff = ptf_known.index.max()
    base_features = _features_at_cutoff(by_datetime, cutoff)
    if base_features is None:
        raise ValueError("Canli tahmin icin ozellikler olusturulamadi.")

    rows: list[dict] = []
    for horizon in range(1, FORECAST_HORIZON_HOURS + 1):
        target_datetime = cutoff + pd.Timedelta(hours=horizon)
        kesin_seasonal_anchor = _value_at(
            by_datetime,
            target_datetime - pd.Timedelta(hours=24),
            "ptf_kesinlesmis",
        )
        kesin_at_target_week = _value_at(
            by_datetime,
            target_datetime - pd.Timedelta(hours=168),
            "ptf_kesinlesmis",
        )

        row = {
            "issue_datetime": cutoff,
            "target_datetime": target_datetime,
            "forecast_horizon": horizon,
            **base_features,
            **_target_calendar_features(target_datetime),
            "load_forecast_plan_target_hour": _load_forecast_at(
                by_datetime, target_datetime, cutoff
            ),
            SEASONAL_ANCHOR_COLUMN: kesin_seasonal_anchor,
            "kesin_at_target_week_ago": kesin_at_target_week,
            INTERIM_ANCHOR_COLUMN: float(base_features.get("ptf_lag_1", np.nan)),
        }
        feature_row = enrich_forecast_features(pd.DataFrame([row]))
        for column in feature_columns:
            if column not in feature_row.columns:
                feature_row[column] = np.nan

        model = models[horizon]
        residual_pred = predict_with_optional_log_transform(
            model, feature_row[feature_columns], use_log_target=use_log
        )[0]
        kesin_pred = max(
            0.0,
            float(_to_kesinlesmis_prediction(feature_row, np.array([residual_pred]))[0]),
        )
        rows.append(
            {
                "issue_datetime": cutoff,
                "target_datetime": target_datetime,
                "forecast_horizon": horizon,
                "predicted_ptf": kesin_pred,
                SEASONAL_ANCHOR_COLUMN: kesin_seasonal_anchor,
                INTERIM_ANCHOR_COLUMN: float(base_features.get("ptf_lag_1", np.nan)),
            }
        )

    return pd.DataFrame(rows)


def _read_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"12 saatlik veri seti bulunamadi: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN]).reset_index(drop=True)


def _training_target_column(data: pd.DataFrame) -> str:
    if USE_RESIDUAL_TARGET and RESIDUAL_TARGET_COLUMN in data.columns:
        return RESIDUAL_TARGET_COLUMN
    if DELTA_TARGET_COLUMN in data.columns:
        return DELTA_TARGET_COLUMN
    return TARGET_COLUMN


def _to_kesinlesmis_prediction(frame: pd.DataFrame, residual_preds: np.ndarray) -> np.ndarray:
    if USE_RESIDUAL_TARGET and SEASONAL_ANCHOR_COLUMN in frame.columns:
        anchor = frame[SEASONAL_ANCHOR_COLUMN].to_numpy(dtype=float)
        return np.maximum(0.0, anchor + np.asarray(residual_preds, dtype=float))
    if INTERIM_ANCHOR_COLUMN in frame.columns:
        anchor = frame[INTERIM_ANCHOR_COLUMN].to_numpy(dtype=float)
        return np.maximum(0.0, anchor + np.asarray(residual_preds, dtype=float))
    return np.maximum(0.0, np.asarray(residual_preds, dtype=float))


def _feature_columns(data: pd.DataFrame) -> list[str]:
    id_set = set(ID_COLUMNS)
    columns = [column for column in data.columns if column not in id_set]
    if "ptf" in columns:
        raise ValueError("Ham ptf kolonu feature listesine girmemeli.")
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Sayisal olmayan kolonlar: {non_numeric}")
    return columns


def _transform_target(target: pd.Series) -> pd.Series:
    values = target.astype(float)
    if USE_LOG_TARGET:
        return np.log1p(values)
    return values


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
