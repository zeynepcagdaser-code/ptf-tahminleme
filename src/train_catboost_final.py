from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_feature_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "catboost_final_model.pkl"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_final_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_final_metrics.json"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_final_feature_importance.csv"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"

TARGET_COLUMN = "ptf"
EXCLUDED_FEATURE_COLUMNS = {"datetime", "date", TARGET_COLUMN}
CURRENT_HOUR_MEASURED_COLUMNS = {
    "gop_fiyattan_bagimsiz_alis",
    "gop_fiyattan_bagimsiz_satis",
    "price_independent_buy_sell_ratio",
    "load_forecast_plan",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "real_time_consumption",
    "grf_tl",
    "unlicensed_generation_total",
    "smf",
    "usd_try",
}
EPSILON = 1.0


@dataclass(frozen=True)
class CatBoostFinalResult:
    total_rows: int
    feature_count: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    top_features: list[dict[str, float | str]]
    model_path: str
    predictions_path: str
    metrics_path: str
    feature_importance_path: str


def run_catboost_final_training() -> CatBoostFinalResult:
    dataset = _read_dataset()
    feature_columns = _get_feature_columns(dataset)
    train_df, val_df, test_df = chronological_train_val_test_split(dataset)

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df[feature_columns]
    y_val = val_df[TARGET_COLUMN]
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN]

    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    model.fit(x_train, y_train, eval_set=(x_val, y_val), use_best_model=True)

    predictions = model.predict(x_test)
    metrics = _calculate_metrics(y_test.to_numpy(), predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    prediction_frame = pd.DataFrame(
        {
            "datetime": test_df["datetime"].to_numpy(),
            "date": test_df["date"].to_numpy(),
            "actual_ptf": y_test.to_numpy(),
            "predicted_ptf": predictions,
            "error": y_test.to_numpy() - predictions,
            "absolute_error": np.abs(y_test.to_numpy() - predictions),
        }
    )
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    importance_frame.to_csv(IMPORTANCE_PATH, index=False)

    top_features = importance_frame.head(20).to_dict(orient="records")
    metrics_payload = {
        **metrics,
        "total_rows": len(dataset),
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "top_20_features": top_features,
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_actual_vs_predicted(prediction_frame)
    _plot_error_distribution(prediction_frame)
    _plot_feature_importance(importance_frame)

    return CatBoostFinalResult(
        total_rows=len(dataset),
        feature_count=len(feature_columns),
        train_rows=len(train_df),
        test_rows=len(test_df),
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        mape=metrics["MAPE"],
        smape=metrics["SMAPE"],
        r2=metrics["R2"],
        top_features=top_features,
        model_path=project_relative_path(MODEL_PATH),
        predictions_path=project_relative_path(PREDICTIONS_PATH),
        metrics_path=project_relative_path(METRICS_PATH),
        feature_importance_path=project_relative_path(IMPORTANCE_PATH),
    )


def _read_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Feature dataset bulunamadi: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Target kolon bulunamadi: {TARGET_COLUMN}")

    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
    dataset = dataset.dropna(subset=["datetime", TARGET_COLUMN])
    dataset = dataset.sort_values("datetime").reset_index(drop=True)
    return dataset


def _get_feature_columns(dataset: pd.DataFrame) -> list[str]:
    excluded_columns = EXCLUDED_FEATURE_COLUMNS | CURRENT_HOUR_MEASURED_COLUMNS
    feature_columns = [column for column in dataset.columns if column not in excluded_columns]
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise ValueError(f"Sayisal olmayan feature kolonlari var: {non_numeric[:20]}")
    if TARGET_COLUMN in feature_columns:
        raise ValueError("Target ptf feature listesine girdi; bu veri sızıntısı olur.")
    return feature_columns


def _calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    absolute_percentage_error = np.abs(actual - predicted) / np.maximum(np.abs(actual), EPSILON)
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, EPSILON)

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE": float(np.mean(absolute_percentage_error) * 100),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }


def _plot_actual_vs_predicted(prediction_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(prediction_frame["datetime"], prediction_frame["actual_ptf"], label="Actual PTF", linewidth=1.2)
    plt.plot(prediction_frame["datetime"], prediction_frame["predicted_ptf"], label="Predicted PTF", linewidth=1.2)
    plt.title("CatBoost Final - Actual vs Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("PTF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_final_actual_vs_predicted.png", dpi=150)
    plt.close()


def _plot_error_distribution(prediction_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_frame["error"], bins=60, edgecolor="black", alpha=0.75)
    plt.title("CatBoost Final - Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_final_error_distribution.png", dpi=150)
    plt.close()


def _plot_feature_importance(importance_frame: pd.DataFrame) -> None:
    top_features = importance_frame.head(30).sort_values("importance", ascending=True)
    plt.figure(figsize=(11, 9))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("CatBoost Final - Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_final_feature_importance.png", dpi=150)
    plt.close()


def result_to_dict(result: CatBoostFinalResult) -> dict:
    return asdict(result)
