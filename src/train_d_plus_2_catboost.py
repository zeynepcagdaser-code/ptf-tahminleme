from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT


BEFORE_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_dataset.csv"
AFTER_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_dataset.csv"

BEFORE_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_predictions.csv"
AFTER_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_predictions.csv"
METRICS_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_metrics_comparison.csv"

BEFORE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "d_plus_2_before_14_catboost_model.pkl"
AFTER_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "d_plus_2_after_14_catboost_model.pkl"

TARGET_COLUMN = "ptf_target"
ID_COLUMNS = {"forecast_issue_date", "target_date", "target_hour", "target_datetime", TARGET_COLUMN}
EPSILON = 1.0


@dataclass(frozen=True)
class DPlus2TrainingSummary:
    before_metrics: dict[str, float | int | str]
    after_metrics: dict[str, float | int | str]
    comparison_path: str
    before_predictions_path: str
    after_predictions_path: str


def train_d_plus_2_catboost_models() -> DPlus2TrainingSummary:
    before_metrics = _train_one_scenario(
        scenario="before_14",
        dataset_path=BEFORE_14_PATH,
        predictions_path=BEFORE_PREDICTIONS_PATH,
        model_path=BEFORE_MODEL_PATH,
    )
    after_metrics = _train_one_scenario(
        scenario="after_14",
        dataset_path=AFTER_14_PATH,
        predictions_path=AFTER_PREDICTIONS_PATH,
        model_path=AFTER_MODEL_PATH,
    )

    comparison = pd.DataFrame([before_metrics, after_metrics])
    METRICS_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_COMPARISON_PATH, index=False)

    return DPlus2TrainingSummary(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        comparison_path=str(METRICS_COMPARISON_PATH),
        before_predictions_path=str(BEFORE_PREDICTIONS_PATH),
        after_predictions_path=str(AFTER_PREDICTIONS_PATH),
    )


def _train_one_scenario(
    scenario: str,
    dataset_path,
    predictions_path,
    model_path,
) -> dict[str, float | int | str]:
    data = _read_dataset(dataset_path)
    feature_columns = _feature_columns(data)
    train_df, test_df = _chronological_split(data)

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    model.fit(
        train_df[feature_columns],
        train_df[TARGET_COLUMN],
        eval_set=(test_df[feature_columns], test_df[TARGET_COLUMN]),
        use_best_model=True,
    )

    predictions = model.predict(test_df[feature_columns])
    metrics = _metrics(test_df[TARGET_COLUMN].to_numpy(), predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump(model, file)

    prediction_frame = pd.DataFrame(
        {
            "scenario": scenario,
            "forecast_issue_date": test_df["forecast_issue_date"].to_numpy(),
            "target_date": test_df["target_date"].to_numpy(),
            "target_hour": test_df["target_hour"].to_numpy(),
            "target_datetime": test_df["target_datetime"].to_numpy(),
            "actual_ptf": test_df[TARGET_COLUMN].to_numpy(),
            "predicted_ptf": predictions,
            "error": test_df[TARGET_COLUMN].to_numpy() - predictions,
            "absolute_error": np.abs(test_df[TARGET_COLUMN].to_numpy() - predictions),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_path, index=False)

    return {
        "scenario": scenario,
        "rows": int(len(data)),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "SMAPE": metrics["SMAPE"],
        "R2": metrics["R2"],
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }


def _read_dataset(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"D+2 dataset bulunamadi: {path}")
    data = pd.read_csv(path)
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    data = data.dropna(subset=["target_datetime", TARGET_COLUMN]).sort_values(
        ["forecast_issue_date", "target_hour"]
    )
    return data.reset_index(drop=True)


def _feature_columns(data: pd.DataFrame) -> list[str]:
    columns = [column for column in data.columns if column not in ID_COLUMNS]
    forbidden_exact = {
        "ptf",
        "smf",
        "real_time_consumption",
        "wind_generation",
        "solar_generation",
        "hydro_dam_generation",
        "unlicensed_generation_total",
    }
    leakage = sorted(forbidden_exact.intersection(columns))
    if leakage:
        raise ValueError(f"Leakage kolonlari feature listesine girdi: {leakage}")
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Sayisal olmayan feature kolonlari var: {non_numeric}")
    return columns


def _chronological_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(data) * 0.8)
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("Kronolojik split icin yeterli veri yok.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }
