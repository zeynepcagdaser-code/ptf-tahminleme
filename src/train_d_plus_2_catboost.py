from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.catboost_tuning import fit_tuned_catboost_regressor, predict_with_optional_log_transform
from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path


BEFORE_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_dataset.csv"
AFTER_14_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_dataset.csv"

BEFORE_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_before_14_predictions.csv"
AFTER_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_after_14_predictions.csv"
METRICS_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_metrics_comparison.csv"
TUNING_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_tuning_report.json"

BEFORE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "d_plus_2_before_14_catboost_model.pkl"
AFTER_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "d_plus_2_after_14_catboost_model.pkl"

TARGET_COLUMN = "ptf_target"
ID_COLUMNS = {"forecast_issue_date", "target_date", "target_hour", "target_datetime", TARGET_COLUMN}
EPSILON = 1.0
USE_LOG_TARGET = True


@dataclass(frozen=True)
class DPlus2TrainingSummary:
    before_metrics: dict[str, float | int | str]
    after_metrics: dict[str, float | int | str]
    comparison_path: str
    before_predictions_path: str
    after_predictions_path: str
    tuning_report_path: str


def train_d_plus_2_catboost_models() -> DPlus2TrainingSummary:
    tuning_reports: dict[str, dict] = {}

    before_metrics, before_tuning = _train_one_scenario(
        scenario="before_14",
        dataset_path=BEFORE_14_PATH,
        predictions_path=BEFORE_PREDICTIONS_PATH,
        model_path=BEFORE_MODEL_PATH,
    )
    tuning_reports["before_14"] = before_tuning

    after_metrics, after_tuning = _train_one_scenario(
        scenario="after_14",
        dataset_path=AFTER_14_PATH,
        predictions_path=AFTER_PREDICTIONS_PATH,
        model_path=AFTER_MODEL_PATH,
    )
    tuning_reports["after_14"] = after_tuning

    before_metrics["model"] = "catboost"
    after_metrics["model"] = "catboost"
    comparison = pd.DataFrame([before_metrics, after_metrics])
    METRICS_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(METRICS_COMPARISON_PATH, index=False)

    from src.export_d_plus_2_metrics import write_combined_metrics_file

    write_combined_metrics_file()

    TUNING_REPORT_PATH.write_text(
        json.dumps(tuning_reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return DPlus2TrainingSummary(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        comparison_path=project_relative_path(METRICS_COMPARISON_PATH),
        before_predictions_path=project_relative_path(BEFORE_PREDICTIONS_PATH),
        after_predictions_path=project_relative_path(AFTER_PREDICTIONS_PATH),
        tuning_report_path=project_relative_path(TUNING_REPORT_PATH),
    )


def _train_one_scenario(
    scenario: str,
    dataset_path: Path,
    predictions_path: Path,
    model_path: Path,
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    data = _read_dataset(dataset_path)
    feature_columns = _feature_columns(data)
    train_df, val_df, test_df = chronological_train_val_test_split(data)

    y_train = _transform_target(train_df[TARGET_COLUMN]) if USE_LOG_TARGET else train_df[TARGET_COLUMN]
    y_val = _transform_target(val_df[TARGET_COLUMN]) if USE_LOG_TARGET else val_df[TARGET_COLUMN]

    model, tuning_info = fit_tuned_catboost_regressor(
        train_df[feature_columns],
        y_train,
        val_df[feature_columns],
        y_val,
        verbose=0,
    )

    predictions = predict_with_optional_log_transform(
        model,
        test_df[feature_columns],
        use_log_target=USE_LOG_TARGET,
    )
    metrics = _metrics(test_df[TARGET_COLUMN].to_numpy(), predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as file:
        pickle.dump({"model": model, "use_log_target": USE_LOG_TARGET, "feature_columns": feature_columns}, file)

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

    result = {
        "scenario": scenario,
        "rows": int(len(data)),
        "feature_count": int(len(feature_columns)),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "SMAPE": metrics["SMAPE"],
        "R2": metrics["R2"],
        "use_log_target": USE_LOG_TARGET,
        "model_path": project_relative_path(model_path),
        "predictions_path": project_relative_path(predictions_path),
        **{f"tuning_{key}": value for key, value in tuning_info.items()},
    }
    return result, tuning_info


def _transform_target(target: pd.Series) -> pd.Series:
    return np.log1p(target.astype(float))


def _read_dataset(path: Path) -> pd.DataFrame:
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


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }
