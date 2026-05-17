from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_24h_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "catboost_24h_forecast_model.pkl"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_24h_forecast_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_24h_forecast_metrics.json"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_24h_forecast_feature_importance.csv"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"

TARGET_COLUMN = "ptf_target"
EXCLUDED_COLUMNS = {"datetime", "target_datetime", TARGET_COLUMN}
EPSILON = 1.0


@dataclass(frozen=True)
class Forecast24hTrainingResult:
    total_rows: int
    feature_count: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    smape: float
    r2: float
    top_features: list[dict[str, float | str]]
    model_path: str
    predictions_path: str
    metrics_path: str
    feature_importance_path: str


def train_catboost_24h_forecast() -> Forecast24hTrainingResult:
    data = _read_dataset()
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

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    prediction_frame = pd.DataFrame(
        {
            "datetime": test_df["datetime"].to_numpy(),
            "target_datetime": test_df["target_datetime"].to_numpy(),
            "actual_ptf": test_df[TARGET_COLUMN].to_numpy(),
            "predicted_ptf": predictions,
            "error": test_df[TARGET_COLUMN].to_numpy() - predictions,
            "absolute_error": np.abs(test_df[TARGET_COLUMN].to_numpy() - predictions),
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

    top_features = importance_frame.head(30).to_dict(orient="records")
    payload = {
        **metrics,
        "total_rows": len(data),
        "feature_count": len(feature_columns),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "top_30_features": top_features,
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_actual_vs_predicted(prediction_frame)
    _plot_error_distribution(prediction_frame)
    _plot_feature_importance(importance_frame)

    return Forecast24hTrainingResult(
        total_rows=len(data),
        feature_count=len(feature_columns),
        train_rows=len(train_df),
        test_rows=len(test_df),
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        smape=metrics["SMAPE"],
        r2=metrics["R2"],
        top_features=top_features,
        model_path=str(MODEL_PATH),
        predictions_path=str(PREDICTIONS_PATH),
        metrics_path=str(METRICS_PATH),
        feature_importance_path=str(IMPORTANCE_PATH),
    )


def _read_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"24 saat forecast dataset bulunamadi: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    data = data.dropna(subset=["datetime", "target_datetime", TARGET_COLUMN])
    data = data.sort_values("datetime").reset_index(drop=True)
    return data


def _feature_columns(data: pd.DataFrame) -> list[str]:
    columns = [column for column in data.columns if column not in EXCLUDED_COLUMNS]
    forbidden = {
        "ptf",
        "smf",
        "real_time_consumption",
        "wind_generation",
        "solar_generation",
        "hydro_dam_generation",
        "unlicensed_generation_total",
    }
    leakage_columns = sorted(forbidden.intersection(columns))
    if leakage_columns:
        raise ValueError(f"Leakage riski olan kolonlar feature listesine girdi: {leakage_columns}")

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


def _plot_actual_vs_predicted(prediction_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(prediction_frame["target_datetime"], prediction_frame["actual_ptf"], label="Actual PTF", linewidth=1.2)
    plt.plot(
        prediction_frame["target_datetime"],
        prediction_frame["predicted_ptf"],
        label="Predicted PTF",
        linewidth=1.2,
    )
    plt.title("CatBoost 24h Forecast - Actual vs Predicted")
    plt.xlabel("Target Datetime")
    plt.ylabel("PTF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_24h_forecast_actual_vs_predicted.png", dpi=150)
    plt.close()


def _plot_error_distribution(prediction_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_frame["error"], bins=60, edgecolor="black", alpha=0.75)
    plt.title("CatBoost 24h Forecast - Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_24h_forecast_error_distribution.png", dpi=150)
    plt.close()


def _plot_feature_importance(importance_frame: pd.DataFrame) -> None:
    top_features = importance_frame.head(30).sort_values("importance", ascending=True)
    plt.figure(figsize=(11, 9))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("CatBoost 24h Forecast - Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_24h_forecast_feature_importance.png", dpi=150)
    plt.close()


def result_to_dict(result: Forecast24hTrainingResult) -> dict:
    return asdict(result)
