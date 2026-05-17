from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_DIR = PROCESSED_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


matplotlib.use("Agg")


MODEL_SPECS = {
    "xgboost": {
        "display_name": "XGBoost",
        "factory": lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        ),
    },
    "lightgbm": {
        "display_name": "LightGBM",
        "factory": lambda: LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            objective="regression",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
    },
    "randomforest": {
        "display_name": "RandomForest",
        "factory": lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        ),
    },
    "catboost": {
        "display_name": "CatBoost",
        "factory": lambda: CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=0,
        ),
    },
}


@dataclass(frozen=True)
class MultipleModelPaths:
    features_csv: Path = PROCESSED_DIR / "ptf_features.csv"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    predictions_dir: Path = PROCESSED_DIR / "model_predictions"
    metrics_dir: Path = PROCESSED_DIR / "model_metrics"
    figures_dir: Path = PROCESSED_DIR / "figures"
    all_results_csv: Path = PROCESSED_DIR / "all_model_results.csv"
    best_model_path: Path = PROJECT_ROOT / "data" / "models" / "best_model.pkl"


def run_multiple_model_training(paths: MultipleModelPaths | None = None) -> pd.DataFrame:
    paths = paths or MultipleModelPaths()
    create_output_dirs(paths)

    print("\nCoklu model egitimi basladi")
    print(f"Girdi: {paths.features_csv}")

    df = load_features(paths.features_csv)
    train_df, test_df = chronological_train_test_split(df, train_ratio=0.8)
    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print(f"Train satir sayisi: {len(train_df):,}")
    print(f"Test satir sayisi : {len(test_df):,}")
    print(f"Feature sayisi    : {len(FEATURE_COLUMNS):,}")

    trained_models: dict[str, Any] = {}
    results: list[dict[str, Any]] = []

    for model_key, spec in MODEL_SPECS.items():
        model_name = str(spec["display_name"])
        model = spec["factory"]()
        result = train_one_model(
            model_key=model_key,
            model_name=model_name,
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            test_datetimes=test_df["datetime"],
            paths=paths,
        )
        trained_models[model_key] = model
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df[["model", "MAE", "RMSE", "MAPE", "SMAPE", "R2", "train_time_seconds"]]
    results_df.to_csv(paths.all_results_csv, index=False)

    best_row = select_best_model(results_df)
    best_key = model_name_to_key(str(best_row["model"]))
    joblib.dump(trained_models[best_key], paths.best_model_path)

    save_comparison_figures(results_df, paths)
    print_comparison_summary(results_df, best_row)

    return results_df


def create_output_dirs(paths: MultipleModelPaths) -> None:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.predictions_dir.mkdir(parents=True, exist_ok=True)
    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def load_features(features_csv: Path) -> pd.DataFrame:
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Feature dosyasi bulunamadi: {features_csv}\n"
            "Once feature engineering adimini calistirin."
        )

    df = pd.read_csv(features_csv)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Coklu model egitimi icin eksik kolonlar: {missing_columns}")

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise ValueError("Veri sizintisi riski: target ptf feature listesinde bulunuyor.")

    if df[required_columns].isna().any().any():
        raise ValueError("Feature veya target kolonlarinda NaN deger var.")

    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(df) * train_ratio)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Kronolojik train/test split icin yeterli veri yok.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df["datetime"].max() >= test_df["datetime"].min():
        raise ValueError("Kronolojik split hatali: train verisi test zamanina tasiyor.")

    return train_df, test_df


def train_one_model(
    model_key: str,
    model_name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_datetimes: pd.Series,
    paths: MultipleModelPaths,
) -> dict[str, Any]:
    print(f"\n{model_name} egitiliyor...")
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_time

    y_pred = model.predict(x_test)
    metrics = calculate_metrics(y_test.to_numpy(), np.asarray(y_pred, dtype=float))
    predictions = build_predictions(test_datetimes, y_test, np.asarray(y_pred, dtype=float))

    save_model(model, model_key, paths)
    save_predictions(predictions, model_key, paths)
    save_model_metrics(model_name, metrics, train_time, paths, model_key)
    save_actual_vs_predicted_png(predictions, model_key, model_name, paths)

    print_model_summary(model_name, train_time, metrics)

    return {
        "model": model_name,
        **metrics,
        "train_time_seconds": train_time,
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    epsilon = max(float(np.mean(np.abs(y_true))) * 0.01, 1.0)
    denominator = np.maximum(np.abs(y_true), epsilon)
    mape = np.mean(np.abs((y_true - y_pred) / denominator)) * 100

    smape_denominator = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    smape = np.mean(np.abs(y_true - y_pred) / smape_denominator) * 100

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(mape),
        "SMAPE": float(smape),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_predictions(
    datetimes: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "datetime": datetimes.values,
            "actual_ptf": y_true.to_numpy(),
            "predicted_ptf": y_pred,
        }
    )
    predictions["error"] = predictions["actual_ptf"] - predictions["predicted_ptf"]
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions


def save_model(model: Any, model_key: str, paths: MultipleModelPaths) -> None:
    model_path = paths.models_dir / f"{model_key}_model.pkl"
    joblib.dump(model, model_path)

    if model_key == "xgboost":
        model.save_model(paths.models_dir / "xgboost_model.json")
    elif model_key == "lightgbm" and isinstance(model, LGBMRegressor):
        model.booster_.save_model(paths.models_dir / "lightgbm_model.txt")
    elif model_key == "catboost" and isinstance(model, CatBoostRegressor):
        model.save_model(paths.models_dir / "catboost_model.cbm")


def save_predictions(predictions: pd.DataFrame, model_key: str, paths: MultipleModelPaths) -> None:
    predictions.to_csv(paths.predictions_dir / f"{model_key}_predictions.csv", index=False)


def save_model_metrics(
    model_name: str,
    metrics: dict[str, float],
    train_time: float,
    paths: MultipleModelPaths,
    model_key: str,
) -> None:
    payload = {
        "model": model_name,
        "train_time_seconds": train_time,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
    }
    with (paths.metrics_dir / f"{model_key}_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_actual_vs_predicted_png(
    predictions: pd.DataFrame,
    model_key: str,
    model_name: str,
    paths: MultipleModelPaths,
) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(predictions["datetime"], predictions["actual_ptf"], label="Actual", linewidth=1)
    plt.plot(predictions["datetime"], predictions["predicted_ptf"], label="Predicted", linewidth=1)
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.xlabel("Datetime")
    plt.ylabel("PTF")
    plt.legend()
    plt.grid(alpha=0.3)
    save_matplotlib_figure(paths.figures_dir / f"actual_vs_predicted_{model_key}.png")


def save_comparison_figures(results_df: pd.DataFrame, paths: MultipleModelPaths) -> None:
    sorted_df = results_df.sort_values(["R2", "RMSE"], ascending=[False, True]).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_df["model"], sorted_df["RMSE"])
    plt.title("Model Comparison - RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.grid(axis="y", alpha=0.3)
    save_matplotlib_figure(paths.figures_dir / "model_comparison_barplot.png")

    long_df = results_df.melt(
        id_vars="model",
        value_vars=["MAE", "RMSE", "MAPE", "SMAPE", "R2"],
        var_name="metric",
        value_name="value",
    )
    px.bar(
        long_df,
        x="model",
        y="value",
        color="metric",
        barmode="group",
        title="Model Comparison Metrics",
    ).write_html(paths.figures_dir / "model_comparison_barplot.html")


def save_matplotlib_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def select_best_model(results_df: pd.DataFrame) -> pd.Series:
    return results_df.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]


def model_name_to_key(model_name: str) -> str:
    normalized = model_name.lower()
    for key, spec in MODEL_SPECS.items():
        if str(spec["display_name"]).lower() == normalized:
            return key
    raise ValueError(f"Bilinmeyen model adi: {model_name}")


def print_model_summary(model_name: str, train_time: float, metrics: dict[str, float]) -> None:
    print(f"{model_name} train suresi: {train_time:.2f} sn")
    print(f"  MAE  : {metrics['MAE']:,.4f}")
    print(f"  RMSE : {metrics['RMSE']:,.4f}")
    print(f"  SMAPE: {metrics['SMAPE']:,.4f}")
    print(f"  R2   : {metrics['R2']:,.4f}")


def print_comparison_summary(results_df: pd.DataFrame, best_row: pd.Series) -> None:
    print("\nModel karsilastirma tablosu")
    print("-" * 40)
    print(results_df[["model", "MAE", "RMSE", "MAPE", "SMAPE", "R2"]].to_string(index=False))

    print("\nEn iyi model")
    print("-" * 40)
    print(f"Model          : {best_row['model']}")
    print(f"En iyi R2      : {best_row['R2']:,.4f}")
    print(f"En dusuk RMSE  : {best_row['RMSE']:,.4f}")
