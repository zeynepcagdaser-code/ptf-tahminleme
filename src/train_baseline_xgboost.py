from __future__ import annotations

import json
import os
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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN


matplotlib.use("Agg")


@dataclass(frozen=True)
class BaselineModelPaths:
    features_csv: Path = PROJECT_ROOT / "data" / "processed" / "ptf_features.csv"
    model_path: Path = PROJECT_ROOT / "data" / "models" / "xgboost_baseline_model.json"
    predictions_csv: Path = PROJECT_ROOT / "data" / "processed" / "xgboost_baseline_predictions.csv"
    metrics_json: Path = PROJECT_ROOT / "data" / "processed" / "xgboost_baseline_metrics.json"
    figures_dir: Path = PROJECT_ROOT / "data" / "processed" / "figures"


def run_baseline_xgboost(paths: BaselineModelPaths | None = None) -> dict[str, Any]:
    paths = paths or BaselineModelPaths()
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)
    paths.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    print("\nXGBoost baseline egitimi basladi")
    print(f"Girdi: {paths.features_csv}")

    df = load_features(paths.features_csv)
    train_df, test_df = chronological_train_test_split(df, train_ratio=0.8)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = build_model()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    predictions = build_predictions(test_df, y_test, y_pred)
    metrics = calculate_metrics(y_test.to_numpy(), y_pred)
    feature_importance = get_feature_importance(model)

    model.save_model(paths.model_path)
    predictions.to_csv(paths.predictions_csv, index=False)
    save_metrics(
        metrics=metrics,
        paths=paths,
        train_rows=len(train_df),
        test_rows=len(test_df),
        feature_importance=feature_importance,
    )
    save_model_figures(predictions, feature_importance, paths)
    print_training_summary(metrics, len(train_df), len(test_df), feature_importance)

    return metrics


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

    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Model egitimi icin eksik kolonlar: {missing_columns}")

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise ValueError("Veri sizintisi riski: target ptf feature listesinde bulunuyor.")

    return df


def chronological_train_test_split(
    df: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 0 ile 1 arasinda olmali.")

    split_index = int(len(df) * train_ratio)
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Train/test split icin yeterli veri yok.")

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    if train_df["datetime"].max() >= test_df["datetime"].min():
        raise ValueError("Kronolojik split hatali: train test tarihinden sonra veri iceriyor.")

    return train_df, test_df


def build_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def build_predictions(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "datetime": test_df["datetime"].values,
            "actual_ptf": y_test.to_numpy(),
            "predicted_ptf": y_pred,
        }
    )
    predictions["error"] = predictions["actual_ptf"] - predictions["predicted_ptf"]
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    non_zero_mask = y_true != 0
    if non_zero_mask.any():
        mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
    else:
        mape = float("nan")

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape,
        "R2": float(r2_score(y_true, y_pred)),
    }


def get_feature_importance(model: XGBRegressor) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def save_metrics(
    metrics: dict[str, float],
    paths: BaselineModelPaths,
    train_rows: int,
    test_rows: int,
    feature_importance: pd.DataFrame,
) -> None:
    payload = {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "top_10_features": feature_importance.head(10).to_dict(orient="records"),
    }
    with paths.metrics_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_model_figures(
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    paths: BaselineModelPaths,
) -> None:
    save_actual_vs_predicted_png(predictions, paths.figures_dir / "xgboost_actual_vs_predicted.png")
    save_error_distribution_png(predictions, paths.figures_dir / "xgboost_error_distribution.png")
    save_feature_importance_png(feature_importance, paths.figures_dir / "xgboost_feature_importance.png")

    px.line(
        predictions,
        x="datetime",
        y=["actual_ptf", "predicted_ptf"],
        title="XGBoost Actual vs Predicted PTF",
    ).write_html(paths.figures_dir / "xgboost_actual_vs_predicted.html")


def save_actual_vs_predicted_png(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(predictions["datetime"], predictions["actual_ptf"], label="Actual", linewidth=1)
    plt.plot(predictions["datetime"], predictions["predicted_ptf"], label="Predicted", linewidth=1)
    plt.title("XGBoost Actual vs Predicted PTF")
    plt.xlabel("Datetime")
    plt.ylabel("PTF")
    plt.legend()
    plt.grid(alpha=0.3)
    save_matplotlib_figure(output_path)


def save_error_distribution_png(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(predictions["error"], bins=50, edgecolor="black", alpha=0.8)
    plt.title("XGBoost Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    save_matplotlib_figure(output_path)


def save_feature_importance_png(feature_importance: pd.DataFrame, output_path: Path) -> None:
    top_features = feature_importance.head(10).sort_values("importance")
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("XGBoost Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)
    save_matplotlib_figure(output_path)


def save_matplotlib_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def print_training_summary(
    metrics: dict[str, float],
    train_rows: int,
    test_rows: int,
    feature_importance: pd.DataFrame,
) -> None:
    print("\nXGBoost baseline ozeti")
    print("-" * 40)
    print(f"Train satir sayisi      : {train_rows:,}")
    print(f"Test satir sayisi       : {test_rows:,}")
    print(f"Kullanilan feature sayisi: {len(FEATURE_COLUMNS):,}")
    print(f"MAE                     : {metrics['MAE']:,.4f}")
    print(f"RMSE                    : {metrics['RMSE']:,.4f}")
    print(f"MAPE                    : {metrics['MAPE']:,.4f}")
    print(f"R2                      : {metrics['R2']:,.4f}")
    print("\nEn onemli 10 feature")
    print(feature_importance.head(10).to_string(index=False))
