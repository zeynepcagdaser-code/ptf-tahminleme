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


FEATURE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_feature_dataset.csv"
BASE_IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_final_feature_importance.csv"

SELECTED_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "selected_feature_dataset.csv"
SELECTED_FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "processed" / "selected_feature_list.csv"
REMOVED_FEATURE_LIST_PATH = PROJECT_ROOT / "data" / "processed" / "removed_feature_list.csv"

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "catboost_selected_features_model.pkl"
PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_selected_features_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_selected_features_metrics.json"
IMPORTANCE_PATH = PROJECT_ROOT / "data" / "processed" / "catboost_selected_feature_importance.csv"
FIGURES_DIR = PROJECT_ROOT / "data" / "processed" / "figures"

TARGET_COLUMN = "ptf"
ID_COLUMNS = ["datetime", "date"]
EXCLUDED_COLUMNS = {
    "datetime",
    "date",
    TARGET_COLUMN,
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
TARGET_MIN_FEATURES = 80
TARGET_MAX_FEATURES = 150
INITIAL_CANDIDATE_LIMIT = 180
REDUNDANCY_CORRELATION_THRESHOLD = 0.98
EPSILON = 1.0


@dataclass(frozen=True)
class SelectionRetrainResult:
    initial_feature_count: int
    selected_feature_count: int
    removed_feature_count: int
    train_rows: int
    test_rows: int
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    top_features: list[dict[str, float | str]]
    selected_dataset_path: str
    selected_feature_list_path: str
    removed_feature_list_path: str
    model_path: str
    predictions_path: str
    metrics_path: str
    feature_importance_path: str


def run_feature_selection_and_retrain() -> SelectionRetrainResult:
    dataset = _read_dataset()
    base_importance = _read_base_importance(dataset)
    selected_features, removed_features = _select_features(dataset, base_importance)
    selected_dataset = dataset[ID_COLUMNS + [TARGET_COLUMN] + selected_features].copy()

    SELECTED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    selected_dataset.to_csv(SELECTED_DATASET_PATH, index=False)
    _save_feature_lists(base_importance, selected_features, removed_features)

    train_df, val_df, test_df = chronological_train_val_test_split(selected_dataset)
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.02,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
    )
    model.fit(
        train_df[selected_features],
        train_df[TARGET_COLUMN],
        eval_set=(val_df[selected_features], val_df[TARGET_COLUMN]),
        use_best_model=True,
    )

    predictions = model.predict(test_df[selected_features])
    metrics = _calculate_metrics(test_df[TARGET_COLUMN].to_numpy(), predictions)

    with MODEL_PATH.open("wb") as file:
        pickle.dump(model, file)

    prediction_frame = pd.DataFrame(
        {
            "datetime": test_df["datetime"].to_numpy(),
            "date": test_df["date"].to_numpy(),
            "actual_ptf": test_df[TARGET_COLUMN].to_numpy(),
            "predicted_ptf": predictions,
            "error": test_df[TARGET_COLUMN].to_numpy() - predictions,
            "absolute_error": np.abs(test_df[TARGET_COLUMN].to_numpy() - predictions),
        }
    )
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    retrained_importance = pd.DataFrame(
        {
            "feature": selected_features,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    retrained_importance.to_csv(IMPORTANCE_PATH, index=False)

    top_features = retrained_importance.head(30).to_dict(orient="records")
    metrics_payload = {
        **metrics,
        "initial_feature_count": int(len(base_importance)),
        "selected_feature_count": int(len(selected_features)),
        "removed_feature_count": int(len(base_importance) - len(selected_features)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "top_30_features": top_features,
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_actual_vs_predicted(prediction_frame)
    _plot_error_distribution(prediction_frame)
    _plot_feature_importance(retrained_importance)

    return SelectionRetrainResult(
        initial_feature_count=len(base_importance),
        selected_feature_count=len(selected_features),
        removed_feature_count=len(base_importance) - len(selected_features),
        train_rows=len(train_df),
        test_rows=len(test_df),
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        mape=metrics["MAPE"],
        smape=metrics["SMAPE"],
        r2=metrics["R2"],
        top_features=top_features,
        selected_dataset_path=project_relative_path(SELECTED_DATASET_PATH),
        selected_feature_list_path=project_relative_path(SELECTED_FEATURE_LIST_PATH),
        removed_feature_list_path=project_relative_path(REMOVED_FEATURE_LIST_PATH),
        model_path=project_relative_path(MODEL_PATH),
        predictions_path=project_relative_path(PREDICTIONS_PATH),
        metrics_path=project_relative_path(METRICS_PATH),
        feature_importance_path=project_relative_path(IMPORTANCE_PATH),
    )


def _read_dataset() -> pd.DataFrame:
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(f"Feature dataset bulunamadi: {FEATURE_DATASET_PATH}")

    dataset = pd.read_csv(FEATURE_DATASET_PATH)
    dataset["datetime"] = pd.to_datetime(dataset["datetime"], errors="coerce")
    dataset = dataset.dropna(subset=["datetime", TARGET_COLUMN]).sort_values("datetime").reset_index(drop=True)
    return dataset


def _read_base_importance(dataset: pd.DataFrame) -> pd.DataFrame:
    if not BASE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"CatBoost importance dosyasi bulunamadi: {BASE_IMPORTANCE_PATH}")

    importance = pd.read_csv(BASE_IMPORTANCE_PATH)
    if not {"feature", "importance"}.issubset(importance.columns):
        raise ValueError("Importance dosyasi feature ve importance kolonlarini icermeli.")

    importance = importance.copy()
    importance["importance"] = pd.to_numeric(importance["importance"], errors="coerce").fillna(0.0)
    importance = importance[importance["feature"].isin(dataset.columns)]
    importance = importance[~importance["feature"].isin(EXCLUDED_COLUMNS)]
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)


def _select_features(
    dataset: pd.DataFrame,
    base_importance: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    removed: list[dict[str, str | float]] = []
    importance_map = dict(zip(base_importance["feature"], base_importance["importance"], strict=False))

    positive = base_importance[base_importance["importance"] > 0].copy()
    for _, row in base_importance[base_importance["importance"] <= 0].iterrows():
        removed.append(_removed_row(row["feature"], row["importance"], "zero_or_negative_importance"))

    low_variance_features = _find_low_variance_features(dataset, positive["feature"].tolist())
    for feature in low_variance_features:
        removed.append(_removed_row(feature, importance_map.get(feature, 0.0), "low_variance"))
    positive = positive[~positive["feature"].isin(low_variance_features)]

    preliminary = positive.head(INITIAL_CANDIDATE_LIMIT).copy()
    low_importance = positive[~positive["feature"].isin(preliminary["feature"])]
    for _, row in low_importance.iterrows():
        removed.append(_removed_row(row["feature"], row["importance"], "dynamic_importance_threshold"))

    preliminary_features = preliminary["feature"].tolist()
    preliminary_features = _apply_group_reduction(preliminary_features, importance_map, removed)
    preliminary_features = _apply_redundancy_filter(dataset, preliminary_features, importance_map, removed)

    selected = _resize_selection(preliminary_features, positive["feature"].tolist(), importance_map)
    selected_set = set(selected)

    already_removed = {row["feature"] for row in removed}
    for _, row in positive.iterrows():
        feature = row["feature"]
        if feature not in selected_set and feature not in already_removed:
            removed.append(_removed_row(feature, row["importance"], "final_size_limit_or_redundancy"))

    removed_frame = pd.DataFrame(removed).drop_duplicates(subset=["feature"], keep="first")
    if not removed_frame.empty:
        removed_frame = removed_frame[~removed_frame["feature"].isin(selected_set)]
    return selected, removed_frame


def _find_low_variance_features(dataset: pd.DataFrame, features: list[str]) -> set[str]:
    low_variance: set[str] = set()
    for feature in features:
        series = pd.to_numeric(dataset[feature], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            low_variance.add(feature)
            continue
        if float(series.var(skipna=True)) <= 1e-12:
            low_variance.add(feature)
    return low_variance


def _apply_group_reduction(
    features: list[str],
    importance_map: dict[str, float],
    removed: list[dict[str, str | float]],
) -> list[str]:
    if len(features) <= TARGET_MAX_FEATURES:
        return features

    importance_values = [importance_map[feature] for feature in features]
    median_importance = float(np.median(importance_values))
    weak_lag_tokens = ("_lag_2", "_lag_3", "_lag_6", "_lag_12", "_lag_48", "_lag_72")
    weak_rolling_tokens = ("_rolling_mean_72", "_rolling_mean_168", "_rolling_std_72", "_rolling_std_168")
    weak_volatility_tokens = ("_volatility_", "_variance_", "_zscore_")

    kept: list[str] = []
    for feature in features:
        importance = importance_map[feature]
        weak_group = (
            any(token in feature for token in weak_lag_tokens)
            or any(token in feature for token in weak_rolling_tokens)
            or any(token in feature for token in weak_volatility_tokens)
        )
        if weak_group and importance < median_importance and len(kept) + (len(features) - len(kept)) > TARGET_MIN_FEATURES:
            removed.append(_removed_row(feature, importance, "weak_group_reduction"))
            continue
        kept.append(feature)
    return kept


def _apply_redundancy_filter(
    dataset: pd.DataFrame,
    features: list[str],
    importance_map: dict[str, float],
    removed: list[dict[str, str | float]],
) -> list[str]:
    if len(features) <= 1:
        return features

    data = dataset[features].apply(pd.to_numeric, errors="coerce")
    correlation = data.corr().abs()
    active = set(features)

    ordered_by_low_importance = sorted(features, key=lambda feature: importance_map.get(feature, 0.0))
    for feature in ordered_by_low_importance:
        if feature not in active:
            continue
        high_corr = correlation.index[(correlation[feature] > REDUNDANCY_CORRELATION_THRESHOLD)].tolist()
        high_corr = [other for other in high_corr if other != feature and other in active]
        if not high_corr:
            continue
        better_exists = any(importance_map.get(other, 0.0) >= importance_map.get(feature, 0.0) for other in high_corr)
        if better_exists and len(active) > TARGET_MIN_FEATURES:
            active.remove(feature)
            removed.append(_removed_row(feature, importance_map.get(feature, 0.0), "redundant_correlation_gt_0_98"))

    return [feature for feature in features if feature in active]


def _resize_selection(
    selected: list[str],
    positive_features: list[str],
    importance_map: dict[str, float],
) -> list[str]:
    selected = sorted(dict.fromkeys(selected), key=lambda feature: importance_map.get(feature, 0.0), reverse=True)
    if len(selected) > TARGET_MAX_FEATURES:
        return selected[:TARGET_MAX_FEATURES]

    selected_set = set(selected)
    for feature in positive_features:
        if len(selected) >= TARGET_MIN_FEATURES:
            break
        if feature not in selected_set:
            selected.append(feature)
            selected_set.add(feature)

    return selected


def _save_feature_lists(
    base_importance: pd.DataFrame,
    selected_features: list[str],
    removed_features: pd.DataFrame,
) -> None:
    selected_frame = base_importance[base_importance["feature"].isin(selected_features)].copy()
    selected_frame["rank"] = selected_frame["importance"].rank(method="first", ascending=False).astype(int)
    selected_frame = selected_frame.sort_values("importance", ascending=False)
    selected_frame.to_csv(SELECTED_FEATURE_LIST_PATH, index=False)
    removed_features.sort_values(["reason", "importance"], ascending=[True, False]).to_csv(
        REMOVED_FEATURE_LIST_PATH,
        index=False,
    )


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
    plt.title("CatBoost Selected Features - Actual vs Predicted")
    plt.xlabel("Datetime")
    plt.ylabel("PTF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_selected_actual_vs_predicted.png", dpi=150)
    plt.close()


def _plot_error_distribution(prediction_frame: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(prediction_frame["error"], bins=60, edgecolor="black", alpha=0.75)
    plt.title("CatBoost Selected Features - Error Distribution")
    plt.xlabel("Actual - Predicted")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_selected_error_distribution.png", dpi=150)
    plt.close()


def _plot_feature_importance(importance_frame: pd.DataFrame) -> None:
    top_features = importance_frame.head(30).sort_values("importance", ascending=True)
    plt.figure(figsize=(11, 9))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("CatBoost Selected Features - Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "catboost_selected_feature_importance.png", dpi=150)
    plt.close()


def _removed_row(feature: str, importance: float, reason: str) -> dict[str, str | float]:
    return {"feature": feature, "importance": float(importance), "reason": reason}


def result_to_dict(result: SelectionRetrainResult) -> dict:
    return asdict(result)
