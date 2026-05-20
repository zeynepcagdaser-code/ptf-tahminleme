from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path
from src.ptf_seasonal_baselines import prepare_kesin_hourly, seasonal_predictions_for_target


FORECAST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "forecast_12h_dataset.csv"
HOURLY_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
CATBOOST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_predictions.csv"

FINAL_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_predictions.csv"
FINAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_metrics.json"
FINAL_HORIZON_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_horizon_metrics.csv"
FINAL_COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_final_model_comparison.csv"

BASELINE_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "ptf_12h_metrics.json"

TARGET_COLUMN = "ptf_target"
ID_COLUMNS = ["issue_datetime", "target_datetime", "forecast_horizon", TARGET_COLUMN]
EPSILON = 1.0

ENSEMBLE_SOURCE_COLS = (
    "pred_catboost",
    "pred_seasonal_blend",
    "pred_kesin_same_hour_yesterday",
    "pred_kesin_same_hour_last_week",
    "pred_kesin_rolling_24h",
    "pred_kesin_rolling_168h",
)

ROLLING_WEIGHT_CAPS = {
    "pred_kesin_rolling_24h": 0.35,
    "pred_rolling_24h": 0.35,
    "pred_kesin_rolling_168h": 0.35,
    "pred_rolling_168h": 0.35,
}

SHORT_HORIZON_MIN_CATBOOST_BLEND = {1: 0.70, 2: 0.55, 3: 0.45}
TREND_PENALTY_LAMBDA = 0.25
MOTION_PREFERENCE_MARGIN = 0.06
MAE_TOLERANCE_FOR_MOTION = 1.12


@dataclass(frozen=True)
class FinalEnsembleSummary:
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
    comparison_path: str


def build_final_ensemble() -> FinalEnsembleSummary:
    baseline_metrics = _load_baseline_metrics()

    forecast_data = _read_forecast_data()
    hourly_data = _read_hourly_data()
    catboost_preds = _read_catboost_predictions()
    simple_preds = _create_simple_baselines(forecast_data, hourly_data)

    all_preds = forecast_data[ID_COLUMNS].copy()
    all_preds = all_preds.merge(catboost_preds, on=ID_COLUMNS[:3], how="left")
    all_preds = all_preds.merge(simple_preds, on=ID_COLUMNS[:3], how="left")

    ensemble_weights: dict[int, dict[str, float]] = {}
    blend_weights: dict[int, float] = {}
    bias_corrections: dict[int, float] = {}
    primary_model_by_horizon: dict[int, str] = {}
    clip_bounds: dict[int, dict[str, float]] = {}

    for horizon in range(1, 13):
        print(f"\n=== Optimizing ensemble for horizon {horizon}h ===")
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        horizon_data = horizon_data.sort_values("issue_datetime").reset_index(drop=True)

        _, val_df, _ = chronological_train_val_test_split(horizon_data)
        pred_cols = [c for c in ENSEMBLE_SOURCE_COLS if c in horizon_data.columns]

        weights, bias = _fit_stacked_ensemble(val_df, pred_cols)
        weights = _apply_weight_caps(weights)
        catboost_w, blend_bias = _fit_catboost_seasonal_blend(val_df, horizon)

        ensemble_weights[horizon] = weights
        blend_weights[horizon] = catboost_w
        bias_corrections[horizon] = blend_bias

        primary, val_score = _select_primary_on_validation(val_df, horizon, weights, bias, catboost_w, blend_bias)
        primary_model_by_horizon[horizon] = primary

        val_primary = _predict_with_strategy(val_df, horizon, primary, weights, bias, catboost_w, blend_bias)
        val_primary = val_primary[~np.isnan(val_primary)]
        if horizon <= 3:
            clip_bounds[horizon] = {"lower": 0.0, "upper": 15000.0}
        elif len(val_primary) > 0:
            clip_bounds[horizon] = {
                "lower": max(0.0, float(np.percentile(val_primary, 0.5))),
                "upper": float(np.percentile(val_primary, 99.5)),
            }
        else:
            clip_bounds[horizon] = {"lower": 0.0, "upper": 5000.0}

        print(f"  CatBoost blend weight: {catboost_w:.2f}")
        print(f"  Primary model: {primary} (val composite score: {val_score:.2f})")

    final_predictions = []
    for horizon in range(1, 13):
        horizon_data = all_preds[all_preds["forecast_horizon"] == horizon].copy()
        primary = primary_model_by_horizon.get(horizon, "blend")

        panel_pred = _predict_with_strategy(
            horizon_data,
            horizon,
            primary,
            ensemble_weights[horizon],
            bias_corrections[horizon],
            blend_weights[horizon],
            bias_corrections[horizon],
        )
        stacked_pred = _predict_with_strategy(
            horizon_data,
            horizon,
            "stacked",
            ensemble_weights[horizon],
            bias_corrections[horizon],
            blend_weights[horizon],
            bias_corrections[horizon],
        )

        catboost = horizon_data["pred_catboost"].fillna(horizon_data["pred_seasonal_blend"]).to_numpy()
        seasonal = horizon_data["pred_seasonal_blend"].fillna(horizon_data["pred_catboost"]).to_numpy()

        panel_pred = np.where(np.isnan(panel_pred), catboost, panel_pred)
        panel_pred = np.clip(
            panel_pred,
            clip_bounds[horizon]["lower"],
            clip_bounds[horizon]["upper"],
        )

        horizon_data["pred_catboost_export"] = catboost
        horizon_data["pred_seasonal_blend_export"] = seasonal
        horizon_data["final_predicted_ptf"] = stacked_pred
        horizon_data["panel_predicted_ptf"] = panel_pred
        horizon_data["primary_model"] = primary
        horizon_data["absolute_error"] = np.abs(horizon_data[TARGET_COLUMN] - horizon_data["panel_predicted_ptf"])
        final_predictions.append(horizon_data)

    final_df = pd.concat(final_predictions, ignore_index=True)
    final_df = final_df.dropna(subset=["panel_predicted_ptf"])

    FINAL_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_PREDICTIONS_PATH, index=False)

    overall = _metrics(
        final_df[TARGET_COLUMN].to_numpy(),
        final_df["panel_predicted_ptf"].to_numpy(),
    )
    horizon_metrics = _horizon_metrics(final_df)
    horizon_metrics.to_csv(FINAL_HORIZON_METRICS_PATH, index=False)

    comparison = _create_comparison(final_df, catboost_preds, baseline_metrics)
    comparison.to_csv(FINAL_COMPARISON_PATH, index=False)

    payload = {
        **overall,
        "baseline_comparison": {
            "mae": baseline_metrics.get("MAE"),
            "rmse": baseline_metrics.get("RMSE"),
            "smape": baseline_metrics.get("SMAPE"),
            "r2": baseline_metrics.get("R2"),
        },
        "ensemble_weights": ensemble_weights,
        "blend_weights": blend_weights,
        "bias_corrections": bias_corrections,
        "primary_model_by_horizon": primary_model_by_horizon,
        "clip_bounds": clip_bounds,
        "trend_penalty_lambda": TREND_PENALTY_LAMBDA,
    }
    FINAL_METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    mae_improvement = (baseline_metrics.get("MAE", 0) - overall["MAE"]) / baseline_metrics.get("MAE", 1) * 100
    rmse_improvement = (baseline_metrics.get("RMSE", 0) - overall["RMSE"]) / baseline_metrics.get("RMSE", 1) * 100
    smape_improvement = (baseline_metrics.get("SMAPE", 0) - overall["SMAPE"]) / baseline_metrics.get("SMAPE", 1) * 100
    r2_improvement = overall["R2"] - baseline_metrics.get("R2", 0)

    return FinalEnsembleSummary(
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
        predictions_path=project_relative_path(FINAL_PREDICTIONS_PATH),
        metrics_path=project_relative_path(FINAL_METRICS_PATH),
        horizon_metrics_path=project_relative_path(FINAL_HORIZON_METRICS_PATH),
        comparison_path=project_relative_path(FINAL_COMPARISON_PATH),
    )


def _trend_penalty(y: np.ndarray, pred: np.ndarray) -> float:
    """Fark serileri arasindaki korelasyon eksikligi (0=iyi, 1=kotu)."""
    dy = np.diff(y.astype(float))
    dp = np.diff(pred.astype(float))
    if len(dy) < 3 or np.std(dy) < 1e-6:
        return 0.0
    if np.std(dp) < 1e-6:
        return 1.0
    corr = float(np.corrcoef(dy, dp)[0, 1])
    if np.isnan(corr):
        return 1.0
    return max(0.0, 1.0 - corr)


def _motion_score(y: np.ndarray, pred: np.ndarray) -> float:
    dy = np.diff(y.astype(float))
    dp = np.diff(pred.astype(float))
    if len(dy) < 2 or np.std(dy) < 1e-6 or np.std(dp) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(dy, dp)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _composite_objective(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.nan_to_num(pred, nan=0.0)
    mae = mean_absolute_error(y, pred)
    trend_pen = _trend_penalty(y, pred)
    scale = max(float(np.std(y)), 50.0)
    return mae + TREND_PENALTY_LAMBDA * trend_pen * scale


def _apply_weight_caps(weights: dict[str, float]) -> dict[str, float]:
    capped = dict(weights)
    for col, cap in ROLLING_WEIGHT_CAPS.items():
        if col in capped:
            capped[col] = min(capped[col], cap)
    total = sum(v for k, v in capped.items() if not k.startswith("_"))
    if total > 1e-6:
        for k in capped:
            if not k.startswith("_"):
                capped[k] = capped[k] / total
    return capped


def _load_baseline_metrics() -> dict[str, float]:
    if BASELINE_METRICS_PATH.exists():
        with open(BASELINE_METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"MAE": 0, "RMSE": 0, "SMAPE": 0, "R2": 0}


def _read_forecast_data() -> pd.DataFrame:
    if not FORECAST_DATA_PATH.exists():
        raise FileNotFoundError(f"Forecast dataset not found: {FORECAST_DATA_PATH}")
    data = pd.read_csv(FORECAST_DATA_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN]).reset_index(drop=True)


def _read_hourly_data() -> pd.DataFrame:
    if not HOURLY_DATA_PATH.exists():
        raise FileNotFoundError(f"Hourly dataset not found: {HOURLY_DATA_PATH}")
    data = pd.read_csv(HOURLY_DATA_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)


def _read_catboost_predictions() -> pd.DataFrame:
    if not CATBOOST_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"CatBoost predictions not found: {CATBOOST_PREDICTIONS_PATH}")
    data = pd.read_csv(CATBOOST_PREDICTIONS_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data[["issue_datetime", "target_datetime", "forecast_horizon", "predicted_ptf"]].rename(
        columns={"predicted_ptf": "pred_catboost"}
    )


def _create_simple_baselines(forecast_data: pd.DataFrame, hourly_data: pd.DataFrame) -> pd.DataFrame:
    hourly = prepare_kesin_hourly(hourly_data)
    baselines = []
    for _, row in forecast_data.iterrows():
        issue_dt = row["issue_datetime"]
        target_dt = row["target_datetime"]
        seasonal = seasonal_predictions_for_target(hourly, issue_dt, target_dt)
        baselines.append(
            {
                "issue_datetime": issue_dt,
                "target_datetime": target_dt,
                "forecast_horizon": row["forecast_horizon"],
                **seasonal,
                "pred_same_hour_yesterday": seasonal["pred_kesin_same_hour_yesterday"],
                "pred_same_hour_last_week": seasonal["pred_kesin_same_hour_last_week"],
                "pred_last_24h_mean": seasonal["pred_kesin_last_24h_mean"],
                "pred_rolling_24h": seasonal["pred_kesin_rolling_24h"],
                "pred_rolling_168h": seasonal["pred_kesin_rolling_168h"],
            }
        )
    return pd.DataFrame(baselines)


def _fit_stacked_ensemble(val_df: pd.DataFrame, pred_cols: list[str]) -> tuple[dict[str, float], float]:
    available_cols = [col for col in pred_cols if val_df[col].notna().any()]
    weights_dict = {col: 0.0 for col in pred_cols}

    if not available_cols:
        return weights_dict, 0.0

    if len(available_cols) == 1:
        col = available_cols[0]
        weights_dict[col] = 1.0
        bias = float((val_df[TARGET_COLUMN] - val_df[col]).mean())
        return weights_dict, bias

    X = val_df[available_cols].fillna(val_df[available_cols].median()).to_numpy()
    y = val_df[TARGET_COLUMN].to_numpy()

    reg = LinearRegression(positive=True, fit_intercept=True)
    reg.fit(X, y)

    coef_sum = float(reg.coef_.sum())
    if coef_sum > 1e-6:
        for col, weight in zip(available_cols, reg.coef_ / coef_sum):
            weights_dict[col] = float(weight)
        bias = float(reg.intercept_)
    else:
        weights_dict[available_cols[0]] = 1.0
        bias = float((y - X[:, 0]).mean())

    return weights_dict, bias


def _apply_ensemble(df: pd.DataFrame, pred_cols: list[str], weights: dict[str, float], bias: float) -> np.ndarray:
    X = df[pred_cols].fillna(df[pred_cols].median()).to_numpy()
    weight_array = np.array([weights.get(col, 0.0) for col in pred_cols])
    return np.dot(X, weight_array) + bias


def _hybrid_prediction(df: pd.DataFrame, catboost_weight: float, bias: float) -> np.ndarray:
    catboost = df["pred_catboost"].fillna(df["pred_seasonal_blend"]).to_numpy()
    seasonal = df["pred_seasonal_blend"].fillna(df["pred_catboost"]).to_numpy()
    return catboost_weight * catboost + (1.0 - catboost_weight) * seasonal + bias


def _fit_catboost_seasonal_blend(val_df: pd.DataFrame, horizon: int) -> tuple[float, float]:
    y = val_df[TARGET_COLUMN].to_numpy()
    catboost = val_df["pred_catboost"].fillna(val_df["pred_seasonal_blend"]).to_numpy()
    seasonal = val_df["pred_seasonal_blend"].fillna(val_df["pred_catboost"]).to_numpy()

    min_w = SHORT_HORIZON_MIN_CATBOOST_BLEND.get(horizon, 0.0)
    best_w = max(min_w, 1.0)
    best_score = float("inf")

    for w in np.linspace(min_w, 1.0, 41):
        pred = w * catboost + (1.0 - w) * seasonal
        score = _composite_objective(y, pred)
        if score < best_score:
            best_score = score
            best_w = float(w)

    blend_pred = best_w * catboost + (1.0 - best_w) * seasonal
    bias = float(np.mean(y - blend_pred))
    return best_w, bias


def _predict_with_strategy(
    df: pd.DataFrame,
    horizon: int,
    strategy: str,
    weights: dict[str, float],
    stacked_bias: float,
    blend_w: float,
    blend_bias: float,
) -> np.ndarray:
    pred_cols = [c for c in ENSEMBLE_SOURCE_COLS if c in df.columns]
    capped_weights = _apply_weight_caps(weights)

    if strategy == "catboost":
        return df["pred_catboost"].fillna(df["pred_seasonal_blend"]).to_numpy()
    if strategy == "seasonal":
        return df["pred_seasonal_blend"].fillna(df["pred_catboost"]).to_numpy()
    if strategy == "blend":
        return _hybrid_prediction(df, blend_w, blend_bias)
    return _apply_ensemble(df, pred_cols, capped_weights, stacked_bias)


def _select_primary_on_validation(
    val_df: pd.DataFrame,
    horizon: int,
    weights: dict[str, float],
    stacked_bias: float,
    blend_w: float,
    blend_bias: float,
) -> tuple[str, float]:
    y = val_df[TARGET_COLUMN].to_numpy()
    candidates = {
        "catboost": _predict_with_strategy(val_df, horizon, "catboost", weights, stacked_bias, blend_w, blend_bias),
        "seasonal": _predict_with_strategy(val_df, horizon, "seasonal", weights, stacked_bias, blend_w, blend_bias),
        "blend": _predict_with_strategy(val_df, horizon, "blend", weights, stacked_bias, blend_w, blend_bias),
        "stacked": _predict_with_strategy(val_df, horizon, "stacked", weights, stacked_bias, blend_w, blend_bias),
    }

    scores = {name: _composite_objective(y, pred) for name, pred in candidates.items()}
    motions = {name: _motion_score(y, pred) for name, pred in candidates.items()}

    best_name = min(scores, key=scores.get)
    best_score = scores[best_name]

    cb_score = scores["catboost"]
    cb_motion = motions["catboost"]

    for name in candidates:
        if name == "catboost":
            continue
        if cb_motion >= motions[name] + MOTION_PREFERENCE_MARGIN:
            if cb_score <= scores[name] * MAE_TOLERANCE_FOR_MOTION:
                return "catboost", cb_score

    if horizon <= 3:
        if cb_score <= best_score * MAE_TOLERANCE_FOR_MOTION and cb_motion >= motions.get(best_name, 0.0) - 0.02:
            return "catboost", cb_score

    return best_name, best_score


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
        metrics = _metrics(group[TARGET_COLUMN].to_numpy(), group["panel_predicted_ptf"].to_numpy())
        rows.append({"forecast_horizon": int(horizon), **metrics})
    return pd.DataFrame(rows).sort_values("forecast_horizon")


def _create_comparison(final_df: pd.DataFrame, catboost_preds: pd.DataFrame, baseline_metrics: dict) -> pd.DataFrame:
    comparison = final_df[
        [
            "issue_datetime",
            "target_datetime",
            "forecast_horizon",
            TARGET_COLUMN,
            "panel_predicted_ptf",
            "pred_catboost_export",
            "pred_seasonal_blend_export",
            "primary_model",
        ]
    ].copy()
    comparison = comparison.rename(
        columns={
            "panel_predicted_ptf": "panel_ptf",
            "pred_catboost_export": "pred_catboost",
            "pred_seasonal_blend_export": "pred_seasonal_blend",
        }
    )
    comparison["catboost_error"] = np.abs(comparison[TARGET_COLUMN] - comparison["pred_catboost"])
    comparison["panel_error"] = np.abs(comparison[TARGET_COLUMN] - comparison["panel_ptf"])
    comparison["panel_better"] = comparison["panel_error"] < comparison["catboost_error"]
    return comparison
