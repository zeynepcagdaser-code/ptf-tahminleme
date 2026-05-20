from __future__ import annotations

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.hybrid_config import (
    FORECAST_12H_PATH,
    HOURLY_PATH,
    SPIKE_METRICS_PATH,
    SPIKE_MODEL_PATH,
    SPIKE_PRED_PATH,
)
from src.model_splits import chronological_train_val_test_split, project_relative_path


TARGET_COLUMN = "ptf_target"
ID_COLS = ["issue_datetime", "target_datetime", "forecast_horizon", TARGET_COLUMN]
SPIKE_STD_MULTIPLIER = 2.0
SPIKE_QUANTILE = 0.90


@dataclass(frozen=True)
class SpikeTrainingSummary:
    rows: int
    spike_rate: float
    test_accuracy: float
    test_auc: float
    model_path: str
    predictions_path: str
    metrics_path: str


def train_spike_classifier() -> SpikeTrainingSummary:
    forecast = _read_forecast()
    hourly = _read_hourly()
    dataset = _attach_spike_labels(forecast, hourly)
    feature_cols = _feature_columns(dataset)

    dataset = dataset.sort_values("issue_datetime").reset_index(drop=True)
    train_df, val_df, test_df = chronological_train_val_test_split(dataset)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    model.fit(train_df[feature_cols], train_df["is_spike"].astype(int))

    all_preds = pd.concat(
        [
            _with_probs(train_df, model, feature_cols),
            _with_probs(val_df, model, feature_cols),
            _with_probs(test_df, model, feature_cols),
        ],
        ignore_index=True,
    )
    all_preds = all_preds.sort_values(["issue_datetime", "forecast_horizon"]).reset_index(drop=True)

    test_probs = all_preds[all_preds["issue_datetime"].isin(test_df["issue_datetime"].unique())]
    test_labels = test_df["is_spike"].astype(int)
    test_pred_probs = model.predict_proba(test_df[feature_cols])[:, 1]
    test_pred_bin = (test_pred_probs >= 0.5).astype(int)

    metrics = {
        "rows": int(len(dataset)),
        "spike_rate": float(dataset["is_spike"].mean()),
        "spike_std_multiplier": SPIKE_STD_MULTIPLIER,
        "spike_quantile": SPIKE_QUANTILE,
        "test_accuracy": float(accuracy_score(test_labels, test_pred_bin)),
        "test_precision": float(precision_score(test_labels, test_pred_bin, zero_division=0)),
        "test_recall": float(recall_score(test_labels, test_pred_bin, zero_division=0)),
        "test_f1": float(f1_score(test_labels, test_pred_bin, zero_division=0)),
        "test_auc": float(roc_auc_score(test_labels, test_pred_probs)) if test_labels.nunique() > 1 else 0.0,
        "feature_columns": feature_cols,
    }

    SPIKE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPIKE_MODEL_PATH.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols}, f)

    export_cols = ID_COLS + ["is_spike", "spike_probability", "spike_threshold"]
    all_preds[export_cols].to_csv(SPIKE_PRED_PATH, index=False)
    SPIKE_METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return SpikeTrainingSummary(
        rows=len(dataset),
        spike_rate=float(dataset["is_spike"].mean()),
        test_accuracy=metrics["test_accuracy"],
        test_auc=metrics["test_auc"],
        model_path=project_relative_path(SPIKE_MODEL_PATH),
        predictions_path=project_relative_path(SPIKE_PRED_PATH),
        metrics_path=project_relative_path(SPIKE_METRICS_PATH),
    )


def _read_forecast() -> pd.DataFrame:
    data = pd.read_csv(FORECAST_12H_PATH)
    data["issue_datetime"] = pd.to_datetime(data["issue_datetime"], errors="coerce")
    data["target_datetime"] = pd.to_datetime(data["target_datetime"], errors="coerce")
    return data.dropna(subset=["issue_datetime", "target_datetime", TARGET_COLUMN])


def _read_hourly() -> pd.DataFrame:
    data = pd.read_csv(HOURLY_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"]).sort_values("datetime")


def _attach_spike_labels(forecast: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.set_index("datetime").sort_index()
    kesin = h["ptf_kesinlesmis"].astype(float)
    h["kesin_median_168"] = kesin.rolling(168, min_periods=24).median()
    h["kesin_std_168"] = kesin.rolling(168, min_periods=24).std()
    h["renewable"] = (
        h.get("wind_generation", 0).fillna(0)
        + h.get("solar_generation", 0).fillna(0)
        + h.get("hydro_dam_generation", 0).fillna(0)
    )
    h["net_load"] = h.get("load_forecast_plan", kesin).astype(float) - h["renewable"]
    h["ptf_ramp_1"] = kesin.diff(1)
    h["ptf_ramp_24"] = kesin.diff(24)

    rows: list[dict] = []
    for _, row in forecast.iterrows():
        issue = row["issue_datetime"]
        hist = kesin.loc[kesin.index <= issue]
        if len(hist) < 48:
            continue

        med168 = float(hist.tail(168).median())
        std168 = float(hist.tail(168).std())
        if np.isnan(std168) or std168 < 1e-6:
            std168 = max(float(hist.tail(168).mean()) * 0.05, 50.0)

        q90 = float(hist.quantile(SPIKE_QUANTILE))
        thresh_std = med168 + SPIKE_STD_MULTIPLIER * std168
        threshold = max(thresh_std, q90)

        actual = float(row[TARGET_COLUMN])
        is_spike = int(actual > threshold)

        issue_feats = h.loc[issue] if issue in h.index else None
        feat = {
            **{k: row[k] for k in row.index if k not in ID_COLS},
            "spike_threshold": threshold,
            "rolling_168_median": med168,
            "rolling_168_std": std168,
            "is_spike": is_spike,
        }
        if issue_feats is not None:
            if isinstance(issue_feats, pd.DataFrame):
                issue_feats = issue_feats.iloc[-1]
            for col in ("renewable", "net_load", "ptf_ramp_1", "ptf_ramp_24", "kesin_median_168"):
                if col in h.columns:
                    feat[f"issue_{col}"] = float(issue_feats[col]) if not pd.isna(issue_feats[col]) else np.nan

        rows.append({**row.to_dict(), **feat})

    return pd.DataFrame(rows)


def _feature_columns(data: pd.DataFrame) -> list[str]:
    exclude = set(ID_COLS + ["is_spike", "spike_threshold", "spike_probability"])
    cols = [c for c in data.columns if c not in exclude and pd.api.types.is_numeric_dtype(data[c])]
    return cols


def _with_probs(df: pd.DataFrame, model: HistGradientBoostingClassifier, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["spike_probability"] = model.predict_proba(out[feature_cols])[:, 1]
    return out
