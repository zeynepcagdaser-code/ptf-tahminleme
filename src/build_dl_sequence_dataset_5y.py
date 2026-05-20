from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.dl_5y_config import (
    FEATURE_SUMMARY_PATH,
    HOURLY_5Y_PATH,
    INPUT_WINDOW_5Y,
    OUTPUT_HORIZON_5Y,
    PRICE_COLUMN_5Y,
    SCALERS_5Y_PATH,
    SEQUENCE_NPZ_PATH,
    TABULAR_5Y_PATH,
    TRAIN_RATIO_5Y,
    VAL_RATIO_5Y,
)


EXCLUDE_FEATURE_COLUMNS = {
    "datetime",
    "date",
    "hour",
    "ptf_kesinlesmis",
    "ptf_interim",
    "ges_forecast",
    "yek_portfolio_income",
}


def build_dl_sequence_dataset_5y() -> dict[str, Any]:
    if not HOURLY_5Y_PATH.exists():
        raise FileNotFoundError(f"Once build_hourly_dataset_5y: {HOURLY_5Y_PATH}")

    df = pd.read_csv(HOURLY_5Y_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    feature_cols = _select_feature_columns(df)
    df = _impute_for_sequences(df, feature_cols)

    X_raw, y_raw, cutoffs = _build_sequences(df, feature_cols)
    if len(X_raw) < 500:
        raise ValueError(f"Yetersiz sequence ornegi: {len(X_raw)} (min 500). 5y fetch gerekebilir.")

    n = len(X_raw)
    train_end = int(n * TRAIN_RATIO_5Y)
    val_end = train_end + int(n * VAL_RATIO_5Y)

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_train = X_raw[:train_end]
    y_train = y_raw[:train_end]
    X_val = X_raw[train_end:val_end]
    y_val = y_raw[train_end:val_end]
    X_test = X_raw[val_end:]
    y_test = y_raw[val_end:]

    x_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler.fit(y_train.reshape(-1, 1))

    def scale_x(x):
        sh = x.shape
        flat = x_scaler.transform(x.reshape(-1, sh[-1]))
        return flat.reshape(sh)

    def scale_y(y):
        return y_scaler.transform(y.reshape(-1, 1)).reshape(y.shape)

    payload = {
        "X_train": scale_x(X_train).astype(np.float32),
        "y_train": scale_y(y_train).astype(np.float32),
        "X_val": scale_x(X_val).astype(np.float32),
        "y_val": scale_y(y_val).astype(np.float32),
        "X_test": scale_x(X_test).astype(np.float32),
        "y_test": scale_y(y_test).astype(np.float32),
        "feature_names": np.array(feature_cols),
        "input_window": INPUT_WINDOW_5Y,
        "output_horizon": OUTPUT_HORIZON_5Y,
        "train_cutoff_end": str(cutoffs[train_end - 1]) if train_end else "",
        "val_cutoff_end": str(cutoffs[val_end - 1]) if val_end else "",
        "n_samples": n,
        "n_features": len(feature_cols),
    }

    SEQUENCE_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEQUENCE_NPZ_PATH, **payload)

    scalers = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": feature_cols,
        "price_column": PRICE_COLUMN_5Y,
        "input_window": INPUT_WINDOW_5Y,
        "output_horizon": OUTPUT_HORIZON_5Y,
        "train_samples": train_end,
        "val_samples": val_end - train_end,
        "test_samples": n - val_end,
    }
    SCALERS_5Y_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SCALERS_5Y_PATH.open("wb") as f:
        pickle.dump(scalers, f)

    tabular = _build_tabular(df, feature_cols, cutoffs)
    tabular.to_csv(TABULAR_5Y_PATH, index=False)

    summary = {
        "n_samples": n,
        "n_features": len(feature_cols),
        "train_samples": train_end,
        "val_samples": val_end - train_end,
        "test_samples": n - val_end,
        "sequence_npz": str(SEQUENCE_NPZ_PATH),
        "tabular_csv": str(TABULAR_5Y_PATH),
        "scalers_pkl": str(SCALERS_5Y_PATH),
        "feature_names": feature_cols,
    }
    return summary


def _impute_for_sequences(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    out[PRICE_COLUMN_5Y] = pd.to_numeric(out[PRICE_COLUMN_5Y], errors="coerce")
    out[PRICE_COLUMN_5Y] = out[PRICE_COLUMN_5Y].interpolate(method="linear", limit=6).ffill(limit=24).bfill(limit=24)

    for col in feature_cols:
        if col == PRICE_COLUMN_5Y:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].interpolate(method="linear", limit=12, limit_direction="both")
        out[col] = out[col].ffill(limit=48).bfill(limit=48)
        med = out[col].median()
        out[col] = out[col].fillna(med if pd.notna(med) else 0.0)

    return out


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_FEATURE_COLUMNS:
            continue
        if c == PRICE_COLUMN_5Y:
            cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    price = df[PRICE_COLUMN_5Y].to_numpy(dtype=np.float32)
    times = df["datetime"].astype(str).tolist()

    w = INPUT_WINDOW_5Y
    h = OUTPUT_HORIZON_5Y
    n = len(df) - w - h
    if n <= 0:
        raise ValueError("Sequence ornegi olusturulamadi — veri cok kisa.")

    X = np.zeros((n, w, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n, h), dtype=np.float32)
    cutoffs: list[str] = []

    for i in range(n):
        X[i] = values[i : i + w]
        y[i] = price[i + w : i + w + h]
        cutoffs.append(times[i + w - 1])

        if np.isnan(X[i]).any() or np.isnan(y[i]).any():
            X[i] = np.nan
            y[i] = np.nan

    mask = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
    return X[mask], y[mask], [cutoffs[i] for i in range(n) if mask[i]]


def _build_tabular(df: pd.DataFrame, feature_cols: list[str], cutoffs: list[str]) -> pd.DataFrame:
    """Her sequence cutoff için son saat feature snapshot + 12h hedef özeti."""
    w = INPUT_WINDOW_5Y
    h = OUTPUT_HORIZON_5Y
    rows = []
    valid_positions = []
    for i in range(len(df) - w - h):
        end = i + w
        window = df.iloc[i:end][feature_cols]
        future = df.iloc[end : end + h][PRICE_COLUMN_5Y]
        if window.isna().any().any() or future.isna().any():
            continue
        valid_positions.append(end - 1)

    for pos in valid_positions[: len(cutoffs)]:
        row = {"cutoff_datetime": df.loc[pos, "datetime"]}
        for c in feature_cols:
            row[f"last_{c}"] = df.loc[pos, c]
        future = df.loc[pos + 1 : pos + h, PRICE_COLUMN_5Y].to_numpy(dtype=float)
        row["target_mean_12h"] = float(np.mean(future))
        row["target_max_12h"] = float(np.max(future))
        rows.append(row)
    return pd.DataFrame(rows)
