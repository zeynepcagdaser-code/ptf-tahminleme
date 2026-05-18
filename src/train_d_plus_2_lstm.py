from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from src.build_d_plus_2_forecast_datasets import (
    AFTER_TARGET_OFFSET_DAYS,
    BEFORE_TARGET_OFFSET_DAYS,
    CUTOFF_HOUR,
)
from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

BEFORE_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_lstm_before_14_predictions.csv"
AFTER_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_lstm_after_14_predictions.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_lstm_metrics_comparison.csv"
ALL_MODELS_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "d_plus_2_all_models_metrics.csv"

BEFORE_MODEL_PATH = MODELS_DIR / "d_plus_2_lstm_before_14.keras"
AFTER_MODEL_PATH = MODELS_DIR / "d_plus_2_lstm_after_14.keras"

SEQUENCE_FEATURES = [
    "ptf",
    "smf",
    "real_time_consumption",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "load_forecast_plan",
    "usd_try",
]
SEQ_LEN = 96
EPSILON = 1.0
EPOCHS = 40
BATCH_SIZE = 128


@dataclass(frozen=True)
class LstmTrainingSummary:
    before_metrics: dict[str, float | int | str]
    after_metrics: dict[str, float | int | str]
    metrics_path: str
    all_models_metrics_path: str
    before_predictions_path: str
    after_predictions_path: str


def train_d_plus_2_lstm_models() -> LstmTrainingSummary:
    from tensorflow import keras
    from tensorflow.keras import layers

    hourly = _read_hourly_dataset()
    by_datetime = hourly.set_index("datetime").sort_index()
    available_dates = sorted(hourly["date"].drop_duplicates())

    before_metrics = _train_one_scenario(
        scenario="before_14",
        target_offset_days=BEFORE_TARGET_OFFSET_DAYS,
        by_datetime=by_datetime,
        available_dates=available_dates,
        predictions_path=BEFORE_PREDICTIONS_PATH,
        model_path=BEFORE_MODEL_PATH,
        keras=keras,
        layers=layers,
    )
    after_metrics = _train_one_scenario(
        scenario="after_14",
        target_offset_days=AFTER_TARGET_OFFSET_DAYS,
        by_datetime=by_datetime,
        available_dates=available_dates,
        predictions_path=AFTER_PREDICTIONS_PATH,
        model_path=AFTER_MODEL_PATH,
        keras=keras,
        layers=layers,
    )

    lstm_metrics = pd.DataFrame([before_metrics, after_metrics])
    if "model" not in lstm_metrics.columns:
        lstm_metrics.insert(0, "model", "lstm")
    lstm_metrics.to_csv(METRICS_PATH, index=False)

    from src.export_d_plus_2_metrics import write_combined_metrics_file

    write_combined_metrics_file()

    return LstmTrainingSummary(
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        metrics_path=project_relative_path(METRICS_PATH),
        all_models_metrics_path=project_relative_path(ALL_MODELS_METRICS_PATH),
        before_predictions_path=project_relative_path(BEFORE_PREDICTIONS_PATH),
        after_predictions_path=project_relative_path(AFTER_PREDICTIONS_PATH),
    )


def _train_one_scenario(
    scenario: str,
    target_offset_days: int,
    by_datetime: pd.DataFrame,
    available_dates: list[pd.Timestamp],
    predictions_path: Path,
    model_path: Path,
    keras,
    layers,
) -> dict[str, float | int | str]:
    samples = _build_sequence_samples(by_datetime, available_dates, target_offset_days)
    if not samples:
        raise ValueError(f"LSTM icin ornek uretilemedi: {scenario}")

    frame = pd.DataFrame(samples)
    frame["target_datetime"] = pd.to_datetime(frame["target_datetime"], errors="coerce")
    frame = frame.dropna(subset=["target_datetime"]).sort_values("target_datetime").reset_index(drop=True)

    sequences = np.stack(frame["sequence"].to_numpy())
    targets = frame["ptf_target"].astype(float).to_numpy()
    calendar = frame[["target_hour", "target_day_of_week", "target_is_weekend"]].astype(float).to_numpy()

    train_df, val_df, test_df = chronological_train_val_test_split(frame)
    train_idx = train_df.index.to_numpy()
    val_idx = val_df.index.to_numpy()
    test_idx = test_df.index.to_numpy()

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    calendar_scaler = StandardScaler()

    x_train = sequences[train_idx]
    x_train_2d = x_train.reshape(-1, x_train.shape[-1])
    feature_scaler.fit(x_train_2d)
    x_train = _scale_sequences(x_train, feature_scaler)

    y_train = target_scaler.fit_transform(targets[train_idx].reshape(-1, 1)).ravel()
    cal_train = calendar_scaler.fit_transform(calendar[train_idx])

    x_val = _scale_sequences(sequences[val_idx], feature_scaler)
    y_val = target_scaler.transform(targets[val_idx].reshape(-1, 1)).ravel()
    cal_val = calendar_scaler.transform(calendar[val_idx])

    x_test = _scale_sequences(sequences[test_idx], feature_scaler)
    y_test = targets[test_idx]
    cal_test = calendar_scaler.transform(calendar[test_idx])

    model = _build_model(keras, layers, n_features=x_train.shape[-1])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
    ]
    model.fit(
        [x_train, cal_train],
        y_train,
        validation_data=([x_val, cal_val], y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    scaled_predictions = model.predict([x_test, cal_test], verbose=0).ravel()
    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).ravel()
    predictions = np.clip(predictions, 0, None)
    metrics = _metrics(y_test, predictions)

    prediction_frame = pd.DataFrame(
        {
            "model": "lstm",
            "scenario": scenario,
            "forecast_issue_date": frame.loc[test_idx, "forecast_issue_date"].to_numpy(),
            "target_date": frame.loc[test_idx, "target_date"].to_numpy(),
            "target_hour": frame.loc[test_idx, "target_hour"].to_numpy(),
            "target_datetime": frame.loc[test_idx, "target_datetime"].to_numpy(),
            "actual_ptf": y_test,
            "predicted_ptf": predictions,
            "error": y_test - predictions,
            "absolute_error": np.abs(y_test - predictions),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_path, index=False)

    return {
        "scenario": scenario,
        "model": "lstm",
        "rows": int(len(frame)),
        "sequence_length": SEQ_LEN,
        "sequence_features": len(SEQUENCE_FEATURES),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "SMAPE": metrics["SMAPE"],
        "R2": metrics["R2"],
        "model_path": project_relative_path(model_path),
        "predictions_path": project_relative_path(predictions_path),
    }


def _build_model(keras, layers, n_features: int):
    sequence_input = layers.Input(shape=(SEQ_LEN, n_features), name="sequence_input")
    calendar_input = layers.Input(shape=(3,), name="calendar_input")

    x = layers.LSTM(64, return_sequences=True)(sequence_input)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Concatenate()([x, calendar_input])
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, name="ptf_output")(x)

    model = keras.Model(inputs=[sequence_input, calendar_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


def _build_sequence_samples(
    by_datetime: pd.DataFrame,
    available_dates: list[pd.Timestamp],
    target_offset_days: int,
) -> list[dict]:
    samples: list[dict] = []

    for issue_date in available_dates:
        target_date = issue_date + pd.Timedelta(days=target_offset_days)
        cutoff_datetime = pd.Timestamp.combine(issue_date.date(), time(CUTOFF_HOUR, 0))
        if cutoff_datetime not in by_datetime.index:
            continue

        sequence_times = [
            cutoff_datetime - pd.Timedelta(hours=SEQ_LEN - 1 - hour_offset)
            for hour_offset in range(SEQ_LEN)
        ]
        if not set(sequence_times).issubset(by_datetime.index):
            continue

        sequence_matrix = by_datetime.loc[sequence_times, SEQUENCE_FEATURES].astype(float).to_numpy()
        if np.isnan(sequence_matrix).mean() > 0.15:
            continue

        sequence_matrix = _fill_sequence_nan(sequence_matrix)

        for target_hour in range(24):
            target_datetime = pd.Timestamp.combine(target_date.date(), time(target_hour, 0))
            if target_datetime not in by_datetime.index:
                continue

            target_ptf = by_datetime.loc[target_datetime, "ptf"]
            if pd.isna(target_ptf):
                continue

            samples.append(
                {
                    "forecast_issue_date": issue_date.date().isoformat(),
                    "target_date": target_date.date().isoformat(),
                    "target_hour": target_hour,
                    "target_datetime": target_datetime,
                    "target_day_of_week": int(target_datetime.dayofweek),
                    "target_is_weekend": int(target_datetime.dayofweek in {5, 6}),
                    "ptf_target": float(target_ptf),
                    "sequence": sequence_matrix.astype(np.float32),
                }
            )

    return samples


def _fill_sequence_nan(sequence_matrix: np.ndarray) -> np.ndarray:
    filled = sequence_matrix.copy()
    for column_idx in range(filled.shape[1]):
        column = filled[:, column_idx]
        mask = ~np.isnan(column)
        if mask.any():
            column[~mask] = column[mask][-1] if mask.sum() else 0.0
        else:
            column[:] = 0.0
        filled[:, column_idx] = column
    return filled


def _scale_sequences(sequences: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = sequences.shape
    flat = sequences.reshape(-1, shape[-1])
    scaled = scaler.transform(flat)
    return scaled.reshape(shape)


def _read_hourly_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Final hourly dataset bulunamadi: {INPUT_PATH}")

    data = pd.read_csv(INPUT_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    data["date"] = data["datetime"].dt.normalize()

    for column in SEQUENCE_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }

