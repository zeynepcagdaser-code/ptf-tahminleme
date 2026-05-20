from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from src.hybrid_config import (
    HORIZON,
    HOURLY_PATH,
    HYBRID_GRU_HISTORY_PATH,
    HYBRID_GRU_METRICS_PATH,
    HYBRID_GRU_MODEL_PATH,
    HYBRID_GRU_PRED_PATH,
    HYBRID_GRU_SCALERS_PATH,
)
from src.model_splits import chronological_train_val_test_split, project_relative_path


EPSILON = 1.0


@dataclass(frozen=True)
class HybridGruSummary:
    input_window_hours: int
    mae: float
    rmse: float
    smape: float
    r2: float
    model_path: str
    predictions_path: str
    metrics_path: str


def train_lstm_hybrid(
    *,
    input_window_hours: int = 24,
    max_epochs: int = 10,
    batch_size: int = 256,
    hidden_size: int = 32,
    patience: int = 3,
) -> HybridGruSummary:
    """Hafif GRU — mevcut LSTM pipeline dosyalarina dokunmaz."""
    torch = _require_torch()
    hourly = _read_hourly()
    samples = _build_samples(hourly, input_window_hours=input_window_hours)
    samples = samples.sort_values("issue_datetime").reset_index(drop=True)
    train_df, val_df, test_df = chronological_train_val_test_split(samples)

    x_cols = [c for c in samples.columns if c.startswith("x_")]
    f_cols = [c for c in samples.columns if c.startswith("fut_")]
    y_cols = [f"y_t{h}" for h in range(1, HORIZON + 1)]
    n_features = len(x_cols) // input_window_hours

    x_scaler = RobustScaler().fit(train_df[x_cols])
    f_scaler = RobustScaler().fit(train_df[f_cols])
    y_scaler = RobustScaler().fit(train_df[y_cols])

    def scale_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            x_scaler.transform(df[x_cols]),
            f_scaler.transform(df[f_cols]),
            y_scaler.transform(df[y_cols]),
        )

    x_train, f_train, y_train = scale_split(train_df)
    x_val, f_val, y_val = scale_split(val_df)
    x_test, f_test, y_test = scale_split(test_df)

    device = torch.device("cpu")
    model = _make_gru_model(
        input_dim=n_features,
        future_dim=f_train.shape[1],
        hidden_size=hidden_size,
        horizon=HORIZON,
    ).to(device)

    x_train_t = torch.tensor(x_train.reshape(len(train_df), input_window_hours, n_features), dtype=torch.float32)
    f_train_t = torch.tensor(f_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    x_val_t = torch.tensor(x_val.reshape(len(val_df), input_window_hours, n_features), dtype=torch.float32)
    f_val_t = torch.tensor(f_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    x_test_t = torch.tensor(x_test.reshape(len(test_df), input_window_hours, n_features), dtype=torch.float32)
    f_test_t = torch.tensor(f_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train_t))
        losses = []
        for start in range(0, len(x_train_t), batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train_t[idx].to(device)
            fb = f_train_t[idx].to(device)
            yb = y_train_t[idx].to(device)
            optimizer.zero_grad()
            pred = model(xb, fb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t.to(device), f_val_t.to(device)), y_val_t.to(device)).item())

        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_test_t.to(device), f_test_t.to(device)).cpu().numpy()
    pred = np.maximum(y_scaler.inverse_transform(pred_scaled), 0.0)
    actual = np.maximum(y_scaler.inverse_transform(y_test_t.numpy()), 0.0)

    pred_long_test = _to_long_predictions(test_df, actual, pred)
    pred_long_all = _predict_all_samples(model, samples, x_scaler, f_scaler, y_scaler, x_cols, f_cols, input_window_hours, n_features, device, torch)
    pred_long_all.to_csv(HYBRID_GRU_PRED_PATH, index=False)

    overall = _metrics(actual.reshape(-1), pred.reshape(-1))
    HYBRID_GRU_METRICS_PATH.write_text(json.dumps({**overall, "input_window_hours": input_window_hours}, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(HYBRID_GRU_HISTORY_PATH, index=False)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_window_hours": input_window_hours,
            "input_dim": n_features,
            "future_dim": f_train.shape[1],
            "hidden_size": hidden_size,
            "x_cols": x_cols,
            "f_cols": f_cols,
        },
        HYBRID_GRU_MODEL_PATH,
    )
    HYBRID_GRU_SCALERS_PATH.write_text(
        json.dumps(
            {
                "x_center": x_scaler.center_.tolist(),
                "x_scale": x_scaler.scale_.tolist(),
                "f_center": f_scaler.center_.tolist(),
                "f_scale": f_scaler.scale_.tolist(),
                "y_center": y_scaler.center_.tolist(),
                "y_scale": y_scaler.scale_.tolist(),
            }
        ),
        encoding="utf-8",
    )

    return HybridGruSummary(
        input_window_hours=input_window_hours,
        mae=overall["MAE"],
        rmse=overall["RMSE"],
        smape=overall["SMAPE"],
        r2=overall["R2"],
        model_path=project_relative_path(HYBRID_GRU_MODEL_PATH),
        predictions_path=project_relative_path(HYBRID_GRU_PRED_PATH),
        metrics_path=project_relative_path(HYBRID_GRU_METRICS_PATH),
    )


def _make_gru_model(*, input_dim: int, future_dim: int, hidden_size: int, horizon: int):
    import torch.nn as nn

    class Seq2MultiGru(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size + future_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, horizon),
            )

        def forward(self, x, fut):
            import torch

            _, h = self.gru(x)
            h = h[-1]
            z = torch.cat([h, fut], dim=1)
            return self.head(z)

    return Seq2MultiGru()


def _build_samples(hourly: pd.DataFrame, *, input_window_hours: int) -> pd.DataFrame:
    df = hourly.sort_values("datetime").reset_index(drop=True)
    ptf = df["ptf_kesinlesmis"].fillna(df.get("ptf", np.nan)).astype(float)
    load = pd.to_numeric(df.get("load_forecast_plan", ptf), errors="coerce").ffill()
    wind = pd.to_numeric(df.get("wind_generation", 0), errors="coerce").fillna(0)
    solar = pd.to_numeric(df.get("solar_generation", 0), errors="coerce").fillna(0)
    hydro = pd.to_numeric(df.get("hydro_dam_generation", 0), errors="coerce").fillna(0)
    renewable = wind + solar + hydro
    net_load = load - renewable
    ramp1 = ptf.diff(1).fillna(0)
    ramp24 = ptf.diff(24).fillna(0)

    dt = pd.to_datetime(df["datetime"])
    hour_sin = np.sin(2 * np.pi * dt.dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * dt.dt.hour / 24)
    dow_sin = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    per_step = np.column_stack(
        [ptf, load, renewable, net_load, ramp1, ramp24, hour_sin, hour_cos, dow_sin, dow_cos]
    ).astype(np.float32)
    per_step = np.nan_to_num(per_step, nan=0.0)

    target = ptf.to_numpy(dtype=np.float32)
    rows: list[dict] = []
    n = len(df)

    for i in range(input_window_hours - 1, n - HORIZON):
        issue_dt = df.loc[i, "datetime"]
        window = per_step[i - input_window_hours + 1 : i + 1]
        future_y = target[i + 1 : i + HORIZON + 1]
        if np.isnan(future_y).any():
            continue

        fut_dt = pd.date_range(issue_dt + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
        fut_cal = np.column_stack(
            [
                np.sin(2 * np.pi * fut_dt.hour / 24),
                np.cos(2 * np.pi * fut_dt.hour / 24),
                np.sin(2 * np.pi * fut_dt.dayofweek / 7),
                np.cos(2 * np.pi * fut_dt.dayofweek / 7),
            ]
        ).astype(np.float32)

        row: dict = {"issue_datetime": issue_dt}
        flat = window.reshape(-1)
        for j, v in enumerate(flat):
            row[f"x_{j}"] = float(v)
        for j, v in enumerate(fut_cal.reshape(-1)):
            row[f"fut_{j}"] = float(v)
        for h, v in enumerate(future_y, start=1):
            row[f"y_t{h}"] = float(v)
        rows.append(row)

    return pd.DataFrame(rows)


def _predict_all_samples(
    model,
    samples: pd.DataFrame,
    x_scaler,
    f_scaler,
    y_scaler,
    x_cols: list[str],
    f_cols: list[str],
    input_window_hours: int,
    n_features: int,
    device,
    torch,
) -> pd.DataFrame:
    x = x_scaler.transform(samples[x_cols])
    f = f_scaler.transform(samples[f_cols])
    x_t = torch.tensor(x.reshape(len(samples), input_window_hours, n_features), dtype=torch.float32)
    f_t = torch.tensor(f, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_t.to(device), f_t.to(device)).cpu().numpy()
    pred = np.maximum(y_scaler.inverse_transform(pred_scaled), 0.0)

    records: list[dict] = []
    for idx, (_, row) in enumerate(samples.iterrows()):
        issue = row["issue_datetime"]
        for h in range(1, HORIZON + 1):
            records.append(
                {
                    "issue_datetime": issue,
                    "target_datetime": issue + pd.Timedelta(hours=h),
                    "forecast_horizon": h,
                    "pred_gru": float(pred[idx, h - 1]),
                }
            )
    return pd.DataFrame(records)


def _to_long_predictions(test_df: pd.DataFrame, actual: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    records: list[dict] = []
    for idx, (_, row) in enumerate(test_df.iterrows()):
        issue = row["issue_datetime"]
        for h in range(1, HORIZON + 1):
            records.append(
                {
                    "issue_datetime": issue,
                    "target_datetime": issue + pd.Timedelta(hours=h),
                    "forecast_horizon": h,
                    "actual_ptf": float(actual[idx, h - 1]),
                    "pred_gru": float(pred[idx, h - 1]),
                }
            )
    return pd.DataFrame(records)


def _read_hourly() -> pd.DataFrame:
    data = pd.read_csv(HOURLY_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data.dropna(subset=["datetime"])


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum((np.abs(actual) + np.abs(pred)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "SMAPE": float(np.mean(np.abs(actual - pred) / denom) * 100),
        "R2": float(r2_score(actual, pred)),
    }


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch gerekli: pip install torch") from exc
    return torch
