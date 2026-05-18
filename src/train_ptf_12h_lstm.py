from __future__ import annotations

import json
import time
import copy
from dataclasses import asdict, dataclass
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from src.config import PROJECT_ROOT
from src.model_splits import chronological_train_val_test_split, project_relative_path


FINAL_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "data" / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = MODEL_DIR / "ptf_12h_lstm_model.pt"
SCALERS_PATH = MODEL_DIR / "ptf_12h_lstm_scalers.json"
HISTORY_PATH = PROCESSED_DIR / "ptf_12h_lstm_training_history.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "ptf_12h_lstm_predictions.csv"
METRICS_PATH = PROCESSED_DIR / "ptf_12h_lstm_metrics.json"
HORIZON_METRICS_PATH = PROCESSED_DIR / "ptf_12h_lstm_horizon_metrics.csv"
COMPARISON_PATH = PROCESSED_DIR / "ptf_12h_metrics_comparison.csv"
EXPERIMENTS_PATH = PROCESSED_DIR / "ptf_12h_lstm_experiments.csv"

CATBOOST_METRICS_PATH = PROCESSED_DIR / "ptf_12h_metrics.json"


HORIZON = 12
DEFAULT_INPUT_WINDOW_HOURS = 48

EPSILON = 1.0
TARGET_EPSILON = 1e-6


SUPPORTED_MODEL_TYPES = ("lstm", "gru", "cnn_lstm")
SUPPORTED_TARGET_TRANSFORMS = ("raw", "log1p")


@dataclass(frozen=True)
class Lstm12hTrainingSummary:
    model_type: str
    target_transform: str
    input_window_hours: int
    rows: int
    feature_dim: int
    calendar_future_dim: int
    train_rows: int
    validation_rows: int
    test_rows: int
    mae: float
    rmse: float
    smape: float
    r2: float
    model_path: str
    predictions_path: str
    metrics_path: str
    horizon_metrics_path: str
    history_path: str
    comparison_path: str
    run_tag: str = ""


def train_ptf_12h_lstm(
    *,
    input_window_hours: int = DEFAULT_INPUT_WINDOW_HOURS,
    max_epochs: int = 18,
    batch_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    patience: int = 4,
    model_type: str = "lstm",
    target_transform: str = "log1p",
    use_scheduler: bool = True,
    scheduler_patience: int = 2,
    min_lr: float = 1e-5,
    seed: int = 42,
    run_tag: str = "",
    model_path: Path = MODEL_PATH,
    scalers_path: Path = SCALERS_PATH,
    history_path: Path = HISTORY_PATH,
    predictions_path: Path = PREDICTIONS_PATH,
    metrics_path: Path = METRICS_PATH,
    horizon_metrics_path: Path = HORIZON_METRICS_PATH,
    comparison_path: Path = COMPARISON_PATH,
) -> Lstm12hTrainingSummary:
    torch = _require_torch()
    _set_seed(torch, seed)

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"model_type gecersiz: {model_type} (desteklenen: {SUPPORTED_MODEL_TYPES})")
    if target_transform not in SUPPORTED_TARGET_TRANSFORMS:
        raise ValueError(
            f"target_transform gecersiz: {target_transform} (desteklenen: {SUPPORTED_TARGET_TRANSFORMS})"
        )

    hourly = _read_final()
    samples = _build_seq2multi_samples(hourly, input_window_hours=input_window_hours, horizon=HORIZON)

    if len(samples) < 500:
        raise ValueError(f"LSTM egitimi icin az ornek: {len(samples)} (min 500 onerilir).")

    # Chronological split by issue_datetime.
    samples = samples.sort_values("issue_datetime").reset_index(drop=True)
    train_df, val_df, test_df = chronological_train_val_test_split(samples)

    feature_columns = [c for c in samples.columns if c.startswith("x_")]
    future_columns = [c for c in samples.columns if c.startswith("fut_")]
    target_columns = [f"y_t{h}" for h in range(1, HORIZON + 1)]

    x_scaler = RobustScaler()
    fut_scaler = RobustScaler()
    y_scaler = RobustScaler()

    x_train = train_df[feature_columns].to_numpy(dtype=np.float32)
    fut_train = train_df[future_columns].to_numpy(dtype=np.float32)
    y_train_raw = train_df[target_columns].to_numpy(dtype=np.float32)

    y_train = _transform_target_matrix(y_train_raw, mode=target_transform)

    x_scaler.fit(x_train)
    fut_scaler.fit(fut_train)
    y_scaler.fit(y_train)

    def to_tensors(frame: pd.DataFrame):
        x = x_scaler.transform(frame[feature_columns].to_numpy(dtype=np.float32))
        fut = fut_scaler.transform(frame[future_columns].to_numpy(dtype=np.float32))
        y_raw = frame[target_columns].to_numpy(dtype=np.float32)
        y = y_scaler.transform(_transform_target_matrix(y_raw, mode=target_transform))
        # Reshape: (N, T, D) where D is per-timestep features.
        x = x.reshape(len(frame), input_window_hours, -1)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(fut, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    x_train_t, fut_train_t, y_train_t = to_tensors(train_df)
    x_val_t, fut_val_t, y_val_t = to_tensors(val_df)
    x_test_t, fut_test_t, y_test_t = to_tensors(test_df)

    model = _build_model(
        torch=torch,
        model_type=model_type,
        input_dim=int(x_train_t.shape[-1]),
        future_dim=int(fut_train_t.shape[-1]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        horizon=HORIZON,
    )

    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

    history_rows: list[dict] = []
    best_val = float("inf")
    best_state = None
    no_improve = 0

    train_loader = _data_loader(torch, x_train_t, fut_train_t, y_train_t, batch_size=batch_size, shuffle=True)
    val_loader = _data_loader(torch, x_val_t, fut_val_t, y_val_t, batch_size=batch_size, shuffle=False)

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, fb, yb in train_loader:
            xb = xb.to(device)
            fb = fb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, fb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, fb, yb in val_loader:
                pred = model(xb.to(device), fb.to(device))
                loss = loss_fn(pred, yb.to(device))
                val_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_seconds": float(time.time() - start_time),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if scheduler is not None and not np.isnan(val_loss):
            scheduler.step(val_loss)

        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = _cpu_detached_state_dict(model.state_dict(), torch=torch)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test.
    model.eval()
    with torch.no_grad():
        scaled_pred = model(x_test_t.to(device), fut_test_t.to(device)).cpu().numpy()

    pred_t = y_scaler.inverse_transform(scaled_pred)
    actual_t = y_scaler.inverse_transform(y_test_t.cpu().numpy())

    pred = _inverse_transform_target_matrix(pred_t, mode=target_transform)
    actual = _inverse_transform_target_matrix(actual_t, mode=target_transform)

    pred = np.maximum(pred, 0.0)
    actual = np.maximum(actual, 0.0)

    # Flatten for overall metrics (N*H).
    overall = _metrics(actual.reshape(-1), pred.reshape(-1))
    horizon_metrics = _horizon_metrics(actual, pred)

    # Save artifacts.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_window_hours": int(input_window_hours),
            "input_dim": int(x_train_t.shape[-1]),
            "future_dim": int(fut_train_t.shape[-1]),
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "horizon": int(HORIZON),
            "model_type": model_type,
            "target_transform": target_transform,
            "feature_columns": feature_columns,
            "future_columns": future_columns,
            "target_columns": target_columns,
        },
        model_path,
    )

    scalers_payload = {
        "x_scaler": _serialize_robust_scaler(x_scaler),
        "future_scaler": _serialize_robust_scaler(fut_scaler),
        "y_scaler": _serialize_robust_scaler(y_scaler),
    }
    scalers_path.write_text(json.dumps(scalers_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    horizon_metrics.to_csv(horizon_metrics_path, index=False)

    pred_frame = _prediction_frame(test_df, actual, pred)
    pred_frame.to_csv(predictions_path, index=False)

    metrics_path.write_text(
        json.dumps(
            {
                **overall,
                "rows": int(len(samples)),
                "feature_dim": int(x_train_t.shape[-1]),
                "future_calendar_dim": int(fut_train_t.shape[-1]),
                "input_window_hours": int(input_window_hours),
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(val_df)),
                "test_rows": int(len(test_df)),
                "epochs_ran": int(len(history_rows)),
                "best_val_loss": float(best_val),
                "model_type": model_type,
                "target_transform": target_transform,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    comparison = _write_comparison(
        overall,
        horizon_metrics,
        input_window_hours=input_window_hours,
        model_type=model_type,
        target_transform=target_transform,
        out_path=comparison_path,
    )

    return Lstm12hTrainingSummary(
        model_type=model_type,
        target_transform=target_transform,
        input_window_hours=int(input_window_hours),
        rows=int(len(samples)),
        feature_dim=int(x_train_t.shape[-1]),
        calendar_future_dim=int(fut_train_t.shape[-1]),
        train_rows=int(len(train_df)),
        validation_rows=int(len(val_df)),
        test_rows=int(len(test_df)),
        mae=overall["MAE"],
        rmse=overall["RMSE"],
        smape=overall["SMAPE"],
        r2=overall["R2"],
        model_path=project_relative_path(model_path),
        predictions_path=project_relative_path(predictions_path),
        metrics_path=project_relative_path(metrics_path),
        horizon_metrics_path=project_relative_path(horizon_metrics_path),
        history_path=project_relative_path(history_path),
        comparison_path=project_relative_path(comparison),
        run_tag=run_tag,
    )


class Seq2MultiLstm:
    def __init__(
        self,
        *,
        torch,
        input_dim: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ) -> None:
        self.torch = torch
        nn = torch.nn

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def to(self, device):
        self.encoder.to(device)
        self.head.to(device)
        return self

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.head.parameters()

    def state_dict(self):
        return {"encoder": self.encoder.state_dict(), "head": self.head.state_dict()}

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.head.load_state_dict(state["head"])

    def train(self):
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.encoder.eval()
        self.head.eval()

    def __call__(self, x, future_vec):
        # x: (N, T, D)
        out, (h, _) = self.encoder(x)
        last_h = h[-1]  # (N, hidden)
        combined = self.torch.cat([last_h, future_vec], dim=1)
        return self.head(combined)


class Seq2MultiGru(Seq2MultiLstm):
    def __init__(
        self,
        *,
        torch,
        input_dim: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ) -> None:
        self.torch = torch
        nn = torch.nn
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def __call__(self, x, future_vec):
        out, h = self.encoder(x)
        last_h = h[-1]
        combined = self.torch.cat([last_h, future_vec], dim=1)
        return self.head(combined)


class Seq2MultiCnnLstm(Seq2MultiLstm):
    def __init__(
        self,
        *,
        torch,
        input_dim: int,
        future_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        horizon: int,
        conv_channels: int = 32,
        kernel_size: int = 5,
    ) -> None:
        self.torch = torch
        nn = torch.nn
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + future_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def to(self, device):
        self.conv.to(device)
        return super().to(device)

    def parameters(self):
        yield from self.conv.parameters()
        yield from super().parameters()

    def state_dict(self):
        base = super().state_dict()
        base["conv"] = self.conv.state_dict()
        return base

    def load_state_dict(self, state):
        if "conv" in state:
            self.conv.load_state_dict(state["conv"])
        super().load_state_dict(state)

    def train(self):
        self.conv.train()
        super().train()

    def eval(self):
        self.conv.eval()
        super().eval()

    def __call__(self, x, future_vec):
        # x: (N, T, D) -> conv expects (N, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, (h, _) = self.encoder(x)
        last_h = h[-1]
        combined = self.torch.cat([last_h, future_vec], dim=1)
        return self.head(combined)


def _read_final() -> pd.DataFrame:
    if not FINAL_PATH.exists():
        raise FileNotFoundError(f"Final dataset bulunamadi: {FINAL_PATH}")
    df = pd.read_csv(FINAL_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def _ptf_target_series(df: pd.DataFrame) -> pd.Series:
    # Align with CatBoost dataset convention: use kesinlesmis if available else ptf.
    kesin = pd.to_numeric(df.get("ptf_kesinlesmis"), errors="coerce")
    interim = pd.to_numeric(df.get("ptf"), errors="coerce")
    target = kesin.copy()
    target[target.isna()] = interim[target.isna()]
    return target


def _build_seq2multi_samples(
    hourly: pd.DataFrame,
    *,
    input_window_hours: int,
    horizon: int,
) -> pd.DataFrame:
    df = hourly.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Numeric feature columns (past-known only).
    base_cols = [
        "ptf",
        "smf",
        "load_forecast_plan",
        "wind_generation",
        "solar_generation",
        "hydro_dam_generation",
        "usd_try",
        "grf_tl",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Simple missing handling: forward/back fill then median, keeping time order.
    df[base_cols] = df[base_cols].ffill().bfill()
    for c in base_cols:
        med = float(df[c].median(skipna=True)) if df[c].notna().any() else 0.0
        df[c] = df[c].fillna(med)

    # Calendar features for each timestep.
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    hour = dt.dt.hour.to_numpy()
    dow = dt.dt.dayofweek.to_numpy()
    df["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    per_step_cols = base_cols + ["cal_hour_sin", "cal_hour_cos", "cal_dow_sin", "cal_dow_cos"]
    per_step = df[per_step_cols].to_numpy(dtype=np.float32)

    target = _ptf_target_series(df).to_numpy(dtype=np.float32)

    rows: list[dict] = []
    datetimes = df["datetime"].to_list()
    n = len(df)

    for i in range(input_window_hours - 1, n - horizon):
        issue_dt = datetimes[i]
        # Past window [i-window+1 .. i]
        window = per_step[i - input_window_hours + 1 : i + 1]

        # Targets are future ptf values at i+1..i+horizon
        future_targets = target[i + 1 : i + horizon + 1]
        if np.isnan(future_targets).any():
            continue

        # Future calendar vector for next horizon hours (known at issue time).
        fut_dt = pd.date_range(issue_dt + pd.Timedelta(hours=1), periods=horizon, freq="h")
        fut_hour = fut_dt.hour.to_numpy()
        fut_dow = fut_dt.dayofweek.to_numpy()
        fut = np.stack(
            [
                np.sin(2 * np.pi * fut_hour / 24.0),
                np.cos(2 * np.pi * fut_hour / 24.0),
                np.sin(2 * np.pi * fut_dow / 7.0),
                np.cos(2 * np.pi * fut_dow / 7.0),
            ],
            axis=1,
        ).astype(np.float32)

        row: dict[str, object] = {
            "issue_datetime": issue_dt,
            "target_datetime": issue_dt + pd.Timedelta(hours=horizon),
        }

        # Flatten window into columns x_0.. (kept numeric for scalers).
        flat = window.reshape(-1)
        for j, val in enumerate(flat):
            row[f"x_{j}"] = float(val)

        fut_flat = fut.reshape(-1)
        for j, val in enumerate(fut_flat):
            row[f"fut_{j}"] = float(val)

        for h in range(1, horizon + 1):
            row[f"y_t{h}"] = float(future_targets[h - 1])

        rows.append(row)

    return pd.DataFrame(rows)


def _transform_target_matrix(y: np.ndarray, *, mode: str) -> np.ndarray:
    y = y.astype(np.float32)
    y = np.maximum(y, 0.0)
    if mode == "raw":
        return y
    if mode == "log1p":
        return np.log1p(y)
    raise ValueError(f"unknown target transform: {mode}")


def _inverse_transform_target_matrix(y: np.ndarray, *, mode: str) -> np.ndarray:
    y = y.astype(np.float32)
    if mode == "raw":
        return y
    if mode == "log1p":
        y = np.maximum(y, 0.0)
        return np.expm1(y)
    raise ValueError(f"unknown target transform: {mode}")


def _build_model(
    *,
    torch,
    model_type: str,
    input_dim: int,
    future_dim: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    horizon: int,
):
    if model_type == "lstm":
        return Seq2MultiLstm(
            torch=torch,
            input_dim=input_dim,
            future_dim=future_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    if model_type == "gru":
        return Seq2MultiGru(
            torch=torch,
            input_dim=input_dim,
            future_dim=future_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    if model_type == "cnn_lstm":
        return Seq2MultiCnnLstm(
            torch=torch,
            input_dim=input_dim,
            future_dim=future_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
        )
    raise ValueError(f"unknown model_type: {model_type}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    smape_denominator = np.maximum((np.abs(actual) + np.abs(predicted)) / 2.0, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "SMAPE": float(np.mean(np.abs(actual - predicted) / smape_denominator) * 100),
        "R2": float(r2_score(actual, predicted)),
    }


def _horizon_metrics(actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for h in range(1, HORIZON + 1):
        metrics = _metrics(actual[:, h - 1], predicted[:, h - 1])
        rows.append({"forecast_horizon": int(h), **metrics})
    return pd.DataFrame(rows)


def _prediction_frame(test_df: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    # Expand to long format: one row per issue_datetime * horizon.
    out_rows: list[dict] = []
    issue_times = pd.to_datetime(test_df["issue_datetime"], errors="coerce")
    for i in range(len(test_df)):
        issue_dt = issue_times.iloc[i]
        for h in range(1, HORIZON + 1):
            target_dt = issue_dt + pd.Timedelta(hours=h)
            a = float(actual[i, h - 1])
            p = float(predicted[i, h - 1])
            out_rows.append(
                {
                    "issue_datetime": issue_dt,
                    "target_datetime": target_dt,
                    "forecast_horizon": int(h),
                    "actual_ptf": a,
                    "predicted_ptf": p,
                    "absolute_error": float(abs(a - p)),
                }
            )
    return pd.DataFrame(out_rows)


def _write_comparison(
    overall: dict[str, float],
    horizon_metrics: pd.DataFrame,
    *,
    input_window_hours: int,
    model_type: str,
    target_transform: str,
    out_path: Path,
) -> Path:
    # Combine overall and per-horizon metrics for quick comparison with CatBoost baseline.
    rows: list[dict] = []
    model_name = f"{model_type}_window_{input_window_hours}h_{target_transform}"
    rows.append({"model": model_name, "scope": "overall", **overall})
    for _, r in horizon_metrics.iterrows():
        rows.append(
            {
                "model": model_name,
                "scope": f"h{int(r['forecast_horizon'])}",
                "MAE": float(r["MAE"]),
                "RMSE": float(r["RMSE"]),
                "SMAPE": float(r["SMAPE"]),
                "R2": float(r["R2"]),
            }
        )

    if CATBOOST_METRICS_PATH.exists():
        try:
            cat = json.loads(CATBOOST_METRICS_PATH.read_text(encoding="utf-8"))
            rows.append(
                {
                    "model": "catboost_baseline",
                    "scope": "overall",
                    "MAE": float(cat.get("MAE")),
                    "RMSE": float(cat.get("RMSE")),
                    "SMAPE": float(cat.get("SMAPE")),
                    "R2": float(cat.get("R2")),
                }
            )
        except Exception:
            pass

    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _serialize_robust_scaler(scaler: RobustScaler) -> dict:
    # Keep as JSON for portability.
    center = scaler.center_.tolist() if getattr(scaler, "center_", None) is not None else None
    scale = scaler.scale_.tolist() if getattr(scaler, "scale_", None) is not None else None
    return {"center": center, "scale": scale}


def _cpu_detached_state_dict(state, *, torch):
    # Our model state_dict is a nested mapping: {"encoder": OrderedDict(...), "head": OrderedDict(...)}.
    # Only tensors support detach()/cpu(); keep other values as deep-copied objects.
    if isinstance(state, torch.Tensor):
        return state.detach().cpu()
    if isinstance(state, dict):
        return {k: _cpu_detached_state_dict(v, torch=torch) for k, v in state.items()}
    try:
        # OrderedDict and similar mappings
        items = state.items()  # type: ignore[attr-defined]
    except Exception:
        return copy.deepcopy(state)
    return state.__class__({k: _cpu_detached_state_dict(v, torch=torch) for k, v in items})


def _require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyTorch bulunamadi. LSTM egitimi icin once PyTorch kur:\n"
            "  pip install torch torchvision torchaudio\n"
            f"Hata: {type(exc).__name__}: {exc}"
        ) from exc
    return torch


def _set_seed(torch, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _data_loader(torch, x, fut, y, *, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(x, fut, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def run_fast_experiments() -> pd.DataFrame:
    # 3-5 runs total for CPU (MacBook Air) friendliness.
    experiments = [
        {"model_type": "lstm", "target_transform": "log1p", "input_window_hours": 48, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 1e-3, "max_epochs": 14},
        {"model_type": "gru", "target_transform": "log1p", "input_window_hours": 48, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 1e-3, "max_epochs": 14},
        {"model_type": "cnn_lstm", "target_transform": "log1p", "input_window_hours": 48, "hidden_size": 64, "num_layers": 1, "dropout": 0.15, "learning_rate": 1e-3, "max_epochs": 14},
        {"model_type": "lstm", "target_transform": "log1p", "input_window_hours": 24, "hidden_size": 48, "num_layers": 1, "dropout": 0.1, "learning_rate": 1e-3, "max_epochs": 14},
        {"model_type": "lstm", "target_transform": "log1p", "input_window_hours": 168, "hidden_size": 64, "num_layers": 2, "dropout": 0.2, "learning_rate": 5e-4, "max_epochs": 10},
    ]

    rows = []
    best_summary = None
    best_rmse = float("inf")

    runs_dir = PROCESSED_DIR / "ptf_12h_lstm_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for idx, cfg in enumerate(experiments, start=1):
        run_tag = f"run{idx}_{cfg['model_type']}_w{cfg['input_window_hours']}_{cfg['target_transform']}"
        summary = train_ptf_12h_lstm(
            input_window_hours=int(cfg["input_window_hours"]),
            max_epochs=int(cfg["max_epochs"]),
            batch_size=128,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            learning_rate=float(cfg["learning_rate"]),
            patience=4,
            model_type=str(cfg["model_type"]),
            target_transform=str(cfg["target_transform"]),
            use_scheduler=True,
            run_tag=run_tag,
            model_path=MODEL_DIR / f"ptf_12h_lstm_model_{run_tag}.pt",
            scalers_path=MODEL_DIR / f"ptf_12h_lstm_scalers_{run_tag}.json",
            history_path=runs_dir / f"history_{run_tag}.csv",
            predictions_path=runs_dir / f"predictions_{run_tag}.csv",
            metrics_path=runs_dir / f"metrics_{run_tag}.json",
            horizon_metrics_path=runs_dir / f"horizon_metrics_{run_tag}.csv",
            comparison_path=runs_dir / f"comparison_{run_tag}.csv",
        )
        row = asdict(summary)
        row.update(cfg)
        rows.append(row)
        if summary.rmse < best_rmse:
            best_rmse = summary.rmse
            best_summary = summary

    out = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    out.to_csv(EXPERIMENTS_PATH, index=False)

    # Promote best run artifacts to canonical paths for dashboard consumption.
    if best_summary is not None:
        _promote_best_run(best_summary, torch=_require_torch())

    return out


def _promote_best_run(best: Lstm12hTrainingSummary, *, torch) -> None:
    # Copy best run artifacts into canonical locations used by the dashboard.
    import shutil

    shutil.copyfile(PROJECT_ROOT / best.model_path, MODEL_PATH)
    shutil.copyfile(PROJECT_ROOT / best.metrics_path, METRICS_PATH)
    shutil.copyfile(PROJECT_ROOT / best.predictions_path, PREDICTIONS_PATH)
    shutil.copyfile(PROJECT_ROOT / best.horizon_metrics_path, HORIZON_METRICS_PATH)
    shutil.copyfile(PROJECT_ROOT / best.history_path, HISTORY_PATH)
    shutil.copyfile(PROJECT_ROOT / best.comparison_path, COMPARISON_PATH)
    # scalers live in models dir
    scalers_src = MODEL_DIR / f"ptf_12h_lstm_scalers_{best.run_tag}.json"
    if scalers_src.exists():
        shutil.copyfile(scalers_src, SCALERS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description="PTF 12 saatlik LSTM/GRU/CNN-LSTM egitimi")
    parser.add_argument("--search", action="store_true", help="Hizli 3-5 kombinasyon dene ve karsilastir")
    parser.add_argument("--window", type=int, default=DEFAULT_INPUT_WINDOW_HOURS, help="Girdi pencere saati (24/48/168)")
    parser.add_argument("--model", type=str, default="lstm", help="Model tipi: lstm/gru/cnn_lstm")
    parser.add_argument("--target", type=str, default="log1p", help="Target donusumu: raw/log1p")
    args = parser.parse_args()

    if args.search:
        df = run_fast_experiments()
        print(df.head(10).to_string(index=False))
        return

    summary = train_ptf_12h_lstm(input_window_hours=args.window, model_type=args.model, target_transform=args.target)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
