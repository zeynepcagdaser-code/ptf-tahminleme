from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dl_5y_config import (
    DL_COMPARISON_5Y_PATH,
    DL_METRICS_5Y_PATH,
    EPSILON_5Y,
    MODELS_5Y_DIR,
    OUTPUT_HORIZON_5Y,
    SCALERS_5Y_PATH,
    SEQUENCE_NPZ_PATH,
)


@dataclass(frozen=True)
class Dl5yRunSummary:
    model_type: str
    mae: float
    rmse: float
    smape: float
    r2: float
    epochs_ran: int
    train_seconds: float


def train_dl_baselines_5y(
    *,
    model_types: tuple[str, ...] = ("gru", "lstm", "cnn_lstm", "patch_tst"),
    max_epochs: int = 12,
    batch_size: int = 256,
    hidden_size: int = 64,
    patience: int = 4,
    quick: bool = True,
) -> dict:
    torch = _require_torch()
    data = np.load(SEQUENCE_NPZ_PATH, allow_pickle=True)
    import pickle

    with SCALERS_5Y_PATH.open("rb") as f:
        scalers = pickle.load(f)
    y_scaler = scalers["y_scaler"]

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.float32)
    X_val = torch.tensor(data["X_val"], dtype=torch.float32)
    y_val = torch.tensor(data["y_val"], dtype=torch.float32)
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.float32)

    if quick:
        cap = min(8000, len(X_train))
        X_train, y_train = X_train[:cap], y_train[:cap]
        X_val, y_val = X_val[:2000], y_val[:2000]

    input_dim = int(X_train.shape[-1])
    device = torch.device("cpu")
    results: list[dict] = []

    for model_type in model_types:
        if model_type == "patch_tst" and input_dim < 8:
            continue
        try:
            summary = _train_one(
                torch,
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                y_scaler=y_scaler,
                input_dim=input_dim,
                device=device,
                max_epochs=max_epochs,
                batch_size=batch_size,
                hidden_size=hidden_size,
                patience=patience,
            )
            results.append(asdict(summary))
            print(f"  [{model_type}] MAE={summary.mae:.1f} R2={summary.r2:.3f} ({summary.train_seconds:.0f}s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{model_type}] FAILED: {exc}")
            results.append({"model_type": model_type, "error": str(exc)})

    comparison = pd.DataFrame(results)
    comparison.to_csv(DL_COMPARISON_5Y_PATH, index=False)

    valid = comparison[~comparison.get("mae", pd.Series(dtype=float)).isna()] if "mae" in comparison.columns else comparison
    best = None
    if len(valid):
        best = valid.loc[valid["mae"].astype(float).idxmin(), "model_type"]

    payload = {"models": results, "best_model": best, "sequence_npz": str(SEQUENCE_NPZ_PATH)}
    DL_METRICS_5Y_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _train_one(
    torch,
    *,
    model_type: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    y_scaler,
    input_dim: int,
    device,
    max_epochs: int,
    batch_size: int,
    hidden_size: int,
    patience: int,
) -> Dl5yRunSummary:
    model = _build_model(torch, model_type, input_dim, hidden_size, OUTPUT_HORIZON_5Y).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()
    epochs_ran = 0

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()
        perm = torch.randperm(len(X_train))
        for start in range(0, len(X_train), batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val.to(device)), y_val.to(device)).item())
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
        pred_s = model(X_test.to(device)).cpu().numpy()
    actual_s = y_test.numpy()
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(pred_s.shape)
    actual = y_scaler.inverse_transform(actual_s.reshape(-1, 1)).reshape(actual_s.shape)
    pred = np.maximum(pred, 0.0)
    actual = np.maximum(actual, 0.0)

    m = _metrics(actual.reshape(-1), pred.reshape(-1))

    MODELS_5Y_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_type": model_type, "input_dim": input_dim}, MODELS_5Y_DIR / f"best_{model_type}_5y.pt")

    return Dl5yRunSummary(
        model_type=model_type,
        mae=m["MAE"],
        rmse=m["RMSE"],
        smape=m["SMAPE"],
        r2=m["R2"],
        epochs_ran=epochs_ran,
        train_seconds=time.time() - t0,
    )


def _build_model(torch, model_type: str, input_dim: int, hidden: int, horizon: int):
    nn = torch.nn
    Module = nn.Module

    if model_type == "gru":
        class M(Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.GRU(input_dim, hidden, batch_first=True, num_layers=2, dropout=0.15)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                _, h = self.rnn(x)
                return self.head(h[-1])

        return M()

    if model_type == "lstm":
        class M(Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2, dropout=0.15)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                _, (h, _) = self.rnn(x)
                return self.head(h[-1])

        return M()

    if model_type == "cnn_lstm":
        class M(Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
                self.act = nn.ReLU()
                self.rnn = nn.LSTM(hidden, hidden, batch_first=True, num_layers=1)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                z = self.act(self.conv(x.transpose(1, 2))).transpose(1, 2)
                _, (h, _) = self.rnn(z)
                return self.head(h[-1])

        return M()

    if model_type == "patch_tst":
        patch = 24

        class M(Module):
            def __init__(self):
                super().__init__()
                self.patch = patch
                self.proj = nn.Linear(input_dim * patch, hidden)
                self.encoder = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden, nhead=4, batch_first=True, dim_feedforward=hidden * 2),
                    num_layers=2,
                )
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                b, t, f = x.shape
                n_patch = t // self.patch
                x = x[:, : n_patch * self.patch, :].reshape(b, n_patch, self.patch * f)
                z = self.proj(x)
                z = self.encoder(z)
                return self.head(z[:, -1, :])

        return M()

    raise ValueError(f"Unknown model_type: {model_type}")


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum((np.abs(actual) + np.abs(pred)) / 2, EPSILON_5Y)
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
