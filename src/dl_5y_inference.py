from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.dl_5y_config import HOURLY_5Y_PATH, INPUT_WINDOW_5Y, OUTPUT_HORIZON_5Y, PRICE_COLUMN_5Y, SCALERS_5Y_PATH


MODELS_DIR = PROJECT_ROOT / "data" / "models" / "dl_5y"
PANEL_HOURLY_5Y_PATH = PROJECT_ROOT / "app_data" / "panel_hourly_5y.csv"


ModelType = Literal["lstm", "cnn_lstm"]


@dataclass(frozen=True)
class DlLiveForecast:
    model_type: str
    issue_datetime: str
    rows: int
    output: pd.DataFrame


def predict_next_12h_from_5y(*, model_type: ModelType = "cnn_lstm") -> DlLiveForecast:
    """
    Use the trained DL 5Y model and the last INPUT_WINDOW_5Y hours in HOURLY_5Y_PATH
    to forecast the next OUTPUT_HORIZON_5Y hours.

    This is a lightweight inference path intended for dashboard use (CPU).
    """
    torch = _require_torch()
    _require_sklearn()

    model_path = MODELS_DIR / f"best_{model_type}_5y.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"DL model bulunamadi: {model_path}")
    hourly_path = _choose_best_hourly_path()
    if not hourly_path.exists():
        raise FileNotFoundError(f"5y hourly dataset yok: {HOURLY_5Y_PATH} (panel fallback: {PANEL_HOURLY_5Y_PATH})")
    if not SCALERS_5Y_PATH.exists():
        raise FileNotFoundError(f"scalers_5y.pkl yok: {SCALERS_5Y_PATH}")

    import pickle

    with SCALERS_5Y_PATH.open("rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    feature_names = list(scalers.get("feature_names") or [])

    hourly = pd.read_csv(hourly_path)
    hourly["datetime"] = pd.to_datetime(hourly["datetime"], errors="coerce")
    hourly = hourly.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)

    if not feature_names:
        # Fallback: derive numeric cols similarly to build_dl_sequence_dataset_5y
        feature_names = _fallback_feature_names(hourly)

    # Training uses PRICE_COLUMN_5Y (=ptf_price). Panel artifact may only have ptf_kesinlesmis/ptf.
    if PRICE_COLUMN_5Y not in hourly.columns:
        if "ptf_kesinlesmis" in hourly.columns:
            hourly[PRICE_COLUMN_5Y] = pd.to_numeric(hourly["ptf_kesinlesmis"], errors="coerce")
        elif "ptf" in hourly.columns:
            hourly[PRICE_COLUMN_5Y] = pd.to_numeric(hourly["ptf"], errors="coerce")
        else:
            raise ValueError(f"Target price column missing: {PRICE_COLUMN_5Y}")

    hourly = _impute_like_training(hourly, feature_names)
    if len(hourly) < INPUT_WINDOW_5Y:
        raise ValueError(f"Hourly dataset too short: {len(hourly)} rows < window {INPUT_WINDOW_5Y}")

    window = hourly.tail(INPUT_WINDOW_5Y).copy()
    issue_dt = window["datetime"].iloc[-1]

    X = window[feature_names].to_numpy(dtype=np.float32)  # (w, f)
    Xs = x_scaler.transform(X)  # still (w, f)
    xb = torch.tensor(Xs[None, :, :], dtype=torch.float32)

    bundle = torch.load(model_path, map_location="cpu")
    state_dict = bundle.get("state_dict") if isinstance(bundle, dict) else bundle
    if not isinstance(state_dict, dict):
        raise TypeError("Unexpected checkpoint format: state_dict is not a dict")

    # Infer hidden size from checkpoint tensors (keeps inference compatible with training config).
    hidden = _infer_hidden_size(state_dict)

    model = _build_model(torch, model_type=model_type, input_dim=int(xb.shape[-1]), hidden=hidden, horizon=OUTPUT_HORIZON_5Y)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        pred_s = model(xb).cpu().numpy().reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).reshape(-1)
    pred = np.maximum(pred, 0.0)

    out_rows = []
    for h in range(1, OUTPUT_HORIZON_5Y + 1):
        out_rows.append(
            {
                "issue_datetime": str(issue_dt),
                "target_datetime": str(issue_dt + pd.Timedelta(hours=h)),
                "forecast_horizon": h,
                "predicted_ptf": float(pred[h - 1]),
                "model_name": f"dl_{model_type}_5y",
            }
        )
    out = pd.DataFrame(out_rows)
    out["issue_datetime"] = pd.to_datetime(out["issue_datetime"])
    out["target_datetime"] = pd.to_datetime(out["target_datetime"])
    return DlLiveForecast(model_type=model_type, issue_datetime=str(issue_dt), rows=len(out), output=out)


def _infer_hidden_size(state_dict: dict) -> int:
    # Prefer head weight (shape: [horizon, hidden])
    w = state_dict.get("head.weight")
    if hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    # CNN-LSTM conv (shape: [hidden, input_dim, k])
    cw = state_dict.get("conv.weight")
    if hasattr(cw, "shape") and len(cw.shape) >= 1:
        return int(cw.shape[0])
    raise ValueError("Cannot infer hidden size from checkpoint")


def _choose_best_hourly_path() -> Path:
    """
    Prefer whichever hourly dataset ends later (covers more recent datetimes).
    On Streamlit Cloud we usually only have PANEL_HOURLY_5Y_PATH.
    """
    candidates = [p for p in (HOURLY_5Y_PATH, PANEL_HOURLY_5Y_PATH) if p.exists()]
    if not candidates:
        return HOURLY_5Y_PATH

    best = candidates[0]
    best_end = None
    for p in candidates:
        try:
            df = pd.read_csv(p, usecols=["datetime"])
            dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
            end = dt.max() if len(dt) else None
        except Exception:
            end = None
        if best_end is None and end is not None:
            best, best_end = p, end
        elif end is not None and best_end is not None and end > best_end:
            best, best_end = p, end
    return best


def _impute_like_training(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.sort_values("datetime").reset_index(drop=True).copy()
    out[PRICE_COLUMN_5Y] = pd.to_numeric(out[PRICE_COLUMN_5Y], errors="coerce")
    out[PRICE_COLUMN_5Y] = out[PRICE_COLUMN_5Y].interpolate(method="linear", limit=6).ffill(limit=24).bfill(limit=24)

    for col in feature_cols:
        if col == PRICE_COLUMN_5Y:
            continue
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[col] = out[col].interpolate(method="linear", limit=12, limit_direction="both")
        out[col] = out[col].ffill(limit=48).bfill(limit=48)
        med = out[col].median()
        out[col] = out[col].fillna(med if pd.notna(med) else 0.0)
    return out


def _fallback_feature_names(df: pd.DataFrame) -> list[str]:
    exclude = {"datetime", "date", "hour", "ptf_kesinlesmis", "ptf_interim"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c == PRICE_COLUMN_5Y:
            cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    if PRICE_COLUMN_5Y not in cols:
        cols.append(PRICE_COLUMN_5Y)
    return cols


def _build_model(torch, *, model_type: ModelType, input_dim: int, hidden: int, horizon: int):
    nn = torch.nn

    if model_type == "lstm":
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(input_dim, hidden, batch_first=True, num_layers=2, dropout=0.15)
                self.head = nn.Linear(hidden, horizon)

            def forward(self, x):
                _, (h, _) = self.rnn(x)
                return self.head(h[-1])

        return M()

    if model_type == "cnn_lstm":
        class M(nn.Module):
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

    raise ValueError(f"Unknown model_type: {model_type}")


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("PyTorch gerekli (Streamlit Cloud icin requirements.txt'e torch ekleyin).") from exc
    import torch
    return torch


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scikit-learn gerekli (scalers_5y.pkl icin).") from exc
