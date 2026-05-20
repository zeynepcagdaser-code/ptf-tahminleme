from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.hybrid_config import (
    HYBRID_CALIBRATION_CHART_PATH,
    HYBRID_CALIBRATED_METRICS_PATH,
    TARGET_COLUMN,
)
from src.model_splits import chronological_train_val_test_split


EPSILON = 1.0
SPIKE_LOW = 0.35
SPIKE_HIGH = 0.70
UPPER_Q95_MULT = 1.15
RAMP_DOWN_DROP_RATIO = 0.10
RAMP_DOWN_GRU_SHIFT = 0.12
RAMP_DOWN_CB_SHIFT = 0.12


def build_volatility_context(hourly: pd.DataFrame) -> pd.DataFrame:
    """Kesin PTF üzerinden issue anı volatilite rejimi (lookahead yok)."""
    df = hourly.copy()
    if "datetime" not in df.columns:
        df = df.reset_index()
        if "datetime" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
    df = df.sort_values("datetime")
    ptf = df["ptf_kesinlesmis"].fillna(df.get("ptf", np.nan)).astype(float)

    ctx = pd.DataFrame({"issue_datetime": df["datetime"]})
    ctx["rolling_24_std"] = ptf.rolling(24, min_periods=12).std().to_numpy()
    ctx["rolling_168_std"] = ptf.rolling(168, min_periods=48).std().to_numpy()
    ctx["rolling_168_median"] = ptf.rolling(168, min_periods=48).median().to_numpy()
    ctx["rolling_24_q10"] = ptf.rolling(24, min_periods=12).quantile(0.10).to_numpy()
    ctx["rolling_24_q95"] = ptf.rolling(24, min_periods=12).quantile(0.95).to_numpy()
    ctx["ramp_abs_1h"] = ptf.diff(1).abs().fillna(0.0).to_numpy()
    ctx["ramp_abs_3h"] = ptf.diff(3).abs().fillna(0.0).to_numpy()
    ctx["intraday_range"] = (ptf.rolling(24, min_periods=12).max() - ptf.rolling(24, min_periods=12).min()).to_numpy()

    ctx["lower_bound"] = ctx["rolling_24_q10"].fillna(ptf.quantile(0.10))
    ctx["upper_bound"] = (ctx["rolling_24_q95"] * UPPER_Q95_MULT).fillna(ptf.quantile(0.95) * UPPER_Q95_MULT)
    return ctx


def attach_volatility_features(merged: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    ctx = build_volatility_context(hourly)
    out = merged.merge(ctx, on="issue_datetime", how="left", suffixes=("", "_vol"))
    for col in ("lower_bound", "upper_bound", "rolling_24_std", "rolling_168_std"):
        if col in out.columns:
            out[col] = out[col].fillna(out[col].median())
    return out


def dynamic_amplitude_factor(spike_probability: np.ndarray) -> np.ndarray:
    """Düşük spike → baseline'a çek; yüksek spike → agresif genlik."""
    sp = np.clip(spike_probability.astype(float), 0.0, 1.0)
    amp = np.zeros_like(sp)
    low = sp < SPIKE_LOW
    mid = (sp >= SPIKE_LOW) & (sp <= SPIKE_HIGH)
    high = sp > SPIKE_HIGH

    amp[low] = 0.28 + 0.50 * (sp[low] / SPIKE_LOW)
    amp[mid] = 0.78 + 0.22 * ((sp[mid] - SPIKE_LOW) / (SPIKE_HIGH - SPIKE_LOW))
    amp[high] = 1.05 + 0.50 * ((sp[high] - SPIKE_HIGH) / (1.0 - SPIKE_HIGH))
    return amp


def apply_dynamic_amplitude(
    core: np.ndarray,
    seasonal: np.ndarray,
    spike_probability: np.ndarray,
) -> np.ndarray:
    deviation = core - seasonal
    amp = dynamic_amplitude_factor(spike_probability)
    return seasonal + deviation * amp


def clip_predictions(
    pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    spike_probability: np.ndarray,
) -> np.ndarray:
    lo = np.nan_to_num(lower, nan=0.0)
    hi = np.maximum(upper, lo + 50.0)
    out = pred.copy()
    sp = np.clip(spike_probability.astype(float), 0.0, 1.0)

    calm = sp < SPIKE_LOW
    out[calm] = np.clip(out[calm], lo[calm], hi[calm])

    mid = (sp >= SPIKE_LOW) & (sp <= SPIKE_HIGH)
    out[mid] = np.maximum(out[mid], lo[mid])
    out[mid] = np.minimum(out[mid], hi[mid] * 1.05)

    spike = sp > SPIKE_HIGH
    out[spike] = np.maximum(out[spike], lo[spike])
    out[spike] = np.minimum(out[spike], hi[spike] * 1.12)
    return out


def adjust_weights_ramp_down(
    weights: dict[str, float],
    *,
    prev_level: float,
    curr_level: float,
    high_threshold: float,
) -> dict[str, float]:
    if prev_level < high_threshold:
        return weights
    drop_ratio = (prev_level - curr_level) / max(prev_level, EPSILON)
    if drop_ratio < RAMP_DOWN_DROP_RATIO:
        return weights
    w = dict(weights)
    w["gru"] = min(0.65, w["gru"] + RAMP_DOWN_GRU_SHIFT)
    w["catboost"] = max(0.05, w["catboost"] - RAMP_DOWN_CB_SHIFT)
    total = w["catboost"] + w["seasonal"] + w["gru"]
    return {k: v / total for k, v in w.items()}


def peak_preserving_smooth(
    pred: np.ndarray,
    spike_probability: np.ndarray,
    *,
    rolling_24_std: float,
) -> np.ndarray:
    """EMA: sakin saatlerde daha fazla, peak/spike'ta daha az düzleştirme."""
    n = len(pred)
    if n <= 1:
        return pred.copy()

    out = np.zeros(n, dtype=float)
    out[0] = pred[0]
    vol_scale = min(1.0, rolling_24_std / max(float(np.nanmedian(np.abs(pred))), 80.0))

    for i in range(1, n):
        sp = float(spike_probability[i])
        is_peak = i < n - 1 and pred[i] >= pred[i - 1] and pred[i] >= pred[i + 1]
        if is_peak and sp > 0.45:
            out[i] = pred[i]
            continue

        if sp < SPIKE_LOW:
            alpha = 0.38 + 0.12 * (1.0 - vol_scale)
            out[i] = alpha * pred[i] + (1.0 - alpha) * out[i - 1]
        elif sp > SPIKE_HIGH:
            out[i] = pred[i]
        else:
            alpha = 0.68
            out[i] = alpha * pred[i] + (1.0 - alpha) * out[i - 1]

    return out


def build_raw_issue_path(group: pd.DataFrame, base_weights: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Ramp-down ağırlık düzeltmeli çekirdek blend (ham hibrit)."""
    g = group.sort_values("forecast_horizon").copy()
    high_threshold = float(g["rolling_168_median"].iloc[0] + 1.2 * g["rolling_168_std"].iloc[0])
    legacy_amp = 0.45

    cores: list[float] = []
    hybrids: list[float] = []
    prev_core: float | None = None
    for _, row in g.iterrows():
        h = int(row["forecast_horizon"])
        w = dict(base_weights[h])
        cand = w["catboost"] * row["pred_catboost"] + w["seasonal"] * row["pred_seasonal"] + w["gru"] * row["pred_gru"]
        if prev_core is not None:
            w = adjust_weights_ramp_down(
                w,
                prev_level=prev_core,
                curr_level=cand,
                high_threshold=high_threshold,
            )
            cand = w["catboost"] * row["pred_catboost"] + w["seasonal"] * row["pred_seasonal"] + w["gru"] * row["pred_gru"]
        dev = cand - row["pred_seasonal"]
        hybrid = row["pred_seasonal"] + dev * (1.0 + row["spike_probability"] * legacy_amp)
        cores.append(float(cand))
        hybrids.append(max(float(hybrid), 0.0))
        prev_core = float(cand)

    g["hybrid_core_ptf"] = cores
    g["hybrid_raw_predicted_ptf"] = hybrids
    return g


def calibrate_issue_path(group: pd.DataFrame) -> pd.DataFrame:
    """Ham hibrit üzerine volatilite kalibrasyonu."""
    g = group.sort_values("forecast_horizon").copy()
    lower = g["lower_bound"].to_numpy()
    upper = g["upper_bound"].to_numpy()
    seasonal = g["pred_seasonal"].to_numpy()
    spike = g["spike_probability"].to_numpy()
    core = g["hybrid_core_ptf"].to_numpy()
    raw = g["hybrid_raw_predicted_ptf"].to_numpy()

    calm = spike < SPIKE_LOW
    mid = (spike >= SPIKE_LOW) & (spike <= SPIKE_HIGH)
    hot = spike > SPIKE_HIGH
    gru = g["pred_gru"].to_numpy()

    final = raw.copy()

    # Sakin saatler: diplerde yüksek kalan tahmini baseline'a çek
    if calm.any():
        excess = final > seasonal
        final[calm & excess] = seasonal[calm & excess] + (final[calm & excess] - seasonal[calm & excess]) * 0.55
        final[calm] = np.maximum(final[calm], lower[calm])

    # Orta: keskin ramp-down → GRU
    for i in range(1, len(final)):
        if not mid[i]:
            continue
        drop = raw[i - 1] - raw[i]
        if drop > 0 and drop / max(raw[i - 1], 1.0) > 0.11:
            final[i] = 0.52 * raw[i] + 0.48 * gru[i]

    # Spike (>0.7): yalnızca agresif genlik + overshoot tavanı
    if hot.any():
        boosted = apply_dynamic_amplitude(core, seasonal, spike)
        final[hot] = boosted[hot]
        overshoot = final > upper * 1.04
        final[hot & overshoot] = upper[hot & overshoot] * 1.02 + (final[hot & overshoot] - upper[hot & overshoot]) * 0.35

    g["hybrid_predicted_ptf"] = np.maximum(final, 0.0)
    return g


def build_and_calibrate_hybrid(
    merged: pd.DataFrame,
    blend_weights: dict[int, dict[str, float]],
) -> pd.DataFrame:
    parts_raw = [build_raw_issue_path(grp, blend_weights) for _, grp in merged.groupby("issue_datetime", sort=False)]
    with_raw = pd.concat(parts_raw, ignore_index=True)
    parts_cal = [calibrate_issue_path(grp) for _, grp in with_raw.groupby("issue_datetime", sort=False)]
    return pd.concat(parts_cal, ignore_index=True)


def direction_accuracy(df: pd.DataFrame, pred_col: str) -> float:
    correct = 0
    total = 0
    for _, g in df.groupby("issue_datetime"):
        g = g.sort_values("forecast_horizon")
        actual = g[TARGET_COLUMN].to_numpy(dtype=float)
        pred = g[pred_col].to_numpy(dtype=float)
        if len(actual) < 2:
            continue
        da = np.diff(actual)
        dp = np.diff(pred)
        mask = (np.abs(da) > 5.0) | (np.abs(dp) > 5.0)
        if not mask.any():
            continue
        correct += int(np.sum((da[mask] > 0) == (dp[mask] > 0)))
        total += int(mask.sum())
    return float(correct / total) if total else 0.0


def evaluate_calibration(df: pd.DataFrame) -> dict:
    _, _, test_df = chronological_train_val_test_split(df.sort_values("issue_datetime"))
    eval_df = test_df if len(test_df) else df

    spike_mask = (
        eval_df["is_spike"].astype(int) == 1
        if "is_spike" in eval_df.columns
        else eval_df["spike_probability"] >= 0.5
    )

    def block(col: str) -> dict[str, float]:
        y = eval_df[TARGET_COLUMN].to_numpy()
        p = eval_df[col].to_numpy()
        m = _metrics(y, p)
        m["spike_hours_mae"] = float(
            mean_absolute_error(eval_df.loc[spike_mask, TARGET_COLUMN], eval_df.loc[spike_mask, col])
        ) if spike_mask.any() else 0.0
        m["non_spike_hours_mae"] = float(
            mean_absolute_error(eval_df.loc[~spike_mask, TARGET_COLUMN], eval_df.loc[~spike_mask, col])
        ) if (~spike_mask).any() else 0.0
        m["direction_accuracy"] = direction_accuracy(eval_df, col)
        m["trend_correlation"] = _trend_correlation(y, p)
        return m

    return {
        "raw": block("hybrid_raw_predicted_ptf"),
        "calibrated": block("hybrid_predicted_ptf"),
        "eval_rows": int(len(eval_df)),
    }


def generate_calibration_chart(df: pd.DataFrame, metrics: dict, output_path: Path | None = None) -> Path:
    output_path = output_path or HYBRID_CALIBRATION_CHART_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, test_df = chronological_train_volatility_split(df)
    if test_df.empty:
        test_df = df

    test_df = test_df.copy()
    test_df["abs_err_raw"] = np.abs(test_df[TARGET_COLUMN] - test_df["hybrid_raw_predicted_ptf"])
    agg = (
        test_df.groupby("issue_datetime")
        .agg(vol=("rolling_24_std", "mean"), spike_max=("spike_probability", "max"), mae_raw=("abs_err_raw", "mean"))
        .reset_index()
    )
    agg = agg.sort_values(["spike_max", "vol"], ascending=False)
    pick = agg.iloc[0]["issue_datetime"] if len(agg) else test_df["issue_datetime"].iloc[0]

    sample = test_df[test_df["issue_datetime"] == pick].sort_values("forecast_horizon")
    hours = sample["forecast_horizon"].to_numpy()
    actual = sample[TARGET_COLUMN].to_numpy()
    raw = sample["hybrid_raw_predicted_ptf"].to_numpy()
    cal = sample["hybrid_predicted_ptf"].to_numpy()
    seas = sample["pred_seasonal"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [2.2, 1]})

    axes[0].plot(hours, actual, "k-o", label="Gerçek PTF", linewidth=2, markersize=5)
    axes[0].plot(hours, raw, "r--s", label="Hibrit (ham)", linewidth=1.5, markersize=4)
    axes[0].plot(hours, cal, "b-^", label="Hibrit (kalibre)", linewidth=1.8, markersize=4)
    axes[0].plot(hours, seas, color="gray", linestyle=":", label="Mevsimsel baseline")
    axes[0].fill_between(hours, sample["lower_bound"].iloc[0], sample["upper_bound"].iloc[0], alpha=0.12, color="green", label="Clip band")
    axes[0].set_title(f"Volatility calibration — issue {pd.Timestamp(pick).strftime('%Y-%m-%d %H:%M')}")
    axes[0].set_xlabel("Horizon (saat)")
    axes[0].set_ylabel("PTF (TL/MWh)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    err_raw = np.abs(actual - raw)
    err_cal = np.abs(actual - cal)
    axes[1].bar(hours - 0.15, err_raw, width=0.3, label="Ham |hata|", color="salmon", alpha=0.8)
    axes[1].bar(hours + 0.15, err_cal, width=0.3, label="Kalibre |hata|", color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Horizon (saat)")
    axes[1].set_ylabel("Mutlak hata")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    cal_m = metrics.get("calibrated", {})
    raw_m = metrics.get("raw", {})
    caption = (
        f"Test — Ham MAE={raw_m.get('MAE', 0):.0f} | Kalibre MAE={cal_m.get('MAE', 0):.0f} | "
        f"Spike MAE {cal_m.get('spike_hours_mae', 0):.0f} | Non-spike MAE {cal_m.get('non_spike_hours_mae', 0):.0f} | "
        f"Yön doğruluğu {cal_m.get('direction_accuracy', 0):.1%}"
    )
    fig.text(0.5, 0.01, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return output_path


def chronological_train_volatility_split(df: pd.DataFrame):
    return chronological_train_val_test_split(df.sort_values("issue_datetime"))


def save_calibration_artifacts(df: pd.DataFrame, metrics: dict) -> Path:
    chart_path = generate_calibration_chart(df, metrics)
    payload = {**metrics, "calibration_chart": str(chart_path)}
    HYBRID_CALIBRATED_METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return chart_path


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum((np.abs(actual) + np.abs(pred)) / 2, EPSILON)
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "SMAPE": float(np.mean(np.abs(actual - pred) / denom) * 100),
        "R2": float(r2_score(actual, pred)),
    }


def _trend_correlation(y: np.ndarray, pred: np.ndarray) -> float:
    dy = np.diff(y.astype(float))
    dp = np.diff(pred.astype(float))
    if len(dy) < 3 or np.std(dy) < 1e-6 or np.std(dp) < 1e-6:
        return 0.0
    corr = float(np.corrcoef(dy, dp)[0, 1])
    return 0.0 if np.isnan(corr) else corr
