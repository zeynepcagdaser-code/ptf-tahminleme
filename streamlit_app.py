from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from src.config import PROJECT_ROOT
from src.dl_5y_config import HOURLY_5Y_PATH, START_DATE_5Y
from src.panel_data import (
    DASHBOARD_DATA_PATH,
    get_active_panel_sources,
    load_active_metrics,
    load_leaderboard,
)
from src.epias_5y_panel import (
    FEATURE_LABELS,
    FETCH_LIVE_APP_PATH,
    FETCH_PROGRESS_PATH,
    RAW_5Y_DIR,
    build_epias_5y_inventory,
    get_live_fetch_snapshot,
    inventory_summary_stats,
    is_fetch_process_running,
    load_epias_5y_timeseries,
    sync_fetch_live_to_app_data,
)


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
APP_DATA_PATH = DASHBOARD_DATA_PATH
LEGACY_METRICS_PATH = PROCESSED / "ptf_12h_final_metrics.json"
LEGACY_PREDICTIONS_PATH = PROCESSED / "ptf_12h_final_predictions.csv"
LEGACY_HORIZON_METRICS_PATH = PROCESSED / "ptf_12h_final_horizon_metrics.csv"
BASELINE_HORIZON_METRICS_PATH = PROCESSED / "ptf_12h_horizon_metrics.csv"
LIVE_FORECAST_PATH = PROCESSED / "ptf_12h_live_forecast.csv"
LIVE_BUNDLE_PATH = PROCESSED / "ptf_12h_live_bundle.csv"
DL_COMPARISON_5Y_PATH = PROCESSED / "dl_models_comparison_5y.csv"
DL_METRICS_5Y_PATH = PROCESSED / "dl_models_metrics_5y.json"

PANEL_SOURCES = get_active_panel_sources()


st.set_page_config(
    page_title="PTF 12 Saat Tahmin - 5Y Best",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0c1e33 0%, #14558f 100%);
        color: #f5f9ff;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p { margin: 0.6rem 0 0; opacity: 0.95; font-size: 1.05rem; }
    .metric-box {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0ea5e9;
    }
    .improvement-positive { color: #10b981; font-weight: 600; }
    .improvement-negative { color: #ef4444; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_secrets() -> None:
    load_dotenv(ROOT / ".env")
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    for key in ("EPIAS_USERNAME", "EPIAS_PASSWORD", "EPIAS_TGT", "EPIAS_BASE_URL", "EPIAS_AUTH_URL"):
        value = secrets.get(key)
        if value is not None and str(value).strip():
            os.environ[key] = str(value)


def has_credentials() -> bool:
    apply_secrets()
    return bool(os.getenv("EPIAS_TGT") or (os.getenv("EPIAS_USERNAME") and os.getenv("EPIAS_PASSWORD")))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_final_metrics() -> dict:
    return load_active_metrics()


def load_final_predictions() -> pd.DataFrame:
    if PANEL_SOURCES.predictions_path.exists():
        return read_csv(PANEL_SOURCES.predictions_path)
    return read_csv(ROOT / "app_data" / "ptf_12h_best_predictions_sample.csv")


def load_final_horizon_metrics() -> pd.DataFrame:
    path = PANEL_SOURCES.horizon_metrics_path
    if path.suffix == ".csv":
        return read_csv(path)
    return pd.DataFrame()


def load_legacy_metrics() -> dict:
    if LEGACY_METRICS_PATH.exists():
        return json.loads(LEGACY_METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def load_baseline_horizon_metrics() -> pd.DataFrame:
    return read_csv(BASELINE_HORIZON_METRICS_PATH)


def fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def fmt_pct(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+.{digits}f}%"


def _coerce_datetime_col(df: pd.DataFrame, col: str = "datetime") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


@st.cache_data(ttl=600, show_spinner=False)
def load_final_dataset() -> pd.DataFrame:
    path = PANEL_SOURCES.hourly_dataset_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = _coerce_datetime_col(df, "datetime")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def build_naive_live_forecast(final_df: pd.DataFrame, horizons: int = 12) -> pd.DataFrame:
    """
    Snapshot yoksa canlı tahmini panel içinde hızlıca üretmek için basit bir baseline.
    Yalnızca mevcut PTF geçmişini kullanır.
    """
    if final_df.empty or "datetime" not in final_df.columns or "ptf" not in final_df.columns:
        return pd.DataFrame()
    df = final_df.dropna(subset=["datetime", "ptf"]).sort_values("datetime").copy()
    if df.empty:
        return pd.DataFrame()

    cutoff_dt = df["datetime"].max()
    last_24 = df.tail(24)["ptf"].mean() if len(df) >= 24 else df["ptf"].mean()

    preds: list[dict] = []
    for h in range(1, horizons + 1):
        target_dt = cutoff_dt + pd.Timedelta(hours=h)

        y_dt = target_dt - pd.Timedelta(hours=24)
        w_dt = target_dt - pd.Timedelta(hours=168)
        y_val = df.loc[df["datetime"] == y_dt, "ptf"]
        w_val = df.loc[df["datetime"] == w_dt, "ptf"]
        y_ptf = float(y_val.iloc[0]) if len(y_val) else None
        w_ptf = float(w_val.iloc[0]) if len(w_val) else None

        parts = [float(last_24)]
        weights = [0.45]
        if y_ptf is not None:
            parts.append(y_ptf)
            weights.append(0.35)
        if w_ptf is not None:
            parts.append(w_ptf)
            weights.append(0.20)

        wsum = sum(weights)
        pred = sum(p * w for p, w in zip(parts, weights)) / wsum if wsum else float(last_24)

        preds.append(
            {
                "forecast_horizon": h,
                "target_datetime": target_dt,
                "predicted_ptf": pred,
                "model_name": "baseline_panel",
            }
        )

    return pd.DataFrame(preds)


def load_live_forecast_csv() -> pd.DataFrame:
    df = read_csv(LIVE_FORECAST_PATH)
    if df.empty:
        return df
    for c in ("issue_datetime", "target_datetime"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_dl_comparison_5y() -> pd.DataFrame:
    return read_csv(DL_COMPARISON_5Y_PATH)


def load_dl_metrics_5y() -> dict:
    if DL_METRICS_5Y_PATH.exists():
        try:
            return json.loads(DL_METRICS_5Y_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_data(ttl=600, show_spinner=False)
def load_live_bundle() -> pd.DataFrame:
    live_path = PANEL_SOURCES.live_forecast_path
    df = read_csv(live_path if live_path.exists() else LIVE_BUNDLE_PATH)
    if df.empty:
        return df
    for c in ("issue_datetime", "target_datetime"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "forecast_horizon" in df.columns:
        df["forecast_horizon"] = pd.to_numeric(df["forecast_horizon"], errors="coerce")
    return df.sort_values("forecast_horizon")


@st.cache_data(ttl=900, show_spinner=False)
def load_final_predictions_prepared() -> pd.DataFrame:
    """
    Final ensemble tahmin dosyasini dashboard icin standardize eder.
    - actual_ptf: gercek hedef PTF (csv'de ptf_target)
    - predicted_ptf: final ensemble tahmini (csv'de final_predicted_ptf)
    """
    df = load_final_predictions()
    if df.empty:
        return df

    out = df.copy()
    for c in ("issue_datetime", "target_datetime"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")

    if "forecast_horizon" in out.columns:
        out["forecast_horizon"] = pd.to_numeric(out["forecast_horizon"], errors="coerce")

    if "ptf_target" in out.columns and "actual_ptf" not in out.columns:
        out = out.rename(columns={"ptf_target": "actual_ptf"})
    if "panel_predicted_ptf" in out.columns:
        out["predicted_ptf"] = out["panel_predicted_ptf"]
    elif "final_predicted_ptf" in out.columns and "predicted_ptf" not in out.columns:
        out = out.rename(columns={"final_predicted_ptf": "predicted_ptf"})
    elif "predicted_ptf" in out.columns and "actual_ptf" not in out.columns and "ptf_target" in out.columns:
        out["actual_ptf"] = out["ptf_target"]

    if "pred_catboost_export" in out.columns and "pred_catboost" not in out.columns:
        out["pred_catboost"] = out["pred_catboost_export"]
    if "pred_seasonal_blend_export" in out.columns and "pred_seasonal_blend" not in out.columns:
        out["pred_seasonal_blend"] = out["pred_seasonal_blend_export"]

    needed = {"target_datetime", "forecast_horizon", "actual_ptf", "predicted_ptf"}
    if not needed.issubset(set(out.columns)):
        return df

    out = out.dropna(subset=["target_datetime", "forecast_horizon", "actual_ptf", "predicted_ptf"]).copy()
    out["absolute_error"] = (out["predicted_ptf"] - out["actual_ptf"]).abs()
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def run_live_update() -> dict:
    from src.run_ptf_pipeline import run_full_ptf_pipeline

    run_full_ptf_pipeline(fetch_live=True)
    return load_final_metrics()


# --- Sidebar ---
with st.sidebar:
    st.header("⚡ PTF Tahmin Sistemi")
    st.markdown("---")
    
    st.markdown("### 📊 Menü")
    view_mode = st.radio(
        "Seçiniz",
        ["Genel Bakış", "Canlı Tahmin", "Veriler", "Model Karşılaştırma", "Debug"],
        label_visibility="collapsed",
    )
    
    with st.expander("Bilgi", expanded=False):
        st.markdown(
            "Bu panel, 12 saat ileri PTF tahmini için üretilen **final ensemble** çıktıları ve 5y DL denemelerini gösterir."
        )
    
    st.markdown("---")
    st.success(f"**Aktif model:** {PANEL_SOURCES.label}")
    if PANEL_SOURCES.is_best_5y:
        st.caption("Kaynak: 5 yıllık EPİAŞ paneli + CatBoost")
    else:
        st.caption("Kaynak: legacy final ensemble")

    st.markdown("### 🤖 Tahmin Modeli")
    forecast_model = st.selectbox(
        "Seçiniz",
        ["Final Ensemble", "CNN-LSTM (5Y)", "LSTM (5Y)"],
        index=0,
        label_visibility="collapsed",
        key="forecast_model_choice",
    )

    refresh = st.button("🔄 Veriyi Güncelle", type="primary", width="stretch")
    st.caption("EPİAŞ çekimi + pipeline (kimlik bilgisi gerekli)")


if refresh:
    run_live_update.clear()

if refresh and has_credentials():
    with st.spinner("EPİAŞ verisi çekiliyor, ensemble oluşturuluyor..."):
        try:
            data = run_live_update()
            st.success("✅ Güncelleme tamamlandı!")
        except Exception as exc:
            st.error(f"❌ Hata: {exc}")
            data = load_final_metrics()
elif refresh:
    st.warning("⚠️ EPİAŞ secrets gerekli.")
    data = load_final_metrics()
else:
    data = load_final_metrics()

# Load data
final_metrics = data
leaderboard = load_leaderboard()
legacy_metrics = load_legacy_metrics()
final_preds = load_final_predictions_prepared()
final_horizon = load_final_horizon_metrics()
baseline_horizon = load_baseline_horizon_metrics()
final_dataset = load_final_dataset()

# Load dashboard data for live forecast
if APP_DATA_PATH.exists():
    dashboard_data = json.loads(APP_DATA_PATH.read_text(encoding="utf-8"))
    live_df = pd.DataFrame(dashboard_data.get("live_forecast", []))
else:
    dashboard_data = {}
    live_df = pd.DataFrame()


# --- Hero Section ---
_active_mae = final_metrics.get("MAE", 0) if final_metrics else 0
_hero_sub = (
    f"5 yıllık veri (2020–2024) ile eğitilmiş CatBoost — test MAE ≈ {_active_mae:.0f} TL/MWh."
    if PANEL_SOURCES.is_best_5y
    else "Kesinleşmemiş PTF (I-MCP) ile 12 saat MCP tahmini (legacy ensemble)."
)
st.markdown(
    f"""
    <div class="hero">
      <h1>⚡ PTF 12 Saat Tahmin — {PANEL_SOURCES.label}</h1>
      <p>{_hero_sub}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Menü sadeleştirme: Mevcut alt sayfaları aynı kodu kullanarak yeniden yönlendiriyoruz.
if view_mode == "Model Karşılaştırma":
    view_mode = "DL Modeller (5Y)"
elif view_mode == "Debug":
    st.markdown("## 🛠️ Debug")
    debug_view = st.radio(
        "Sayfa",
        ["5Y EPİAŞ Verileri", "Ensemble Detayları", "Performans Analizi"],
        horizontal=True,
        label_visibility="collapsed",
    )
    view_mode = debug_view


# --- Overview Section ---
if view_mode == "Genel Bakış":
    st.markdown("## 📈 Genel Performans Özeti")
    
    if final_metrics:
        # Metrics comparison
        baseline_mae = final_metrics.get("baseline_comparison", {}).get("mae", 0)
        baseline_rmse = final_metrics.get("baseline_comparison", {}).get("rmse", 0)
        baseline_smape = final_metrics.get("baseline_comparison", {}).get("smape", 0)
        baseline_r2 = final_metrics.get("baseline_comparison", {}).get("r2", 0)
        
        final_mae = final_metrics.get("MAE", 0)
        final_rmse = final_metrics.get("RMSE", 0)
        final_smape = final_metrics.get("SMAPE", 0)
        final_r2 = final_metrics.get("R2", 0)
        
        mae_imp = (baseline_mae - final_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        rmse_imp = (baseline_rmse - final_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
        smape_imp = (baseline_smape - final_smape) / baseline_smape * 100 if baseline_smape > 0 else 0
        r2_imp = final_r2 - baseline_r2
        
        # Comparison cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "MAE",
                f"{final_mae:.2f} TL",
                f"{mae_imp:+.1f}%",
                delta_color="normal" if mae_imp > 0 else "inverse"
            )
            st.caption(f"Baseline: {baseline_mae:.2f} TL")
        
        with col2:
            st.metric(
                "RMSE",
                f"{final_rmse:.2f} TL",
                f"{rmse_imp:+.1f}%",
                delta_color="normal" if rmse_imp > 0 else "inverse"
            )
            st.caption(f"Baseline: {baseline_rmse:.2f} TL")
        
        with col3:
            st.metric(
                "SMAPE",
                f"{final_smape:.2f}%",
                f"{smape_imp:+.1f}%",
                delta_color="normal" if smape_imp > 0 else "inverse"
            )
            st.caption(f"Baseline: {baseline_smape:.2f}%")
        
        with col4:
            st.metric(
                "R²",
                f"{final_r2:.4f}",
                f"{r2_imp:+.4f}",
                delta_color="normal" if r2_imp > 0 else "inverse"
            )
            st.caption(f"Baseline: {baseline_r2:.4f}")

        with st.expander("Baseline Comparison (CatBoost) vs Final Ensemble", expanded=False):
            comp = pd.DataFrame(
                [
                    {"Model": "Baseline (CatBoost)", "MAE": baseline_mae, "RMSE": baseline_rmse, "SMAPE": baseline_smape, "R2": baseline_r2},
                    {"Model": "Final (Ensemble)", "MAE": final_mae, "RMSE": final_rmse, "SMAPE": final_smape, "R2": final_r2},
                ]
            )
            st.dataframe(
                comp.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "SMAPE": "{:.2f}", "R2": "{:.4f}"}),
                width="stretch",
                hide_index=True,
            )
        
        st.markdown("---")
        
        # System info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 **Veri Seti Boyutu:** {final_preds.shape[0] if not final_preds.empty else 0:,} satır")
        with col2:
            st.info(f"🎯 **Model:** {PANEL_SOURCES.label}")
        with col3:
            if dashboard_data:
                st.info(f"🕐 **Son Güncelleme:** {dashboard_data.get('generated_at', '-')[:16]}")

        st.markdown("---")
        st.markdown("### 📉 Gerçek vs Tahmin")
        if leaderboard and PANEL_SOURCES.is_best_5y:
            with st.expander("5Y model karşılaştırması (leaderboard)", expanded=False):
                comp_rows = [
                    {
                        "Model": "CatBoost 5Y (aktif)",
                        "MAE": leaderboard.get("catboost_mae"),
                        "R²": leaderboard.get("catboost_r2"),
                    },
                    {
                        "Model": "Ensemble 5Y",
                        "MAE": leaderboard.get("ensemble_mae"),
                        "R²": leaderboard.get("ensemble_r2"),
                    },
                ]
                if legacy_metrics:
                    comp_rows.append(
                        {
                            "Model": "Final ensemble (eski ~1y)",
                            "MAE": legacy_metrics.get("MAE"),
                            "R²": legacy_metrics.get("R2"),
                        }
                    )
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        if final_preds.empty:
            st.warning(
                f"Tahmin dosyası bulunamadı: `{PANEL_SOURCES.predictions_path.relative_to(ROOT)}`"
            )
        else:
            preds = final_preds.copy()
            required = {"target_datetime", "forecast_horizon", "actual_ptf", "predicted_ptf"}
            if not required.issubset(set(preds.columns)):
                st.error("Final tahmin dosyası formatı beklenen gibi değil. Beklenen kolonlar: " + ", ".join(sorted(required)))
            else:
                horizon_sel = st.selectbox("Horizon (saat)", options=list(range(1, 13)), index=0)
                view_days = st.slider("Görüntülenecek gün", min_value=3, max_value=60, value=14, step=1)

                preds = preds[preds["forecast_horizon"] == int(horizon_sel)].sort_values("target_datetime")
                if preds.empty:
                    st.warning("Bu horizon için tahmin verisi bulunamadı.")
                else:
                    end_dt = preds["target_datetime"].max()
                    start_dt = end_dt - pd.Timedelta(days=int(view_days))
                    preds = preds[preds["target_datetime"] >= start_dt]
                    if preds.empty:
                        st.warning("Seçilen gün aralığında veri yok. Gün sayısını artırmayı deneyin.")
                    else:
                        mae_panel = float((preds["predicted_ptf"] - preds["actual_ptf"]).abs().mean())
                        primary_label = "Panel tahmini"
                        if "primary_model" in preds.columns and not preds["primary_model"].isna().all():
                            primary_label = f"Panel ({preds['primary_model'].iloc[-1]})"
                        st.caption(
                            f"Seçili aralık MAE (h={horizon_sel}): {mae_panel:,.2f} TL/MWh | "
                            f"Nokta: {len(preds):,} | {primary_label}"
                        )

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=preds["target_datetime"],
                                y=preds["actual_ptf"],
                                mode="lines",
                                name="Gerçek PTF (kesinleşmiş)",
                                line=dict(color="#0f172a", width=2.5),
                            )
                        )
                        if "pred_seasonal_blend" in preds.columns:
                            mae_seas = float((preds["pred_seasonal_blend"] - preds["actual_ptf"]).abs().mean())
                            fig.add_trace(
                                go.Scatter(
                                    x=preds["target_datetime"],
                                    y=preds["pred_seasonal_blend"],
                                    mode="lines",
                                    name=f"Mevsimsel baseline (MAE {mae_seas:,.0f})",
                                    line=dict(color="#f59e0b", width=1.8, dash="dot"),
                                )
                            )
                        if "pred_catboost" in preds.columns:
                            mae_cb = float((preds["pred_catboost"] - preds["actual_ptf"]).abs().mean())
                            fig.add_trace(
                                go.Scatter(
                                    x=preds["target_datetime"],
                                    y=preds["pred_catboost"],
                                    mode="lines",
                                    name=f"CatBoost residual (MAE {mae_cb:,.0f})",
                                    line=dict(color="#a855f7", width=2, dash="dash"),
                                )
                            )
                        fig.add_trace(
                            go.Scatter(
                                x=preds["target_datetime"],
                                y=preds["predicted_ptf"],
                                mode="lines",
                                name=f"Panel tahmini (MAE {mae_panel:,.0f})",
                                line=dict(color="#10b981", width=2.5),
                            )
                        )
                        fig.update_layout(
                            height=480,
                            margin=dict(l=20, r=20, t=30, b=20),
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig, width="stretch")
    
    else:
        st.warning("⚠️ Final metrikler bulunamadı. Önce ensemble sistemini çalıştırın.")
        st.code("python main.py")


# --- Data Section ---
elif view_mode == "Veriler":
    st.markdown("## 🗂️ Veriler")
    st.caption("Bu sayfa, panelin kullandığı son `final_hourly_dataset.csv` verisini gösterir.")

    if final_dataset.empty:
        st.error("Final veriseti bulunamadı (`data/processed/final_hourly_dataset.csv`).")
        st.markdown("Panel içinde canlı veri çekip dataset üretmek için sol menüden **Veriyi Güncelle** butonunu kullanın.")
    else:
        st.success(
            f"Yüklendi: {len(final_dataset):,} satır, {final_dataset.shape[1]} kolon | "
            f"Aralık: {final_dataset['datetime'].min()} → {final_dataset['datetime'].max()}"
        )

        st.markdown("### Son Kayıtlar")
        n = st.slider("Gösterilecek satır sayısı", min_value=50, max_value=5000, value=500, step=50)
        df_tail = final_dataset.sort_values("datetime").tail(int(n)).copy()
        st.dataframe(df_tail, width="stretch", hide_index=True)

        st.markdown("### PTF Grafiği (Son 7 Gün)")
        df_plot = final_dataset.dropna(subset=["datetime", "ptf"]).sort_values("datetime").copy()
        if not df_plot.empty:
            end_dt = df_plot["datetime"].max()
            start_dt = end_dt - pd.Timedelta(days=7)
            df_plot = df_plot[df_plot["datetime"] >= start_dt]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_plot["datetime"],
                    y=df_plot["ptf"],
                    mode="lines",
                    name="PTF",
                    line=dict(color="#0ea5e9", width=2),
                )
            )
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, width="stretch")


# --- 5Y EPİAŞ raw data ---
elif view_mode == "5Y EPİAŞ Verileri":
    st.markdown("## 📡 5 Yıllık EPİAŞ Ham Verileri")
    st.caption(
        f"Hedef: **{START_DATE_5Y}** → bugün | Ham veri: `data/raw/epias_5y/` "
        f"| GitHub canlı özet: `app_data/epias_5y_fetch_live.json`"
    )

    sync_fetch_live_to_app_data()
    auto_refresh = st.checkbox("Canlı izleme (5 sn otomatik yenile)", value=True, key="epias5y_live")
    snap = get_live_fetch_snapshot()
    stats = snap["stats"]
    inv = snap["inventory"]
    prog = snap.get("progress") or {}
    idx = int(prog.get("index") or 0)
    tot = int(prog.get("total") or 26)
    cur = prog.get("current") or "—"
    pct = min(1.0, idx / tot) if tot else 0.0

    st.markdown("### Canlı veri çekimi")
    live_cols = st.columns([3, 1, 1, 1])
    with live_cols[0]:
        if snap["running"]:
            st.success(f"Çekim **devam ediyor** — `{cur}` ({idx}/{tot})")
        else:
            st.info("Çekim şu an çalışmıyor (yerel) veya son durum GitHub özetinden okunuyor.")
        st.progress(pct, text=f"İlerleme: {idx}/{tot} seri · Tam 5y: {stats.get('full', 0)}")
    live_cols[1].metric("Tam 5y", stats.get("full", 0))
    live_cols[2].metric("Kısmi", stats.get("partial", 0) + stats.get("short", 0))
    live_cols[3].metric("Eksik", stats.get("missing", 0))

    gh_cols = st.columns(2)
    with gh_cols[0]:
        st.caption(
            f"GitHub özet güncelleme: **{snap.get('app_live_updated') or '—'}**  \n"
            f"Dosya: `{FETCH_LIVE_APP_PATH.name}`"
        )
    with gh_cols[1]:
        st.code(
            "python scripts/push_fetch_live_to_github.py\n"
            "# veya çekim sırasında:\n"
            "python scripts/watch_5y_fetch.py --push-github",
            language="bash",
        )
    st.caption(
        "Streamlit Cloud bu repodan deploy edilir; `push` sonrası panel birkaç dakika içinde güncellenir. "
        "Ham CSV dosyaları git'e gitmez (boyut), yalnızca özet JSON paylaşılır."
    )

    if not inv.empty:
        inv_show = inv.copy()
        if "Durum" in inv_show.columns:
            inv_show["Durum"] = inv_show["Durum"].map(
                {
                    "Tam (5y)": "🟢 Tam (5y)",
                    "Kısmi": "🟡 Kısmi",
                    "Kısa (~1 yıl)": "🟠 Kısa",
                    "Yok": "⚪ Yok",
                }
            ).fillna(inv_show["Durum"])
        st.dataframe(
            inv_show.style.format({"Kapsam %": "{:.1f}"}),
            width="stretch",
            hide_index=True,
            height=min(520, 38 + len(inv_show) * 35),
        )

    if inv.empty and not RAW_5Y_DIR.exists() and not FETCH_LIVE_APP_PATH.exists():
        st.error("Henüz veri yok. Çekim: `python scripts/run_5y_dl_pipeline.py --fetch-only`")
    else:
        if stats.get("full", 0) >= stats.get("total", 26):
            st.success("Tüm seriler 5 yıllık kapsama yakın.")
        elif stats.get("full", 0) >= 8:
            st.warning(
                f"Ana serilerin çoğu tamam ({stats.get('full', 0)}/{stats.get('total', 26)} tam 5y). "
                "Kalan GOP/GİP/denge serileri sırada veya EPİAŞ'ta yok."
            )

        if HOURLY_5Y_PATH.exists():
            h5 = pd.read_csv(HOURLY_5Y_PATH, parse_dates=["datetime"])
            st.markdown("### İşlenmiş 5Y panel (`final_hourly_dataset_5y.csv`)")
            st.success(
                f"{len(h5):,} satır | {h5['datetime'].min()} → {h5['datetime'].max()} | {h5.shape[1]} kolon"
            )

        st.markdown("### Zaman serisi grafikleri")
        plot_options = {
            "PTF kesinleşmiş": ("ptf_kesinlesmis.csv", ["price", "mcp"]),
            "PTF I-MCP": ("ptf_interim.csv", ["marketTradePrice", "price"]),
            "Yük planı": ("load_forecast_plan.csv", ["lep"]),
            "RES üretim/tahmin": ("res_generation_forecast.csv", None),
            "Gerçek zamanlı üretim (toplam)": ("realtime_generation.csv", ["total"]),
            "SMF": ("smf.csv", ["systemMarginalPrice"]),
            "Sistem yönü": ("system_direction.csv", ["systemDirection", "direction"]),
        }
        choice = st.multiselect(
            "Gösterilecek seriler",
            list(plot_options.keys()),
            default=["PTF kesinleşmiş", "RES üretim/tahmin"],
        )
        days = st.slider("Grafik penceresi (gün)", 7, 365 * 5, 90, 7)

        for name in choice:
            fname, cols = plot_options[name]
            ts = load_epias_5y_timeseries(fname, cols)
            if ts.empty:
                st.info(f"{name}: veri yok veya okunamadı.")
                continue
            end = ts["datetime"].max()
            ts = ts[ts["datetime"] >= end - pd.Timedelta(days=int(days))]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=ts["datetime"], y=ts["value"], mode="lines", name=name, line=dict(width=1.5))
            )
            fig.update_layout(title=name, height=360, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width="stretch")

        with st.expander("Ham CSV önizleme"):
            pick = st.selectbox("Dosya", list(FEATURE_LABELS.keys()))
            p = RAW_5Y_DIR / pick
            if p.exists():
                preview = pd.read_csv(p).tail(200)
                st.dataframe(preview, width="stretch", hide_index=True)
            else:
                st.caption("Dosya henüz yok.")

    if auto_refresh and is_fetch_process_running():
        import time

        time.sleep(5)
        st.rerun()


# --- Ensemble Details Section ---
elif view_mode == "Ensemble Detayları":
    st.markdown("## 🎯 Ensemble Ağırlıkları ve Bias Düzeltmeleri")
    
    if final_metrics and "ensemble_weights" in final_metrics:
        weights = final_metrics["ensemble_weights"]
        bias = final_metrics["bias_corrections"]
        
        # Weight visualization
        st.markdown("### Horizon Bazlı Ağırlıklar")
        
        weight_data = []
        for horizon in range(1, 13):
            w = weights.get(horizon, {})
            weight_data.append({
                "Horizon": f"{horizon}h",
                "CatBoost": w.get("pred_catboost", 0) * 100,
                "Dün Aynı Saat": w.get("pred_same_hour_yesterday", 0) * 100,
                "Geçen Hafta Aynı Saat": w.get("pred_same_hour_last_week", 0) * 100,
                "Son 24s Ort": w.get("pred_last_24h_mean", 0) * 100,
                "Rolling 24s": w.get("pred_rolling_24h", 0) * 100,
                "Rolling 168s": w.get("pred_rolling_168h", 0) * 100,
            })
        
        weight_df = pd.DataFrame(weight_data)
        weight_df = weight_df.set_index("Horizon")
        
        st.dataframe(weight_df.style.format("{:.1f}%"), width="stretch")
        
        # Bias correction
        st.markdown("### Horizon Bazlı Bias Düzeltmeleri")
        
        bias_data = []
        for horizon in range(1, 13):
            bias_data.append({
                "Horizon": f"{horizon}h",
                "Bias Düzeltmesi (TL)": bias.get(horizon, 0)
            })
        
        bias_df = pd.DataFrame(bias_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=bias_df["Horizon"],
            y=bias_df["Bias Düzeltmesi (TL)"],
            marker_color="#6366f1"
        ))
        fig.update_layout(
            title="Bias Düzeltmeleri (Negatif = Underprediction düzeltmesi)",
            xaxis_title="Tahmin Ufku",
            yaxis_title="Bias (TL)",
            height=400
        )
        st.plotly_chart(fig, width="stretch")
        
        # Key insights
        st.markdown("### 🔍 Temel Bulgular")
        st.markdown("""
        - **CatBoost ağırlığı 0%**: Log dönüşümü nedeniyle sistemik underprediction
        - **Rolling 168h dominant**: Uzun horizonlarda haftalık desenler önemli
        - **Rolling 24h önemli**: Kısa horizonlarda günlük desenler etkili
        - **Bias düzeltmeleri negatif**: Sistemik underprediction'ı düzeltiyor
        """)
    elif PANEL_SOURCES.is_best_5y and leaderboard.get("ensemble_metrics", {}).get("ensemble_weights"):
        st.info("Aktif model CatBoost; aşağıda 5Y ensemble ağırlıkları (karşılaştırma).")
        em = leaderboard["ensemble_metrics"]
        weights = em["ensemble_weights"]
        bias = em.get("bias_corrections", {})
        weight_data = []
        for horizon in range(1, 13):
            w = weights.get(str(horizon), weights.get(horizon, {}))
            weight_data.append({
                "Horizon": f"{horizon}h",
                "CatBoost": w.get("pred_catboost", 0) * 100,
                "Dün Aynı Saat": w.get("pred_kesin_same_hour_yesterday", 0) * 100,
                "Geçen Hafta": w.get("pred_kesin_same_hour_last_week", 0) * 100,
                "Rolling 24s": w.get("pred_kesin_rolling_24h", 0) * 100,
            })
        st.dataframe(pd.DataFrame(weight_data).set_index("Horizon").style.format("{:.1f}%"), width="stretch")
    else:
        st.warning("⚠️ Ensemble ağırlıkları bulunamadı (CatBoost aktif modelde ağırlık tablosu yok).")


# --- DL Models (5Y) ---
elif view_mode == "DL Modeller (5Y)":
    st.markdown("## 🧠 Derin Öğrenme Modelleri (5Y)")
    st.caption("Kaynak: `forecast_12h_sequence_dataset_5y.npz` ile eğitilmiş LSTM / CNN-LSTM. (CPU uyumlu hızlı baseline)")

    comp = load_dl_comparison_5y()
    metrics = load_dl_metrics_5y()

    if comp.empty:
        st.warning("DL metrikleri bulunamadı (`data/processed/dl_models_comparison_5y.csv`).")
        st.code("python -c 'from src.train_dl_models_5y import train_dl_baselines_5y; train_dl_baselines_5y()'")
    else:
        show = comp.copy()
        # normalize types
        for c in ("mae", "rmse", "smape", "r2"):
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
        show = show.sort_values("mae")

        best_model = metrics.get("best_model") if isinstance(metrics, dict) else None
        if not best_model and "model_type" in show.columns and show["mae"].notna().any():
            best_model = str(show.iloc[0]["model_type"])

        cols = st.columns(3)
        cols[0].metric("Model sayısı", int(len(show)))
        cols[1].metric("En iyi model", best_model or "-")
        if best_model and "mae" in show.columns and show["mae"].notna().any():
            best_row = show[show["model_type"] == best_model].head(1)
            if not best_row.empty:
                cols[2].metric("En iyi MAE", f"{float(best_row.iloc[0]['mae']):.1f}")
            else:
                cols[2].metric("En iyi MAE", f"{float(show.iloc[0]['mae']):.1f}")

        st.markdown("### Sonuç Tablosu")
        st.dataframe(
            show.style.format({"mae": "{:.2f}", "rmse": "{:.2f}", "smape": "{:.2f}", "r2": "{:.4f}", "train_seconds": "{:.0f}"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("### Grafik")
        metric_choice = st.selectbox("Metrik", options=["mae", "rmse", "smape", "r2"], index=0)
        if metric_choice in show.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=show["model_type"],
                    y=show[metric_choice],
                    marker_color=["#10b981" if str(m) == str(best_model) else "#94a3b8" for m in show["model_type"]],
                )
            )
            title_map = {"mae": "MAE (düşük iyi)", "rmse": "RMSE (düşük iyi)", "smape": "SMAPE (düşük iyi)", "r2": "R² (yüksek iyi)"}
            fig.update_layout(
                title=f"DL Model Karşılaştırma — {title_map.get(metric_choice, metric_choice)}",
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig, width="stretch")


# --- Live Forecast Section ---
elif view_mode == "Canlı Tahmin":
    st.markdown("## 🔮 Canlı 12 Saat Tahmini")
    st.caption("Kesim anındaki I-MCP, naive baseline, CatBoost ve final ensemble birlikte gösterilir.")

    # If user selects DL model, generate a lightweight live forecast using the last available 5Y window.
    if st.session_state.get("forecast_model_choice") in ("CNN-LSTM (5Y)", "LSTM (5Y)"):
        try:
            from src.dl_5y_inference import predict_next_12h_from_5y

            mt = "cnn_lstm" if st.session_state.get("forecast_model_choice") == "CNN-LSTM (5Y)" else "lstm"
            dl = predict_next_12h_from_5y(model_type=mt)
            live_bundle = dl.output.rename(columns={"predicted_ptf": "ensemble_ptf"})
            st.info(f"DL aktif: {dl.model_type} | issue: {dl.issue_datetime}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"DL live forecast üretilemedi: {exc}")
            live_bundle = load_live_bundle()
    else:
        live_bundle = load_live_bundle()
    if live_bundle.empty and not live_df.empty:
        rename_map = {}
        if "predicted_ptf" in live_df.columns and "ensemble_ptf" not in live_df.columns:
            rename_map["predicted_ptf"] = "ensemble_ptf"
        live_bundle = live_df.rename(columns=rename_map)

    if live_bundle.empty or len(live_bundle) < 12:
        if not final_dataset.empty:
            naive_only = build_naive_live_forecast(final_dataset, horizons=12)
            if not naive_only.empty:
                cutoff_dt = final_dataset.dropna(subset=["datetime", "ptf"])["datetime"].max()
                interim = float(
                    final_dataset.loc[final_dataset["datetime"] == cutoff_dt, "ptf"].iloc[0]
                )
                live_bundle = naive_only.copy()
                live_bundle["issue_datetime"] = cutoff_dt
                live_bundle["interim_ptf"] = interim
                live_bundle["naive_ptf"] = live_bundle["predicted_ptf"]
                live_bundle["ensemble_ptf"] = live_bundle["predicted_ptf"]
                live_bundle["catboost_ptf"] = np.nan

    if not live_bundle.empty:
        cutoff_display = dashboard_data.get("cutoff_datetime") if dashboard_data else None
        if not cutoff_display and "issue_datetime" in live_bundle.columns:
            cutoff_display = str(live_bundle["issue_datetime"].max())
        if cutoff_display:
            st.info(f"🕐 **Kesim (son I-MCP saati):** {cutoff_display}")

        if "interim_ptf" in live_bundle.columns and live_bundle["interim_ptf"].notna().any():
            st.metric(
                "Son I-MCP (TL/MWh)",
                f"{live_bundle['interim_ptf'].iloc[0]:,.2f}",
            )

        live_bundle = live_bundle.sort_values("forecast_horizon")
        fig = go.Figure()
        if "interim_ptf" in live_bundle.columns:
            fig.add_trace(
                go.Scatter(
                    x=live_bundle["target_datetime"],
                    y=[live_bundle["interim_ptf"].iloc[0]] * len(live_bundle),
                    mode="lines",
                    name="I-MCP (kesim)",
                    line=dict(color="#94a3b8", width=2, dash="dot"),
                )
            )
        if "naive_ptf" in live_bundle.columns:
            fig.add_trace(
                go.Scatter(
                    x=live_bundle["target_datetime"],
                    y=live_bundle["naive_ptf"],
                    mode="lines+markers",
                    name="Naive",
                    line=dict(color="#f97316", width=2),
                )
            )
        if "catboost_ptf" in live_bundle.columns and live_bundle["catboost_ptf"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=live_bundle["target_datetime"],
                    y=live_bundle["catboost_ptf"],
                    mode="lines+markers",
                    name="CatBoost (delta)",
                    line=dict(color="#a855f7", width=2, dash="dash"),
                )
            )
        panel_col = "panel_ptf" if "panel_ptf" in live_bundle.columns else "ensemble_ptf"
        if panel_col not in live_bundle.columns:
            panel_col = "predicted_ptf" if "predicted_ptf" in live_bundle.columns else "ensemble_ptf"
        primary_note = ""
        if "primary_model" in live_bundle.columns:
            primary_note = f" [{live_bundle['primary_model'].iloc[0]}]"
        fig.add_trace(
            go.Scatter(
                x=live_bundle["target_datetime"],
                y=live_bundle[panel_col],
                mode="lines+markers",
                name=f"Panel tahmini{primary_note}",
                line=dict(color="#0ea5e9", width=3),
                marker=dict(size=9),
            )
        )
        fig.update_layout(
            title="Gelecek 12 Saat — I-MCP / Naive / CatBoost / Ensemble",
            yaxis_title="PTF (TL/MWh)",
            xaxis_title="Hedef saat",
            hovermode="x unified",
            height=520,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )
        st.plotly_chart(fig, width="stretch")

        table_cols = {
            "forecast_horizon": "Saat (+)",
            "target_datetime": "Hedef Zaman",
            "interim_ptf": "I-MCP (kesim)",
            "seasonal_ptf": "Mevsimsel",
            "catboost_ptf": "CatBoost",
            panel_col: "Panel tahmin",
        }
        show_cols = [c for c in table_cols if c in live_bundle.columns]
        table = live_bundle[show_cols].rename(columns=table_cols)
        fmt = {v: "{:.2f}" for v in table_cols.values() if v != "Hedef Zaman" and v != "Saat (+)"}
        st.dataframe(table.style.format(fmt), width="stretch", hide_index=True)
    else:
        st.warning("⚠️ Canlı tahmin yok. Sidebar'dan **Veriyi Güncelle** ile pipeline çalıştırın.")


# --- Performance Analysis Section ---
elif view_mode == "Performans Analizi":
    st.markdown("## 📊 Horizon Bazlı Performans Analizi")
    
    if not final_horizon.empty and not baseline_horizon.empty:
        # Merge horizon metrics
        comparison = baseline_horizon.merge(
            final_horizon, 
            on="forecast_horizon", 
            suffixes=("_baseline", "_final")
        )
        comparison["mae_improvement"] = (comparison["MAE_baseline"] - comparison["MAE_final"]) / comparison["MAE_baseline"] * 100
        comparison["r2_improvement"] = comparison["R2_final"] - comparison["R2_baseline"]
        
        # MAE comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison["forecast_horizon"],
            y=comparison["MAE_baseline"],
            name="Baseline (CatBoost)",
            marker_color="#94a3b8"
        ))
        fig.add_trace(go.Bar(
            x=comparison["forecast_horizon"],
            y=comparison["MAE_final"],
            name="Final Ensemble",
            marker_color="#10b981"
        ))
        fig.update_layout(
            title="MAE Karşılaştırması - Horizon Bazında",
            xaxis_title="Tahmin Ufku (Saat)",
            yaxis_title="MAE (TL/MWh)",
            barmode="group",
            height=450
        )
        st.plotly_chart(fig, width="stretch")
        
        # R² comparison chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=comparison["forecast_horizon"],
            y=comparison["R2_baseline"],
            name="Baseline (CatBoost)",
            marker_color="#94a3b8"
        ))
        fig2.add_trace(go.Bar(
            x=comparison["forecast_horizon"],
            y=comparison["R2_final"],
            name="Final Ensemble",
            marker_color="#10b981"
        ))
        fig2.update_layout(
            title="R² Karşılaştırması - Horizon Bazında",
            xaxis_title="Tahmin Ufku (Saat)",
            yaxis_title="R²",
            barmode="group",
            height=450
        )
        st.plotly_chart(fig2, width="stretch")
        
        # Improvement table
        st.markdown("### İyileştirme Tablosu")
        improvement_table = comparison[[
            "forecast_horizon", 
            "MAE_baseline", 
            "MAE_final", 
            "mae_improvement",
            "R2_baseline",
            "R2_final",
            "r2_improvement"
        ]].rename(columns={
            "forecast_horizon": "Horizon (h)",
            "MAE_baseline": "Baseline MAE",
            "MAE_final": "Ensemble MAE",
            "mae_improvement": "MAE İyileştirme (%)",
            "R2_baseline": "Baseline R²",
            "R2_final": "Ensemble R²",
            "r2_improvement": "R² İyileştirme"
        })
        
        def highlight_positive(val):
            color = '#d4edda' if val > 0 else '#f8d7da'
            return f'background-color: {color}'
        
        st.dataframe(
            improvement_table.style.format({
                "Baseline MAE": "{:.2f}",
                "Ensemble MAE": "{:.2f}",
                "MAE İyileştirme (%)": "{:.1f}",
                "Baseline R²": "{:.4f}",
                "Ensemble R²": "{:.4f}",
                "R² İyileştirme": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )
        
        # Key findings
        st.markdown("### 🔍 Performans Bulguları")
        best_horizon = comparison.loc[comparison["mae_improvement"].idxmax()]
        worst_horizon = comparison.loc[comparison["mae_improvement"].idxmin()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**En İyi Horizon:** {best_horizon['forecast_horizon']}h")
            st.caption(f"MAE İyileştirme: {best_horizon['mae_improvement']:.1f}%")
        with col2:
            st.info(f"En Kötü Horizon: {worst_horizon['forecast_horizon']}h")
            st.caption(f"MAE İyileştirme: {worst_horizon['mae_improvement']:.1f}%")
        
    else:
        st.warning("⚠️ Horizon metrikleri bulunamadı.")
