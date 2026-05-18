from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from src.config import PROJECT_ROOT


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
APP_DATA_PATH = ROOT / "app_data" / "dashboard_data.json"


st.set_page_config(
    page_title="PTF 12 Saat Tahmin",
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
        padding: 1.2rem 1.4rem;
        border-radius: 14px;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.85rem; }
    .hero p { margin: 0.4rem 0 0; opacity: 0.95; }
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


def read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_dashboard_payload() -> dict:
    if APP_DATA_PATH.exists():
        return json.loads(APP_DATA_PATH.read_text(encoding="utf-8"))
    return build_payload()


def build_payload() -> dict:
    metrics = {}
    metrics_path = PROCESSED / "ptf_12h_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    live = read_csv("ptf_12h_live_forecast.csv")
    preds = read_csv("ptf_12h_predictions.csv")
    horizon = read_csv("ptf_12h_horizon_metrics.csv")
    hourly = read_csv("final_hourly_dataset.csv")

    if not live.empty:
        live["issue_datetime"] = pd.to_datetime(live["issue_datetime"], errors="coerce").astype(str)
        live["target_datetime"] = pd.to_datetime(live["target_datetime"], errors="coerce").astype(str)

    if not preds.empty:
        preds["issue_datetime"] = pd.to_datetime(preds["issue_datetime"], errors="coerce").astype(str)
        preds["target_datetime"] = pd.to_datetime(preds["target_datetime"], errors="coerce").astype(str)

    cutoff = None
    if not live.empty:
        cutoff = live["issue_datetime"].iloc[0]
    elif not hourly.empty:
        hourly["datetime"] = pd.to_datetime(hourly["datetime"], errors="coerce")
        ptf = hourly.dropna(subset=["ptf"])
        if not ptf.empty:
            cutoff = str(ptf["datetime"].max())

    return {
        "generated_at": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S %Z"),
        "cutoff_datetime": cutoff,
        "metrics": metrics,
        "live_forecast": live.to_dict(orient="records"),
        "test_predictions": preds.to_dict(orient="records"),
        "horizon_metrics": horizon.to_dict(orient="records"),
    }


def save_payload(payload: dict) -> None:
    APP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_DATA_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def fmt(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


@st.cache_data(ttl=1800, show_spinner=False)
def run_live_update() -> dict:
    from src.run_ptf_pipeline import run_full_ptf_pipeline

    run_full_ptf_pipeline(fetch_live=True)
    payload = build_payload()
    save_payload(payload)
    return payload


# --- Sidebar ---
with st.sidebar:
    st.header("PTF Tahmin")
    view_mode = st.radio("Görünüm", ["Canlı 12 Saat Tahmin", "Test Performansı"], index=0)
    st.divider()
    refresh = st.button("Veriyi çek ve yeniden eğit", type="primary")
    st.caption("İlk kurulum: `python main.py`")
    st.caption("Canlı güncelleme EPİAŞ kimlik bilgisi ister.")

if refresh:
    run_live_update.clear()

if refresh and has_credentials():
    with st.spinner("EPİAŞ verisi çekiliyor, model eğitiliyor..."):
        try:
            data = run_live_update()
            st.success("Güncelleme tamamlandı.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Hata: {exc}")
            data = load_dashboard_payload()
elif refresh:
    st.warning("EPİAŞ secrets gerekli.")
    data = load_dashboard_payload()
else:
    data = load_dashboard_payload()

metrics = data.get("metrics", {})
live_df = pd.DataFrame(data.get("live_forecast", []))
test_df = pd.DataFrame(data.get("test_predictions", []))
horizon_df = pd.DataFrame(data.get("horizon_metrics", []))

st.markdown(
    """
    <div class="hero">
      <h1>PTF — Sonraki 12 Saat Tahmini</h1>
      <p>Son açıklanan kesinleşmemiş PTF (I-MCP) saatinden itibaren gelecek 12 saat için kesinleşmiş PTF (MCP) tahmini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if live_df.empty and test_df.empty:
    st.warning("Henüz tahmin yok. Terminalde `python main.py` çalıştırın.")

h1_row = horizon_df.loc[horizon_df["forecast_horizon"] == 1].iloc[0] if not horizon_df.empty else None

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Son I-MCP Saati", str(data.get("cutoff_datetime", "-"))[:16])
k2.metric("+1 Saat MAE", fmt(h1_row["MAE"]) if h1_row is not None else "-")
k3.metric("+1 Saat R²", fmt(h1_row["R2"], 3) if h1_row is not None else "-")
k4.metric("12 Saat Ort. MAE", fmt(metrics.get("MAE")))
k5.metric("12 Saat Ort. R²", fmt(metrics.get("R2"), 3))
k6.metric("Güncelleme", str(data.get("generated_at", "-"))[:16])

if view_mode == "Canlı 12 Saat Tahmin":
    st.subheader("Gelecek 12 Saat — Tahmin Edilen Kesinleşmiş PTF (MCP)")
    if not live_df.empty:
        live_df["target_datetime"] = pd.to_datetime(live_df["target_datetime"], errors="coerce")
        live_df = live_df.sort_values("forecast_horizon")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=live_df["target_datetime"],
                y=live_df["predicted_ptf"],
                mode="lines+markers",
                name="Tahmin PTF",
                line=dict(color="#0ea5e9", width=3),
                marker=dict(size=9),
            )
        )
        fig.update_layout(
            height=480,
            yaxis_title="PTF (TL/MWh)",
            xaxis_title="Saat",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        table = live_df.rename(
            columns={
                "forecast_horizon": "Saat (+)",
                "target_datetime": "Hedef Zaman",
                "predicted_ptf": "Tahmin PTF",
            }
        )[["Saat (+)", "Hedef Zaman", "Tahmin PTF"]]
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("Canlı tahmin dosyası yok.")

else:
    st.subheader("Test Seti — Gerçek vs Tahmin (son dönem)")
    if not test_df.empty:
        test_df["target_datetime"] = pd.to_datetime(test_df["target_datetime"], errors="coerce")
        recent_issue = test_df["issue_datetime"].max()
        subset = test_df[test_df["issue_datetime"] == recent_issue].copy()
        if subset.empty:
            subset = test_df.sort_values("target_datetime").tail(12)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.06)
        fig.add_trace(
            go.Scatter(
                x=subset["target_datetime"],
                y=subset["actual_ptf"],
                mode="lines+markers",
                name="Gerçek PTF",
                line=dict(color="#111827", width=2.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=subset["target_datetime"],
                y=subset["predicted_ptf"],
                mode="lines+markers",
                name="Tahmin PTF",
                line=dict(color="#0ea5e9", width=2.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=subset["target_datetime"],
                y=subset["absolute_error"],
                name="Mutlak Hata",
                marker_color="rgba(239,68,68,0.5)",
            ),
            row=2,
            col=1,
        )
        fig.update_layout(height=560, yaxis_title="PTF", yaxis2_title="Hata")
        st.plotly_chart(fig, use_container_width=True)

        if not horizon_df.empty:
            st.subheader("Saat Ufku Bazında Hata (MAE)")
            hfig = go.Figure()
            hfig.add_bar(
                x=horizon_df["forecast_horizon"],
                y=horizon_df["MAE"],
                marker_color="#6366f1",
            )
            hfig.update_layout(
                xaxis_title="Tahmin ufku (saat)",
                yaxis_title="MAE (TL/MWh)",
                height=320,
            )
            st.plotly_chart(hfig, use_container_width=True)
    else:
        st.info("Test tahmin dosyası yok. `python main.py` çalıştırın.")
