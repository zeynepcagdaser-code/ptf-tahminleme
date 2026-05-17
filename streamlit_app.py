from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
APP_DATA_PATH = ROOT / "app_data" / "dashboard_data.json"
LOCAL_PROCESSED_DIR = ROOT / "data" / "processed"


st.set_page_config(
    page_title="PTF Tahminleme Paneli",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict:
    if APP_DATA_PATH.exists():
        return json.loads(APP_DATA_PATH.read_text(encoding="utf-8"))
    return build_from_local_processed()


def build_from_local_processed() -> dict:
    def read_csv(name: str, **kwargs) -> pd.DataFrame:
        path = LOCAL_PROCESSED_DIR / name
        return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()

    metrics = read_csv("d_plus_2_metrics_comparison.csv")
    latest = read_csv("latest_available_times_report.csv")
    missing = read_csv("final_dataset_missing_report.csv")
    final = read_csv("final_hourly_dataset.csv", usecols=["datetime", "ptf"])
    before = read_csv("d_plus_2_before_14_predictions.csv")
    after = read_csv("d_plus_2_after_14_predictions.csv")

    summary = {}
    if not final.empty:
        final["datetime"] = pd.to_datetime(final["datetime"], errors="coerce")
        summary = {
            "final_rows": int(len(final)),
            "final_start": str(final["datetime"].min()),
            "final_end": str(final["datetime"].max()),
            "ptf_min": float(final["ptf"].min()),
            "ptf_max": float(final["ptf"].max()),
            "ptf_mean": float(final["ptf"].mean()),
        }

    return {
        "generated_at": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S %Z"),
        "summary": summary,
        "metrics": metrics.to_dict(orient="records"),
        "latest_available": latest.to_dict(orient="records"),
        "missing_report": missing.to_dict(orient="records"),
        "predictions": {
            "before_14": prediction_records(before),
            "after_14": prediction_records(after),
        },
    }


def prediction_records(frame: pd.DataFrame, limit: int = 336) -> list[dict]:
    if frame.empty:
        return []
    keep = ["target_datetime", "actual_ptf", "predicted_ptf", "absolute_error"]
    available = [column for column in keep if column in frame.columns]
    frame = frame[available].tail(limit).copy()
    for column in ["actual_ptf", "predicted_ptf", "absolute_error"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.to_dict(orient="records")


def format_number(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.{digits}f}"


def scenario_label(value: str) -> str:
    return "14:00 Sonrası" if value == "after_14" else "14:00 Öncesi"


data = load_dashboard_data()
summary = data.get("summary", {})
metrics = pd.DataFrame(data.get("metrics", []))
latest = pd.DataFrame(data.get("latest_available", []))
missing = pd.DataFrame(data.get("missing_report", []))

st.title("PTF Tahminleme Paneli")
st.caption("D+2 PTF tahmini için Streamlit paneli. Veri snapshot'ı GitHub reposundan yüklenir.")

with st.sidebar:
    st.header("Panel")
    scenario = st.radio(
        "Tahmin senaryosu",
        ["before_14", "after_14"],
        format_func=scenario_label,
        index=1,
    )
    st.divider()
    st.caption(f"Snapshot zamanı: {data.get('generated_at', '-')}")

metric_cols = st.columns(5)
metric_cols[0].metric("Final Satır", f"{summary.get('final_rows', 0):,}")
metric_cols[1].metric("Başlangıç", str(summary.get("final_start", "-"))[:10])
metric_cols[2].metric("Bitiş", str(summary.get("final_end", "-"))[:10])
metric_cols[3].metric("PTF Ortalama", format_number(summary.get("ptf_mean")))

after_row = metrics.loc[metrics.get("scenario", pd.Series(dtype=str)).eq("after_14")]
after_r2 = after_row["R2"].iloc[0] if not after_row.empty and "R2" in after_row else None
metric_cols[4].metric("After 14 R2", format_number(after_r2, 3))

tab_overview, tab_predictions, tab_data = st.tabs(["Performans", "Tahminler", "Veri Durumu"])

with tab_overview:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Senaryo Performansı")
        if not metrics.empty:
            fig = go.Figure()
            fig.add_bar(x=metrics["scenario"].map(scenario_label), y=metrics["RMSE"], name="RMSE")
            fig.add_bar(x=metrics["scenario"].map(scenario_label), y=metrics["MAE"], name="MAE")
            fig.update_layout(
                barmode="group",
                height=430,
                margin=dict(l=20, r=20, t=30, b=20),
                yaxis_title="Hata",
                legend_orientation="h",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Metrik dosyası bulunamadı.")
    with right:
        st.subheader("Metrik Tablosu")
        if not metrics.empty:
            display = metrics[["scenario", "feature_count", "MAE", "RMSE", "SMAPE", "R2"]].copy()
            display["scenario"] = display["scenario"].map(scenario_label)
            st.dataframe(display, width="stretch", hide_index=True)
        else:
            st.info("Gösterilecek metrik yok.")

with tab_predictions:
    rows = pd.DataFrame(data.get("predictions", {}).get(scenario, []))
    st.subheader(f"{scenario_label(scenario)} - Son Test Tahminleri")
    if not rows.empty:
        rows["target_datetime"] = pd.to_datetime(rows["target_datetime"], errors="coerce")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rows["target_datetime"],
                y=rows["actual_ptf"],
                mode="lines",
                name="Gerçek PTF",
                line=dict(width=2, color="#17202A"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rows["target_datetime"],
                y=rows["predicted_ptf"],
                mode="lines",
                name="Tahmin",
                line=dict(width=2, color="#0B6BCB"),
            )
        )
        fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), yaxis_title="PTF")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Tahmin Tablosu")
        st.dataframe(rows.tail(96), width="stretch", hide_index=True)
    else:
        st.warning("Tahmin verisi bulunamadı.")

with tab_data:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Son Yayınlanan Veri Zamanları")
        if not latest.empty:
            columns = [
                "data_source",
                "latest_available_time",
                "inferred_frequency",
                "delay_text",
                "forecast_usage_recommendation",
            ]
            st.dataframe(latest[[column for column in columns if column in latest.columns]], width="stretch", hide_index=True)
        else:
            st.info("Son yayın zamanı raporu bulunamadı.")
    with right:
        st.subheader("Eksik Veri Raporu")
        if not missing.empty:
            columns = ["column", "missing_count_before", "missing_count_after"]
            st.dataframe(missing[[column for column in columns if column in missing.columns]], width="stretch", hide_index=True)
        else:
            st.info("Eksik veri raporu bulunamadı.")
