from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
APP_DATA_PATH = ROOT / "app_data" / "dashboard_data.json"
LOCAL_PROCESSED_DIR = ROOT / "data" / "processed"

FEATURE_LABELS_TR = {
    "ptf": "PTF",
    "smf": "SMF",
    "real_time_consumption": "Gerçek Zamanlı Tüketim",
    "wind_generation": "Rüzgar Üretimi",
    "solar_generation": "Güneş Üretimi",
    "hydro_dam_generation": "Barajlı Hidro Üretimi",
    "unlicensed_generation_total": "Lisanssız Üretim Toplamı",
    "load_forecast_plan": "Yük Tahmin Planı",
    "grf_tl": "Günlük Referans Fiyatı TL",
    "usd_try": "USD/TRY",
    "gop_fiyattan_bagimsiz_alis": "GÖP Fiyattan Bağımsız Alış",
    "gop_fiyattan_bagimsiz_satis": "GÖP Fiyattan Bağımsız Satış",
    "price_independent_buy_sell_ratio": "Fiyattan Bağımsız Alış/Satış Oranı",
}

COLUMN_LABELS_TR = {
    "scenario": "Senaryo",
    "feature_count": "Özellik Sayısı",
    "MAE": "MAE",
    "RMSE": "RMSE",
    "SMAPE": "SMAPE",
    "R2": "R²",
    "target_datetime": "Hedef Tarih/Saat",
    "actual_ptf": "Gerçek PTF",
    "predicted_ptf": "Tahmin PTF",
    "absolute_error": "Mutlak Hata",
    "data_source": "Veri Kaynağı",
    "latest_available_time": "Son Yayınlanan Zaman",
    "inferred_frequency": "Veri Frekansı",
    "delay_text": "Gecikme",
    "forecast_usage_recommendation": "Tahminde Kullanım Önerisi",
    "column": "Kolon",
    "missing_count_before": "Doldurma Öncesi Eksik Sayısı",
    "missing_count_after": "Doldurma Sonrası Eksik Sayısı",
}

FREQUENCY_LABELS_TR = {
    "hourly": "Saatlik",
    "daily": "Günlük",
    "monthly": "Aylık",
}


st.set_page_config(
    page_title="PTF Tahminleme Paneli",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_snapshot_dashboard_data() -> dict:
    if APP_DATA_PATH.exists():
        return json.loads(APP_DATA_PATH.read_text(encoding="utf-8"))
    return build_from_local_processed()


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_dashboard_data() -> dict:
    snapshot = load_snapshot_dashboard_data()

    from src.build_final_hourly_dataset import build_final_hourly_dataset
    from src.check_latest_available_times import run_latest_available_times_check
    from src.fetch_final_selected_features import run_fetch_final_selected_features
    from src.fill_final_dataset_missing import fill_final_dataset_missing

    fetch_summary = run_fetch_final_selected_features()
    raw_dataset, flags = build_final_hourly_dataset()
    final_dataset, missing_report = fill_final_dataset_missing(raw_dataset)

    output_path = LOCAL_PROCESSED_DIR / "final_hourly_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.to_csv(output_path, index=False)

    latest_report = run_latest_available_times_check()
    summary = build_final_summary(final_dataset)
    summary.update(
        {
            "attempted_features": fetch_summary.get("attempted_features", 0),
            "successful_features": fetch_summary.get("successful_features", 0),
            "failed_features": fetch_summary.get("failed_features", 0),
            "ratio_created": bool(flags.get("ratio_created")),
            "grf_daily_broadcast": bool(flags.get("grf_daily_broadcast")),
            "usd_daily_broadcast": bool(flags.get("usd_daily_broadcast")),
        }
    )

    return {
        "generated_at": pd.Timestamp.now(tz="Europe/Istanbul").strftime("%Y-%m-%d %H:%M:%S %Z"),
        "source": "live_epias",
        "summary": summary,
        "metrics": snapshot.get("metrics", []),
        "latest_available": latest_report.to_dict(orient="records"),
        "missing_report": missing_report.to_dict(orient="records"),
        "predictions": snapshot.get("predictions", {}),
    }


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
        "source": "local_processed",
        "summary": summary,
        "metrics": metrics.to_dict(orient="records"),
        "latest_available": latest.to_dict(orient="records"),
        "missing_report": missing.to_dict(orient="records"),
        "predictions": {
            "before_14": prediction_records(before),
            "after_14": prediction_records(after),
        },
    }


def build_final_summary(final: pd.DataFrame) -> dict:
    if final.empty:
        return {}

    frame = final.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return {
        "final_rows": int(len(frame)),
        "final_start": str(frame["datetime"].min()),
        "final_end": str(frame["datetime"].max()),
        "ptf_min": float(frame["ptf"].min()),
        "ptf_max": float(frame["ptf"].max()),
        "ptf_mean": float(frame["ptf"].mean()),
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


def localized_dataframe(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame

    display = frame.copy()
    if columns is not None:
        display = display[[column for column in columns if column in display.columns]]

    for column in ("data_source", "column"):
        if column in display.columns:
            display[column] = display[column].map(FEATURE_LABELS_TR).fillna(display[column])

    if "scenario" in display.columns:
        display["scenario"] = display["scenario"].map(scenario_label).fillna(display["scenario"])

    if "inferred_frequency" in display.columns:
        display["inferred_frequency"] = display["inferred_frequency"].map(FREQUENCY_LABELS_TR).fillna(
            display["inferred_frequency"]
        )

    return display.rename(columns=COLUMN_LABELS_TR)


def apply_streamlit_secrets_to_env() -> None:
    load_dotenv(ROOT / ".env")
    try:
        secrets = dict(st.secrets)
    except Exception:
        secrets = {}

    for key in (
        "EPIAS_USERNAME",
        "EPIAS_PASSWORD",
        "EPIAS_TGT",
        "EPIAS_TIMEOUT_SECONDS",
        "EPIAS_PAGE_SIZE",
        "EPIAS_BASE_URL",
        "EPIAS_AUTH_URL",
    ):
        value = secrets.get(key)
        if value is not None and str(value).strip():
            os.environ[key] = str(value)


def has_epias_credentials() -> bool:
    apply_streamlit_secrets_to_env()
    return bool(os.getenv("EPIAS_TGT") or (os.getenv("EPIAS_USERNAME") and os.getenv("EPIAS_PASSWORD")))


st.title("PTF Tahminleme Paneli")

with st.sidebar:
    st.header("Panel")
    scenario = st.radio(
        "Tahmin senaryosu",
        ["before_14", "after_14"],
        format_func=scenario_label,
        index=1,
    )
    st.divider()
    live_mode = st.toggle("Canlı EPİAŞ verisi", value=True)
    refresh_now = st.button("Veriyi şimdi güncelle", type="primary")
    st.caption("Canlı mod Streamlit Secrets içindeki EPİAŞ bilgileriyle çalışır.")

if refresh_now:
    load_live_dashboard_data.clear()

credentials_ready = has_epias_credentials()

if live_mode and credentials_ready:
    with st.spinner("EPİAŞ verileri canlı olarak çekiliyor ve final dataset güncelleniyor..."):
        try:
            data = load_live_dashboard_data()
            data_source_label = "Canlı EPİAŞ"
        except Exception as exc:  # noqa: BLE001 - UI should fall back instead of crashing.
            st.error(f"Canlı veri güncellemesi tamamlanamadı: {type(exc).__name__}: {exc}")
            data = load_snapshot_dashboard_data()
            data_source_label = "Snapshot"
elif live_mode:
    st.warning("Canlı veri için Streamlit Secrets içine EPIAS_USERNAME/EPIAS_PASSWORD veya EPIAS_TGT eklenmeli.")
    data = load_snapshot_dashboard_data()
    data_source_label = "Snapshot"
else:
    data = load_snapshot_dashboard_data()
    data_source_label = "Snapshot"

summary = data.get("summary", {})
metrics = pd.DataFrame(data.get("metrics", []))
latest = pd.DataFrame(data.get("latest_available", []))
missing = pd.DataFrame(data.get("missing_report", []))

st.caption(
    "D+2 PTF tahmini paneli. Canlı modda EPİAŞ verisi çekilir; model metrikleri son eğitilmiş snapshot'tan gösterilir."
)

with st.sidebar:
    st.divider()
    st.caption(f"Veri kaynağı: {data_source_label}")
    st.caption(f"Güncelleme zamanı: {data.get('generated_at', '-')}")

metric_cols = st.columns(5)
metric_cols[0].metric("Final Satır", f"{summary.get('final_rows', 0):,}")
metric_cols[1].metric("Başlangıç", str(summary.get("final_start", "-"))[:10])
metric_cols[2].metric("Bitiş", str(summary.get("final_end", "-"))[:10])
metric_cols[3].metric("PTF Ortalama", format_number(summary.get("ptf_mean")))

after_row = metrics.loc[metrics.get("scenario", pd.Series(dtype=str)).eq("after_14")]
after_r2 = after_row["R2"].iloc[0] if not after_row.empty and "R2" in after_row else None
metric_cols[4].metric("14:00 Sonrası R²", format_number(after_r2, 3))

live_cols = st.columns(4)
live_cols[0].metric("Çekilmeye Çalışılan", f"{summary.get('attempted_features', 0):,}")
live_cols[1].metric("Başarılı Veri", f"{summary.get('successful_features', 0):,}")
live_cols[2].metric("Başarısız Veri", f"{summary.get('failed_features', 0):,}")
live_cols[3].metric("USD/GRF Yayılım", "Tamam" if summary.get("usd_daily_broadcast") or summary.get("grf_daily_broadcast") else "-")

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
            display = localized_dataframe(metrics, ["scenario", "feature_count", "MAE", "RMSE", "SMAPE", "R2"])
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
        st.dataframe(
            localized_dataframe(rows.tail(96), ["target_datetime", "actual_ptf", "predicted_ptf", "absolute_error"]),
            width="stretch",
            hide_index=True,
        )
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
            st.dataframe(localized_dataframe(latest, columns), width="stretch", hide_index=True)
        else:
            st.info("Son yayın zamanı raporu bulunamadı.")
    with right:
        st.subheader("Eksik Veri Raporu")
        if not missing.empty:
            columns = ["column", "missing_count_before", "missing_count_after"]
            st.dataframe(localized_dataframe(missing, columns), width="stretch", hide_index=True)
        else:
            st.info("Eksik veri raporu bulunamadı.")
