from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, TIMEZONE


FINAL_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "final_hourly_dataset.csv"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "latest_available_times_report.csv"
OUTPUT_XLSX_PATH = PROJECT_ROOT / "data" / "processed" / "latest_available_times_report.xlsx"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "latest_available_times_report.json"
FETCH_STATUS_PATH = PROJECT_ROOT / "logs" / "final_selected_features_fetch_status.csv"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    final_column: str
    raw_candidates: tuple[str, ...]
    frequency_hint: str
    availability_hint: str
    recommendation: str


@dataclass(frozen=True)
class LatestTimeReportRow:
    data_source: str
    first_timestamp: str | None
    last_timestamp: str | None
    total_records: int
    inferred_frequency: str
    missing_hour_count: int | None
    latest_available_time: str | None
    system_time: str
    delay_hours: float | None
    delay_text: str
    raw_file_used: str
    fetch_status: str
    final_column: str
    availability_class: str
    forecast_usage_recommendation: str


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "ptf_interim",
        "ptf",
        (
            "data/raw/final_selected_features/ptf_interim.csv",
            "data/raw/final_selected_features/ptf.csv",
        ),
        "hourly",
        "interim_mcp",
        "Kesinlesmemis PTF (I-MCP); tahmin kesim aninda kullanilir.",
    ),
    SourceSpec(
        "ptf_kesinlesmis",
        "ptf_kesinlesmis",
        (
            "data/raw/final_selected_features/ptf_kesinlesmis.csv",
            "data/raw/final_selected_features/ptf.csv",
            "data/raw/ptf_2025_to_today.csv",
        ),
        "hourly",
        "published_market_price",
        "Kesinlesmis PTF (MCP); 12 saatlik tahmin hedefi.",
    ),
    SourceSpec(
        "smf",
        "smf",
        ("data/raw/final_selected_features/smf.csv", "data/raw/electricity_features/smf.csv"),
        "hourly",
        "delayed_realized",
        "Ayni hedef saat icin leakage; sadece lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "real_time_consumption",
        "real_time_consumption",
        ("data/raw/final_selected_features/real_time_consumption.csv", "data/raw/electricity_features/gercek_zamanli_tuketim.csv"),
        "hourly",
        "realized_near_real_time",
        "Ayni hedef saat icin leakage; sadece gecmis lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "wind_generation",
        "wind_generation",
        ("data/raw/final_selected_features/realtime_generation.csv", "data/raw/electricity_features/gercek_zamanli_uretim.csv"),
        "hourly",
        "realized_near_real_time",
        "Ayni hedef saat icin leakage; sadece gecmis lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "solar_generation",
        "solar_generation",
        ("data/raw/final_selected_features/realtime_generation.csv", "data/raw/electricity_features/gercek_zamanli_uretim.csv"),
        "hourly",
        "realized_near_real_time",
        "Ayni hedef saat icin leakage; sadece gecmis lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "hydro_dam_generation",
        "hydro_dam_generation",
        ("data/raw/final_selected_features/realtime_generation.csv", "data/raw/electricity_features/gercek_zamanli_uretim.csv"),
        "hourly",
        "realized_near_real_time",
        "Ayni hedef saat icin leakage; sadece gecmis lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "unlicensed_generation_total",
        "unlicensed_generation_total",
        ("data/raw/final_selected_features/unlicensed_generation_total.csv", "data/raw/electricity_features/lisanssiz_uretim_miktari.csv"),
        "hourly",
        "delayed_realized",
        "Ciddi gecikmeli olabilir; hedef saat icin kullanilmaz, sadece gecmis lag/rolling olarak kullanilmali.",
    ),
    SourceSpec(
        "load_forecast_plan",
        "load_forecast_plan",
        ("data/raw/final_selected_features/load_forecast_plan.csv", "data/raw/electricity_features/yuk_tahmin_plani.csv"),
        "hourly",
        "forecast_plan",
        "Tahmin aninda bilinebilir kabul edilebilir; ileri tahmin feature'i olarak kullanilabilir.",
    ),
    SourceSpec(
        "grf_tl",
        "grf_tl",
        ("data/raw/final_selected_features/grf_tl.csv", "data/raw/natural_gas_features/spot_gaz_referans_fiyati.csv"),
        "daily",
        "daily_price",
        "Gunluk veri; yayinlanmis son deger ileriye forward-fill edilebilir.",
    ),
    SourceSpec(
        "usd_try",
        "usd_try",
        ("data/raw/final_selected_features/usd_try.csv",),
        "daily",
        "daily_external_fx",
        "Gunluk veri; yayinlanmis son deger ileriye forward-fill edilebilir.",
    ),
    SourceSpec(
        "gop_fiyattan_bagimsiz_alis",
        "gop_fiyattan_bagimsiz_alis",
        ("data/raw/final_selected_features/gop_fiyattan_bagimsiz_alis.csv", "data/raw/electricity_features/gop_fiyattan_bagimsiz_alis.csv"),
        "hourly",
        "market_order_data",
        "Hedef saat icin kesin bilinirligi ayrica dogrulanmali; guvenli kullanim icin lag olarak kullan.",
    ),
    SourceSpec(
        "gop_fiyattan_bagimsiz_satis",
        "gop_fiyattan_bagimsiz_satis",
        ("data/raw/final_selected_features/gop_fiyattan_bagimsiz_satis.csv", "data/raw/electricity_features/gop_fiyattan_bagimsiz_satis.csv"),
        "hourly",
        "market_order_data",
        "Hedef saat icin kesin bilinirligi ayrica dogrulanmali; guvenli kullanim icin lag olarak kullan.",
    ),
)


def run_latest_available_times_check() -> pd.DataFrame:
    final_dataset = _read_final_dataset()
    fetch_status_map = _read_fetch_status_map()
    system_time = pd.Timestamp.now(tz=TIMEZONE).floor("min").tz_localize(None)

    rows: list[LatestTimeReportRow] = []
    for source in SOURCES:
        raw_data, raw_path = _read_first_existing_raw(source.raw_candidates)
        if raw_data is not None:
            timestamps = _extract_timestamps(raw_data)
            total_records = len(raw_data)
        else:
            timestamps = _timestamps_from_final(final_dataset, source.final_column)
            raw_path = "final_hourly_dataset.csv"
            total_records = int(final_dataset[source.final_column].notna().sum()) if source.final_column in final_dataset else 0

        timestamps = timestamps.dropna().sort_values()
        first_timestamp = timestamps.min() if not timestamps.empty else pd.NaT
        last_timestamp = timestamps.max() if not timestamps.empty else pd.NaT
        frequency = _infer_frequency(timestamps, source.frequency_hint)
        missing_hour_count = _missing_hour_count(timestamps, frequency)
        delay_hours = _delay_hours(system_time, last_timestamp)

        rows.append(
            LatestTimeReportRow(
                data_source=source.name,
                first_timestamp=_format_timestamp(first_timestamp),
                last_timestamp=_format_timestamp(last_timestamp),
                total_records=int(total_records),
                inferred_frequency=frequency,
                missing_hour_count=missing_hour_count,
                latest_available_time=_format_timestamp(last_timestamp),
                system_time=_format_timestamp(system_time),
                delay_hours=delay_hours,
                delay_text=_format_delay(delay_hours),
                raw_file_used=raw_path,
                fetch_status=fetch_status_map.get(source.name, "unknown"),
                final_column=source.final_column,
                availability_class=source.availability_hint,
                forecast_usage_recommendation=source.recommendation,
            )
        )

    report = pd.DataFrame([asdict(row) for row in rows])
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_CSV_PATH, index=False)
    report.to_excel(OUTPUT_XLSX_PATH, index=False)
    OUTPUT_JSON_PATH.write_text(report.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    return report


def _read_fetch_status_map() -> dict[str, str]:
    if not FETCH_STATUS_PATH.exists():
        return {}
    status = pd.read_csv(FETCH_STATUS_PATH)
    if not {"feature_name", "success", "source"}.issubset(status.columns):
        return {}
    result = {}
    for _, row in status.iterrows():
        success = "success" if bool(row["success"]) else "failed"
        result[str(row["feature_name"])] = f"{success}:{row['source']}"
    if "realtime_generation" in result:
        result.setdefault("wind_generation", result["realtime_generation"])
        result.setdefault("solar_generation", result["realtime_generation"])
        result.setdefault("hydro_dam_generation", result["realtime_generation"])
    return result


def _read_final_dataset() -> pd.DataFrame:
    if not FINAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Final dataset bulunamadi: {FINAL_DATASET_PATH}")
    data = pd.read_csv(FINAL_DATASET_PATH)
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    return data


def _read_first_existing_raw(candidates: tuple[str, ...]) -> tuple[pd.DataFrame | None, str]:
    for relative_path in candidates:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            try:
                return pd.read_csv(path), relative_path
            except Exception:
                continue
    return None, ""


def _extract_timestamps(data: pd.DataFrame) -> pd.Series:
    for column in ("datetime", "date", "gasDay", "day"):
        if column in data.columns:
            values = data[column].dropna().astype(str)
            if not values.empty and values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
                return pd.to_datetime(data[column], errors="coerce")
            parsed = pd.to_datetime(data[column], errors="coerce", utc=True)
            if parsed.notna().any():
                return parsed.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
            return pd.to_datetime(data[column], errors="coerce")
    return pd.Series(dtype="datetime64[ns]")


def _timestamps_from_final(final_dataset: pd.DataFrame, column: str) -> pd.Series:
    if column not in final_dataset.columns:
        return pd.Series(dtype="datetime64[ns]")
    return final_dataset.loc[final_dataset[column].notna(), "datetime"]


def _infer_frequency(timestamps: pd.Series, hint: str) -> str:
    unique_times = timestamps.dropna().drop_duplicates().sort_values()
    if len(unique_times) < 2:
        return hint

    median_delta = unique_times.diff().dropna().median()
    if pd.isna(median_delta):
        return hint
    hours = median_delta.total_seconds() / 3600
    if hours <= 1.5:
        return "hourly"
    if 20 <= hours <= 30:
        return "daily"
    if 650 <= hours <= 800:
        return "monthly"
    return f"irregular_median_{hours:.2f}_hours"


def _missing_hour_count(timestamps: pd.Series, frequency: str) -> int | None:
    unique_times = timestamps.dropna().drop_duplicates().sort_values()
    if unique_times.empty:
        return None
    if frequency == "hourly":
        expected = pd.date_range(unique_times.min(), unique_times.max(), freq="h")
        return int(len(expected.difference(pd.DatetimeIndex(unique_times))))
    if frequency == "daily":
        dates = pd.DatetimeIndex(unique_times.dt.normalize().drop_duplicates().sort_values())
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        return int(len(expected.difference(dates)))
    return None


def _delay_hours(system_time: pd.Timestamp, last_timestamp: Any) -> float | None:
    if pd.isna(last_timestamp):
        return None
    return round((system_time - last_timestamp).total_seconds() / 3600, 4)


def _format_delay(delay_hours: float | None) -> str:
    if delay_hours is None:
        return "bilinmiyor"
    minutes = int(round(delay_hours * 60))
    if abs(minutes) < 60:
        return f"{minutes} dakika"
    hours, remainder = divmod(abs(minutes), 60)
    sign = "-" if minutes < 0 else ""
    if remainder == 0:
        return f"{sign}{hours} saat"
    return f"{sign}{hours} saat {remainder} dakika"


def _format_timestamp(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
