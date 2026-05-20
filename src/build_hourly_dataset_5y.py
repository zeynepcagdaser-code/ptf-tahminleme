from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.build_final_hourly_dataset import (
    _choose_date_column,
    _choose_value_column,
    _merge_daily,
    _merge_hourly,
    _parse_date_series,
    _read_csv,
    _read_price_series,
    _with_datetime,
)
from src.config import PROJECT_ROOT
from src.dl_5y_config import (
    FEATURE_SUMMARY_PATH,
    HOURLY_5Y_PATH,
    PRICE_COLUMN_5Y,
    QUALITY_REPORT_PATH,
    RAW_5Y_DIR,
    START_DATE_5Y,
    end_date_5y,
)

FALLBACK_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "final_selected_features"
_USE_FALLBACK_RAW = False


def enable_fallback_raw_dir() -> None:
    global _USE_FALLBACK_RAW
    _USE_FALLBACK_RAW = True


def _resolve_raw_file(filename: str) -> Path:
    """Önce epias_5y, yoksa final_selected_features (API tekrar çekilmez)."""
    primary = RAW_5Y_DIR / filename
    if primary.exists() and primary.stat().st_size > 100:
        return primary
    fallback = FALLBACK_RAW_DIR / filename
    if fallback.exists() and fallback.stat().st_size > 100:
        return fallback
    if filename == "ptf_kesinlesmis.csv":
        legacy = FALLBACK_RAW_DIR / "ptf.csv"
        if legacy.exists():
            return legacy
    return primary


def _active_raw_dir() -> Path:
    """Geriye uyumluluk: tek dizin yerine dosya bazlı çözümleme tercih edilir."""
    if _USE_FALLBACK_RAW and not any(RAW_5Y_DIR.glob("*.csv")):
        if FALLBACK_RAW_DIR.exists():
            print(f"[5Y] epias_5y bos — dosya bazli fallback aktif")
    return RAW_5Y_DIR
from src.turkey_calendar_features import attach_turkey_calendar


GENERATION_MAP = {
    "naturalGas": "natural_gas_generation",
    "importCoal": "imported_coal_generation",
    "lignite": "lignite_generation",
    "blackCoal": "coal_generation",
    "dammedHydro": "hydro_dam_generation",
    "river": "run_of_river_generation",
    "wind": "wind_generation",
    "sun": "solar_generation",
    "geothermal": "geothermal_generation",
    "biomass": "biomass_generation",
    "total": "total_generation",
    "importExport": "net_import_export",
}

CRITICAL_COLUMNS = [
    PRICE_COLUMN_5Y,
    "load_forecast_plan",
    "wind_generation",
    "solar_generation",
]

INTERPOLATE_LIMITS = {
    PRICE_COLUMN_5Y: 3,
    "ptf_interim": 3,
    "smf": 6,
    "load_forecast_plan": 6,
    "real_time_consumption": 6,
    "wind_generation": 6,
    "solar_generation": 6,
    "total_generation": 6,
}


def build_hourly_dataset_5y() -> tuple[pd.DataFrame, dict[str, Any]]:
    RAW_5Y_DIR.mkdir(parents=True, exist_ok=True)
    HOURLY_5Y_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset, flags = _build_merged_panel()
    dataset = _engineer_features(dataset)
    dataset, quality = _finalize_hourly_index(dataset)

    dataset.to_csv(HOURLY_5Y_PATH, index=False)
    _write_feature_summary(dataset)
    quality["build_flags"] = flags
    quality["output_path"] = str(HOURLY_5Y_PATH)
    QUALITY_REPORT_PATH.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset, quality


def _build_merged_panel() -> tuple[pd.DataFrame, dict[str, bool]]:
    _active_raw_dir()
    interim_path = _resolve_raw_file("ptf_interim.csv")
    kesin_path = _resolve_raw_file("ptf_kesinlesmis.csv")

    interim = _safe_price_series(interim_path, "ptf_interim")
    kesin = _safe_price_series(kesin_path, "ptf_kesinlesmis")

    if interim.empty and kesin.empty:
        raise FileNotFoundError(
            f"5y PTF verisi yok. Once: python scripts/run_5y_dl_pipeline.py (veya --fetch-only)\n{RAW_5Y_DIR}"
        )

    if not interim.empty:
        dataset = interim.rename(columns={"price": "ptf_interim"})[["datetime", "ptf_interim"]]
    else:
        dataset = pd.DataFrame(columns=["datetime", "ptf_interim"])

    if not kesin.empty:
        k = kesin.rename(columns={"price": "ptf_kesinlesmis"})[["datetime", "ptf_kesinlesmis"]]
        dataset = dataset.merge(k, on="datetime", how="outer")

    if dataset.empty and not kesin.empty:
        dataset = kesin.rename(columns={"price": "ptf_kesinlesmis"})[["datetime", "ptf_kesinlesmis"]]

    dataset[PRICE_COLUMN_5Y] = dataset.get("ptf_kesinlesmis", pd.Series(dtype=float)).fillna(dataset.get("ptf_interim"))
    dataset["date"] = pd.to_datetime(dataset["datetime"]).dt.date.astype(str)
    dataset["hour"] = pd.to_datetime(dataset["datetime"]).dt.hour

    flags = {
        "has_interim": bool(dataset.get("ptf_interim", pd.Series()).notna().any()),
        "has_kesin": bool(dataset.get("ptf_kesinlesmis", pd.Series()).notna().any()),
    }

    dataset = _merge_hourly(dataset, _resolve_raw_file("load_forecast_plan.csv"), "load_forecast_plan", ["lep"])
    dataset = _merge_hourly(dataset, _resolve_raw_file("real_time_consumption.csv"), "real_time_consumption", ["consumption"])
    dataset = _merge_generation_sources(dataset, _resolve_raw_file("realtime_generation.csv"))
    dataset = _merge_hourly(dataset, _resolve_raw_file("smf.csv"), "smf", ["systemMarginalPrice"])
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("system_direction.csv"), "system_direction", ["systemDirection", "direction"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("res_generation_forecast.csv"), "res_forecast", ["forecast", "generation", "total"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("generation_forecast.csv"), "ges_forecast", ["sun", "solar", "forecast"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("yekdem_realtime.csv"), "yekdem_generation", ["total", "generationAmount"]
    )
    dataset = _merge_hourly(
        dataset,
        _resolve_raw_file("unlicensed_generation_total.csv"),
        "unlicensed_generation_total",
        ["toplam", "total", "totalAmount"],
    )
    dataset = _merge_hourly(
        dataset,
        _resolve_raw_file("gop_fiyattan_bagimsiz_alis.csv"),
        "gop_fiyattan_bagimsiz_alis",
        ["bidVolume"],
    )
    dataset = _merge_hourly(
        dataset,
        _resolve_raw_file("gop_fiyattan_bagimsiz_satis.csv"),
        "gop_fiyattan_bagimsiz_satis",
        ["offerVolume"],
    )
    if {"gop_fiyattan_bagimsiz_alis", "gop_fiyattan_bagimsiz_satis"}.issubset(dataset.columns):
        sales = dataset["gop_fiyattan_bagimsiz_satis"].replace(0, np.nan)
        dataset["price_independent_buy_sell_ratio"] = dataset["gop_fiyattan_bagimsiz_alis"] / sales
        flags["ratio_created"] = True

    dataset, _ = _merge_daily(dataset, _resolve_raw_file("grf_tl.csv"), "grf_tl", ["grfTl"])
    dataset, _ = _merge_daily(dataset, _resolve_raw_file("usd_try.csv"), "usd_try", ["usd_try"])

    dataset = _merge_hourly(
        dataset, _resolve_raw_file("dam_supply_demand.csv"), "dam_supply_demand", ["supply", "demand", "total"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("dam_trade_volume.csv"), "dam_trade_volume", ["tradeVolume", "volume"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("dam_clearing_quantity.csv"), "dam_clearing_quantity", ["clearingQuantity", "quantity"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("bpm_order_summary_up.csv"), "bpm_order_summary_up", ["upAmount", "amount"]
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("bpm_order_summary_down.csv"), "bpm_order_summary_down", ["downAmount", "amount"]
    )
    dataset = _merge_hourly(
        dataset,
        _resolve_raw_file("idm_weighted_average_price.csv"),
        "idm_weighted_average_price",
        ["weightedAveragePrice", "price"],
    )
    dataset = _merge_hourly(
        dataset, _resolve_raw_file("idm_trade_value.csv"), "idm_trade_value", ["tradeValue", "value"]
    )
    dataset = _merge_monthly(
        dataset, _resolve_raw_file("imbalance_quantity.csv"), "imbalance_quantity", ["imbalanceQuantity", "quantity"]
    )
    dataset = _merge_monthly(
        dataset, _resolve_raw_file("imbalance_amount.csv"), "imbalance_amount", ["imbalanceAmount", "amount"]
    )
    dataset = _merge_monthly(
        dataset, _resolve_raw_file("yek_generation_cost.csv"), "yek_generation_cost", ["cost", "amount"]
    )
    dataset = _merge_monthly(
        dataset, _resolve_raw_file("yek_portfolio_income.csv"), "yek_portfolio_income", ["income", "amount"]
    )

    return dataset.sort_values("datetime").reset_index(drop=True), flags


def _merge_monthly(
    dataset: pd.DataFrame,
    path: Path,
    target_column: str,
    preferred_columns: list[str],
) -> pd.DataFrame:
    if not path.exists():
        dataset[target_column] = np.nan
        return dataset
    data = _read_csv(path)
    date_column = _choose_date_column(data)
    if date_column is None:
        dataset[target_column] = np.nan
        return dataset
    value_column = _choose_value_column(data, preferred_columns)
    if value_column is None:
        dataset[target_column] = np.nan
        return dataset
    data["year_month"] = pd.to_datetime(data[date_column], errors="coerce").dt.to_period("M").astype(str)
    data[target_column] = pd.to_numeric(data[value_column], errors="coerce")
    reduced = data[["year_month", target_column]].dropna(subset=["year_month"]).groupby("year_month", as_index=False).mean(
        numeric_only=True
    )
    out = dataset.copy()
    out["year_month"] = pd.to_datetime(out["datetime"]).dt.to_period("M").astype(str)
    out = out.merge(reduced, on="year_month", how="left")
    out = out.drop(columns=["year_month"])
    return out


def _merge_generation_sources(dataset: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return dataset
    gen = _with_datetime(_read_csv(path))
    rename = {src: dst for src, dst in GENERATION_MAP.items() if src in gen.columns}
    gen = gen.rename(columns=rename)
    cols = ["datetime", *[c for c in rename.values() if c in gen.columns]]
    if len(cols) <= 1:
        return dataset
    gen = gen[cols].groupby("datetime", as_index=False).mean(numeric_only=True)
    return dataset.merge(gen, on="datetime", how="left")


def _safe_price_series(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "price"])
    try:
        return _read_price_series(path, None, label)
    except Exception:
        return pd.DataFrame(columns=["datetime", "price"])


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = attach_turkey_calendar(df)
    price = out[PRICE_COLUMN_5Y].astype(float)

    wind = pd.to_numeric(out.get("wind_generation", 0), errors="coerce").fillna(0)
    solar = pd.to_numeric(out.get("solar_generation", 0), errors="coerce").fillna(0)
    load = pd.to_numeric(out.get("load_forecast_plan", np.nan), errors="coerce")
    total_gen = pd.to_numeric(out.get("total_generation", np.nan), errors="coerce")

    out["renewable_total"] = wind + solar
    out["net_load"] = load - out["renewable_total"]
    out["renewable_ratio"] = out["renewable_total"] / total_gen.replace(0, np.nan)
    out["thermal_generation"] = (
        pd.to_numeric(out.get("natural_gas_generation", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("imported_coal_generation", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("lignite_generation", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("coal_generation", 0), errors="coerce").fillna(0)
    )
    out["hydro_total"] = (
        pd.to_numeric(out.get("hydro_dam_generation", 0), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("run_of_river_generation", 0), errors="coerce").fillna(0)
    )

    for lag in (1, 24, 48, 168):
        out[f"price_lag_{lag}h"] = price.shift(lag)
    out["price_rolling_mean_24h"] = price.rolling(24, min_periods=12).mean()
    out["price_rolling_mean_168h"] = price.rolling(168, min_periods=48).mean()
    out["price_rolling_std_24h"] = price.rolling(24, min_periods=12).std()
    out["price_rolling_std_168h"] = price.rolling(168, min_periods=48).std()

    out["load_diff_1h"] = load.diff(1)
    out["load_diff_24h"] = load.diff(24)
    out["renewable_diff_1h"] = out["renewable_total"].diff(1)
    out["renewable_diff_24h"] = out["renewable_total"].diff(24)
    out["net_load_diff_1h"] = out["net_load"].diff(1)

    med168 = price.rolling(168, min_periods=48).median()
    std168 = price.rolling(168, min_periods=48).std()
    out["spike_flag"] = (price > med168 + 2 * std168).astype(int)

    out["ramp_abs_1h"] = price.diff(1).abs()
    out["ramp_abs_3h"] = price.diff(3).abs()
    out["intraday_range"] = price.rolling(24, min_periods=12).max() - price.rolling(24, min_periods=12).min()
    out["rolling_24_std"] = out["price_rolling_std_24h"]
    out["rolling_168_std"] = out["price_rolling_std_168h"]

    return out


def _finalize_hourly_index(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").copy()
    start = pd.Timestamp(START_DATE_5Y)
    end = pd.Timestamp(end_date_5y()) + pd.Timedelta(hours=23)
    full_index = pd.date_range(start, end, freq="h")

    # Ham veri kısa ise (ör. yalnızca 2025 fallback) indeksi gözlemlenen aralığa daralt
    observed = out["datetime"].dropna()
    if len(observed):
        obs_start, obs_end = observed.min(), observed.max()
        coverage = len(out) / max(len(full_index), 1)
        if coverage < 0.35:
            full_index = pd.date_range(obs_start, obs_end, freq="h")
            start, end = obs_start, obs_end

    missing_before = int(len(full_index) - out["datetime"].isin(full_index).sum())
    out = out.set_index("datetime").reindex(full_index).rename_axis("datetime").reset_index()

    nan_rates_before = {c: float(out[c].isna().mean()) for c in out.columns if c != "datetime"}

    out = out.set_index("datetime")
    for col, limit in INTERPOLATE_LIMITS.items():
        if col not in out.columns:
            continue
        out[col] = out[col].interpolate(method="time", limit=limit)

    for col in out.columns:
        if col in INTERPOLATE_LIMITS:
            continue
        if out[col].dtype.kind in "biufc" and out[col].isna().mean() < 0.35:
            out[col] = out[col].interpolate(method="time", limit=12)

    out = out.reset_index()

    spike_cols = {"spike_flag", "is_weekend", "is_holiday", "is_ramadan_bayram", "is_kurban_bayram", "month_start", "month_end"}
    for col in spike_cols:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    rows_before = len(out)
    out = out.dropna(subset=[PRICE_COLUMN_5Y])
    dropped_target = rows_before - len(out)

    nan_rates_after = {c: float(out[c].isna().mean()) for c in out.columns if c != "datetime"}

    quality = {
        "start_date": str(start.date()),
        "end_date": str(end.date()),
        "expected_hours": int(len(full_index)),
        "rows_final": int(len(out)),
        "missing_hours_before_reindex": missing_before,
        "rows_dropped_missing_target": int(dropped_target),
        "nan_rate_before_fill": nan_rates_before,
        "nan_rate_after_fill": nan_rates_after,
        "critical_column_nan_after": {c: nan_rates_after.get(c, 1.0) for c in CRITICAL_COLUMNS},
        "duplicate_datetime_removed": int(df.duplicated(subset=["datetime"]).sum()),
        "leakage_note": "Sequence features use past-only lags/rollings at cutoff; y is future 12h price.",
    }
    return out, quality


def _write_feature_summary(df: pd.DataFrame) -> None:
    rows = []
    for col in df.columns:
        if col == "datetime":
            continue
        series = df[col]
        rows.append(
            {
                "feature": col,
                "dtype": str(series.dtype),
                "nan_rate": float(series.isna().mean()),
                "mean": float(series.mean()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "std": float(series.std()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "min": float(series.min()) if pd.api.types.is_numeric_dtype(series) else np.nan,
                "max": float(series.max()) if pd.api.types.is_numeric_dtype(series) else np.nan,
            }
        )
    pd.DataFrame(rows).to_csv(FEATURE_SUMMARY_PATH, index=False)
