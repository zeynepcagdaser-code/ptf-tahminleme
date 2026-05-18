from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "final_selected_features"
PTF_INTERIM_PATH = RAW_DIR / "ptf_interim.csv"
PTF_KESINLESMIS_PATH = RAW_DIR / "ptf_kesinlesmis.csv"
LEGACY_PTF_PATH = RAW_DIR / "ptf.csv"

FINAL_COLUMNS = [
    "datetime",
    "date",
    "hour",
    "ptf",
    "ptf_kesinlesmis",
    "gop_fiyattan_bagimsiz_alis",
    "gop_fiyattan_bagimsiz_satis",
    "price_independent_buy_sell_ratio",
    "load_forecast_plan",
    "wind_generation",
    "solar_generation",
    "hydro_dam_generation",
    "real_time_consumption",
    "grf_tl",
    "unlicensed_generation_total",
    "smf",
    "usd_try",
]


def build_final_hourly_dataset() -> tuple[pd.DataFrame, dict[str, bool]]:
    interim = _read_price_series(PTF_INTERIM_PATH, None, "Kesinlesmemis PTF (I-MCP)")
    kesinlesmis = _read_price_series(PTF_KESINLESMIS_PATH, LEGACY_PTF_PATH, "Kesinlesmis PTF (MCP)")

    dataset = interim.rename(columns={"price": "ptf"}).copy()
    dataset = dataset.merge(
        kesinlesmis.rename(columns={"price": "ptf_kesinlesmis"})[
            ["datetime", "ptf_kesinlesmis"]
        ],
        on="datetime",
        how="outer",
    )
    dataset["date"] = dataset["datetime"].dt.date.astype(str)
    dataset["hour"] = dataset["datetime"].dt.hour

    flags = {
        "ratio_created": False,
        "grf_daily_broadcast": False,
        "usd_daily_broadcast": False,
        "has_interim_ptf": bool(dataset["ptf"].notna().any()),
        "has_kesinlesmis_ptf": bool(dataset["ptf_kesinlesmis"].notna().any()),
    }

    dataset = _merge_hourly(
        dataset,
        RAW_DIR / "gop_fiyattan_bagimsiz_alis.csv",
        "gop_fiyattan_bagimsiz_alis",
        ["bidVolume"],
    )
    dataset = _merge_hourly(
        dataset,
        RAW_DIR / "gop_fiyattan_bagimsiz_satis.csv",
        "gop_fiyattan_bagimsiz_satis",
        ["offerVolume"],
    )

    if {"gop_fiyattan_bagimsiz_alis", "gop_fiyattan_bagimsiz_satis"}.issubset(dataset.columns):
        sales = dataset["gop_fiyattan_bagimsiz_satis"].replace(0, np.nan)
        dataset["price_independent_buy_sell_ratio"] = dataset["gop_fiyattan_bagimsiz_alis"] / sales
        flags["ratio_created"] = True

    dataset = _merge_hourly(dataset, RAW_DIR / "load_forecast_plan.csv", "load_forecast_plan", ["lep"])

    generation_path = RAW_DIR / "realtime_generation.csv"
    if generation_path.exists():
        generation = _read_csv(generation_path)
        generation = _with_datetime(generation)
        generation = generation.rename(
            columns={
                "wind": "wind_generation",
                "sun": "solar_generation",
                "dammedHydro": "hydro_dam_generation",
            }
        )
        generation_cols = [
            col
            for col in ["datetime", "wind_generation", "solar_generation", "hydro_dam_generation"]
            if col in generation.columns
        ]
        if len(generation_cols) > 1:
            generation = generation[generation_cols].groupby("datetime", as_index=False).mean(numeric_only=True)
            dataset = dataset.merge(generation, on="datetime", how="left")

    dataset = _merge_hourly(
        dataset,
        RAW_DIR / "real_time_consumption.csv",
        "real_time_consumption",
        ["consumption"],
    )
    dataset, grf_broadcast = _merge_daily(dataset, RAW_DIR / "grf_tl.csv", "grf_tl", ["grfTl"])
    flags["grf_daily_broadcast"] = grf_broadcast

    dataset = _merge_hourly(
        dataset,
        RAW_DIR / "unlicensed_generation_total.csv",
        "unlicensed_generation_total",
        ["toplam", "total", "totalAmount", "amount", "generationAmount"],
    )
    dataset = _merge_hourly(dataset, RAW_DIR / "smf.csv", "smf", ["systemMarginalPrice"])

    dataset, usd_broadcast = _merge_daily(dataset, RAW_DIR / "usd_try.csv", "usd_try", ["usd_try"])
    flags["usd_daily_broadcast"] = usd_broadcast

    for column in FINAL_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = np.nan

    dataset = dataset[FINAL_COLUMNS].sort_values("datetime").reset_index(drop=True)
    return dataset, flags


def _read_price_series(primary_path: Path, fallback_path: Path | None, label: str) -> pd.DataFrame:
    path = primary_path
    if not path.exists() and fallback_path is not None and fallback_path.exists():
        path = fallback_path
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dosyasi bulunamadi: {primary_path}\n"
            "EPİAŞ verisi icin: python main.py --fetch"
        )

    data = pd.read_csv(path)
    data = _with_datetime(data)
    value_column = _choose_value_column(data, ["price", "mcp", "ptf", "interimMcp"])
    if value_column is None:
        raise ValueError(f"{label} dosyasinda fiyat kolonu bulunamadi: {list(data.columns)}")

    result = pd.DataFrame(
        {
            "datetime": data["datetime"],
            "price": pd.to_numeric(data[value_column], errors="coerce"),
        }
    )
    return result.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"])


def _merge_hourly(
    dataset: pd.DataFrame,
    path: Path,
    target_column: str,
    preferred_columns: Iterable[str],
) -> pd.DataFrame:
    if not path.exists():
        dataset[target_column] = np.nan
        return dataset

    data = _read_csv(path)
    data = _with_datetime(data)
    value_column = _choose_value_column(data, preferred_columns)
    if value_column is None:
        dataset[target_column] = np.nan
        return dataset

    data[target_column] = pd.to_numeric(data[value_column], errors="coerce")
    reduced = data[["datetime", target_column]].dropna(subset=["datetime"])
    reduced = reduced.groupby("datetime", as_index=False).mean(numeric_only=True)
    return dataset.merge(reduced, on="datetime", how="left")


def _merge_daily(
    dataset: pd.DataFrame,
    path: Path,
    target_column: str,
    preferred_columns: Iterable[str],
) -> tuple[pd.DataFrame, bool]:
    if not path.exists():
        dataset[target_column] = np.nan
        return dataset, False

    data = _read_csv(path)
    value_column = _choose_value_column(data, preferred_columns)
    if value_column is None:
        dataset[target_column] = np.nan
        return dataset, False

    date_column = _choose_date_column(data)
    if date_column is None:
        dataset[target_column] = np.nan
        return dataset, False

    data["date"] = _parse_date_series(data[date_column])
    data[target_column] = pd.to_numeric(data[value_column], errors="coerce")
    reduced = data[["date", target_column]].dropna(subset=["date"])
    reduced = reduced.groupby("date", as_index=False).mean(numeric_only=True)
    return dataset.merge(reduced, on="date", how="left"), True


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _with_datetime(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    date_column = _choose_date_column(data)
    if date_column is None:
        data["datetime"] = pd.NaT
        return data

    parsed = pd.to_datetime(data[date_column], errors="coerce", utc=True)
    if parsed.notna().any():
        data["datetime"] = parsed.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    else:
        data["datetime"] = pd.to_datetime(data[date_column], errors="coerce")

    if data["datetime"].isna().all() and "time" in data.columns:
        data["datetime"] = pd.to_datetime(
            data[date_column].astype(str).str.slice(0, 10) + " " + data["time"].astype(str),
            errors="coerce",
        )

    return data


def _choose_date_column(data: pd.DataFrame) -> str | None:
    for column in ("datetime", "date", "gasDay", "day"):
        if column in data.columns:
            return column
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed.dt.tz_convert("Europe/Istanbul").dt.date.astype(str)
    return pd.to_datetime(series, errors="coerce").dt.date.astype(str)


def _choose_value_column(data: pd.DataFrame, preferred_columns: Iterable[str]) -> str | None:
    for column in preferred_columns:
        if column in data.columns:
            return column

    ignored = {
        "hour",
        "time",
        "date",
        "datetime",
        "feature_name",
        "service",
        "market_type",
        "endpoint_path",
        "frequency",
    }
    numeric_candidates = [
        column
        for column in data.columns
        if column not in ignored and pd.to_numeric(data[column], errors="coerce").notna().any()
    ]
    return numeric_candidates[0] if numeric_candidates else None
