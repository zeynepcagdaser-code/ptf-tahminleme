from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.dl_5y_config import HOURS_5Y, RAW_5Y_DIR, hour_index_5y


def _parse_datetime_column(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    if "date" in df.columns and "time" in df.columns:
        base = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if base.notna().any():
            base = base.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        else:
            base = pd.to_datetime(df["date"], errors="coerce")
        t = df["time"].astype(str).str.slice(0, 8)
        return pd.to_datetime(base.dt.date.astype(str) + " " + t, errors="coerce")
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        return pd.to_datetime(df["date"], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def _parse_idm_contract_datetime(series: pd.Series) -> pd.Series:
    def one(name: object) -> pd.Timestamp | pd.NaT:
        s = str(name)
        m = re.match(r"^P[HB](\d{2})(\d{2})(\d{2})(\d{2})", s)
        if not m:
            return pd.NaT
        yy, mo, dd, hh = m.groups()
        return pd.Timestamp(f"20{yy}-{mo}-{dd} {int(hh):02d}:00:00")

    return series.map(one)


def normalize_idm_trade_value(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "kontratAdi" in work.columns:
        work["datetime"] = _parse_idm_contract_datetime(work["kontratAdi"])
    elif "contractName" in work.columns:
        work["datetime"] = _parse_idm_contract_datetime(work["contractName"])
    else:
        work["datetime"] = _parse_datetime_column(work)

    val_col = next((c for c in ("tradingVolume", "tradeValue", "volume", "total") if c in work.columns), None)
    if val_col is None:
        return work

    work[val_col] = pd.to_numeric(work[val_col], errors="coerce")
    hourly = (
        work.dropna(subset=["datetime"])
        .groupby("datetime", as_index=False)[val_col]
        .sum()
        .rename(columns={val_col: "tradingVolume"})
    )
    return hourly


def normalize_dam_supply_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Fiyat kademeli GOP arz-talep → saatlik tek satır (o günün ortalama arz/talep)."""
    work = df.copy()
    work["datetime"] = _parse_datetime_column(work)
    for col in ("supply", "demand", "price"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "datetime" not in work.columns:
        return work
    agg = work.dropna(subset=["datetime"]).groupby("datetime", as_index=False).agg(
        {c: "mean" for c in ("supply", "demand", "price") if c in work.columns}
    )
    return agg


def reindex_to_hourly_grid(df: pd.DataFrame, *, datetime_col: str = "datetime") -> pd.DataFrame:
    idx = hour_index_5y()
    grid = pd.DataFrame({datetime_col: idx.tz_localize(None) if idx.tz is not None else idx})
    if datetime_col not in df.columns:
        return grid
    work = df.copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    if work[datetime_col].dt.tz is not None:
        work[datetime_col] = work[datetime_col].dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    work = work.dropna(subset=[datetime_col])
    meta = [c for c in work.columns if c not in (datetime_col,) and not pd.api.types.is_numeric_dtype(work[c])]
    num = [c for c in work.columns if c != datetime_col and pd.api.types.is_numeric_dtype(work[c])]
    if num:
        work = work.groupby(datetime_col, as_index=False)[num].mean()
    out = grid.merge(work, on=datetime_col, how="left")
    return out


def normalize_raw_csv(path: Path, feature_name: str, *, frequency: str = "hourly") -> int:
    if not path.exists() or path.stat().st_size < 50:
        return 0
    df = pd.read_csv(path)
    if frequency not in ("hourly",):
        # Aylık/günlük seriler saatlik 43.800 ızgaraya zorlanmaz
        return len(df)

    if feature_name == "idm_trade_value":
        df = normalize_idm_trade_value(df)
    elif feature_name == "dam_supply_demand":
        df = normalize_dam_supply_demand(df)
    elif feature_name == "res_generation_forecast":
        df = df.copy()
        df["datetime"] = _parse_datetime_column(df).dt.floor("h")
        num = [c for c in df.columns if c not in ("datetime", "date", "time", "feature_name", "service", "endpoint_path", "frequency")]
        num = [c for c in num if pd.api.types.is_numeric_dtype(df[c])]
        df = df.dropna(subset=["datetime"]).groupby("datetime", as_index=False)[num].mean()
    else:
        df = df.copy()
        df["datetime"] = _parse_datetime_column(df)

    out = reindex_to_hourly_grid(df)
    out.to_csv(path, index=False)
    return len(out)


def normalize_all_epias_5y_raw(feature_names: dict[str, str] | None = None) -> dict[str, int]:
    from src.fetch_epias_5y import EPIAS_5Y_FEATURES

    rows: dict[str, int] = {}
    specs = {s.feature_name: s.frequency for s in EPIAS_5Y_FEATURES}
    if feature_names:
        specs = {k: feature_names.get(k, "hourly") for k in feature_names}

    for spec in EPIAS_5Y_FEATURES:
        path = RAW_5Y_DIR / spec.output_file
        n = normalize_raw_csv(path, spec.feature_name, frequency=spec.frequency)
        rows[spec.feature_name] = n
    return rows
