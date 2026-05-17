from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT


RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw" / "electricity_features",
    PROJECT_ROOT / "data" / "raw" / "natural_gas_features",
]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "external_features" / "standardized_selected_features.csv"


DATE_CANDIDATES = ["datetime", "date", "startDate", "effectiveDate", "deliveryDate", "period", "tarih"]
META_COLUMNS = {"feature_name", "market_type", "service", "endpoint_path", "frequency"}
NON_FEATURE_NUMERIC_COLUMNS = {"hour", "period", "year", "month", "day"}


def run_standardize_selected_features() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_dir in RAW_DIRS:
        for path in sorted(raw_dir.glob("*.csv")):
            standardized = standardize_one_file(path)
            if not standardized.empty:
                frames.append(standardized)

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["datetime", "date", "hour", "feature_name", "feature_value", "frequency", "market_type"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Standardize edilen satir: {len(result):,}")
    print(f"Standardize cikti: {OUTPUT_PATH}")
    return result


def standardize_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    base_feature = str(df.get("feature_name", pd.Series([path.stem])).iloc[0])
    market_type = str(df.get("market_type", pd.Series(["unknown"])).iloc[0])
    frequency = str(df.get("frequency", pd.Series(["unknown"])).iloc[0])

    date_col = choose_date_column(df)
    if date_col is None:
        return pd.DataFrame()

    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    if dt.isna().all():
        dt = pd.to_datetime(df[date_col], errors="coerce")

    numeric_cols = [
        col for col in df.columns
        if col not in META_COLUMNS
        and col != date_col
        and col not in NON_FEATURE_NUMERIC_COLUMNS
        and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    numeric_cols = [col for col in numeric_cols if pd.to_numeric(df[col], errors="coerce").notna().any()]
    rows: list[pd.DataFrame] = []
    for col in numeric_cols:
        value = pd.to_numeric(df[col], errors="coerce")
        feature_name = base_feature if len(numeric_cols) == 1 else f"{base_feature}__{col}"
        part = pd.DataFrame(
            {
                "datetime": dt.dt.tz_convert(None) if getattr(dt.dt, "tz", None) is not None else dt,
                "feature_name": feature_name,
                "feature_value": value,
                "frequency": frequency,
                "market_type": market_type,
            }
        )
        part = part.dropna(subset=["datetime", "feature_value"])
        part["date"] = part["datetime"].dt.date.astype(str)
        part["hour"] = part["datetime"].dt.hour
        rows.append(part)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def choose_date_column(df: pd.DataFrame) -> str | None:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        sample = pd.to_datetime(df[col], errors="coerce")
        if sample.notna().mean() > 0.5:
            return col
    return None
