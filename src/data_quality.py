from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["datetime", "date", "hour", "ptf"]


def clean_ptf_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("EPİAŞ API boş veri döndürdü.")

    df = raw_df.copy()
    validate_source_columns(df)

    df["datetime"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
    )
    df["ptf"] = pd.to_numeric(df["price"], errors="coerce")

    if "hour" in df.columns:
        df["hour"] = df["hour"].map(parse_hour)
    else:
        df["hour"] = df["datetime"].dt.hour

    if df["datetime"].dt.hour.nunique(dropna=True) == 1 and df["hour"].nunique(dropna=True) > 1:
        df["datetime"] = df["datetime"].dt.normalize() + pd.to_timedelta(df["hour"], unit="h")

    df["date"] = df["datetime"].dt.date.astype(str)
    df = df[REQUIRED_COLUMNS]
    df = df.dropna(subset=["datetime", "ptf", "hour"])
    df["hour"] = df["hour"].astype(int)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.reset_index(drop=True)

    validate_clean_data(df)
    return df


def validate_source_columns(df: pd.DataFrame) -> None:
    missing = [column for column in ("date", "price") if column not in df.columns]
    if missing:
        raise ValueError(
            "EPİAŞ yanıtında beklenen kolonlar yok: "
            f"{missing}. Gelen kolonlar: {list(df.columns)}"
        )


def parse_hour(value: object) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    if ":" in text:
        return int(text.split(":", maxsplit=1)[0])

    return int(float(text))


def validate_clean_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Temizlik sonrası veri kalmadı.")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Temiz veride eksik kolon var: {missing}")

    if df["datetime"].isna().any():
        raise ValueError("datetime kolonunda boş değer var.")

    if df["ptf"].isna().any():
        raise ValueError("ptf kolonunda boş değer var.")

    invalid_hours = ~df["hour"].between(0, 23)
    if invalid_hours.any():
        bad_values = df.loc[invalid_hours, "hour"].unique().tolist()
        raise ValueError(f"hour kolonu 0-23 dışında değer içeriyor: {bad_values}")


def save_ptf_data(df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)


def print_data_summary(df: pd.DataFrame, csv_path: Path, xlsx_path: Path) -> None:
    print("\nVeri ozeti")
    print("-" * 40)
    print(f"Satir sayisi       : {len(df):,}")
    print(f"Kolonlar           : {', '.join(df.columns)}")
    print(f"Tarih araligi      : {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"Eksik deger sayisi : {int(df.isna().sum().sum())}")
    print(f"Tekil saat sayisi  : {df['datetime'].nunique():,}")
    print(f"PTF min            : {df['ptf'].min():,.2f}")
    print(f"PTF ortalama       : {df['ptf'].mean():,.2f}")
    print(f"PTF max            : {df['ptf'].max():,.2f}")
    print(f"CSV kayit          : {csv_path}")
    print(f"Excel kayit        : {xlsx_path}")
    print("\nIlk 5 satir")
    print(df.head().to_string(index=False))
    print("\nSon 5 satir")
    print(df.tail().to_string(index=False))
