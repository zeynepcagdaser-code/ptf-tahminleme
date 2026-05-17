from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT


FEATURE_COLUMNS = [
    "ptf_lag_1",
    "ptf_lag_24",
    "ptf_lag_168",
    "ptf_rolling_mean_24",
    "ptf_rolling_mean_168",
    "ptf_rolling_std_24",
    "ptf_rolling_std_168",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
TARGET_COLUMN = "ptf"


@dataclass(frozen=True)
class FeatureEngineeringPaths:
    input_csv: Path = PROJECT_ROOT / "data" / "processed" / "ptf_clean.csv"
    output_csv: Path = PROJECT_ROOT / "data" / "processed" / "ptf_features.csv"


def run_feature_engineering(paths: FeatureEngineeringPaths | None = None) -> pd.DataFrame:
    paths = paths or FeatureEngineeringPaths()
    print("Feature engineering basladi")
    print(f"Girdi: {paths.input_csv}")

    df = load_clean_ptf(paths.input_csv)
    before_rows = len(df)
    features_df = create_ptf_features(df)
    deleted_nan_rows = before_rows - len(features_df)

    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(paths.output_csv, index=False)

    print("\nFeature engineering ozeti")
    print("-" * 40)
    print(f"Toplam satir sayisi     : {len(features_df):,}")
    print(f"Feature sayisi          : {len(FEATURE_COLUMNS):,}")
    print(f"Silinen NaN satir sayisi: {deleted_nan_rows:,}")
    print("Kullanilan feature listesi:")
    for feature in FEATURE_COLUMNS:
        print(f"  - {feature}")
    print(f"Cikti: {paths.output_csv}")

    return features_df


def load_clean_ptf(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Temiz PTF dosyasi bulunamadi: {input_csv}\n"
            "Once analiz adimini calistirip data/processed/ptf_clean.csv olusturun."
        )

    return pd.read_csv(input_csv)


def create_ptf_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"datetime", "ptf"}
    missing_columns = sorted(required_columns.difference(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Feature engineering icin eksik kolonlar: {missing_columns}")

    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")
    df = df.dropna(subset=["datetime", "ptf"])
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    df["ptf_lag_1"] = df["ptf"].shift(1)
    df["ptf_lag_24"] = df["ptf"].shift(24)
    df["ptf_lag_168"] = df["ptf"].shift(168)

    shifted_ptf = df["ptf"].shift(1)
    df["ptf_rolling_mean_24"] = shifted_ptf.rolling(window=24, min_periods=24).mean()
    df["ptf_rolling_mean_168"] = shifted_ptf.rolling(window=168, min_periods=168).mean()
    df["ptf_rolling_std_24"] = shifted_ptf.rolling(window=24, min_periods=24).std()
    df["ptf_rolling_std_168"] = shifted_ptf.rolling(window=168, min_periods=168).std()

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    output_columns = ["datetime", TARGET_COLUMN, *FEATURE_COLUMNS]
    df = df[output_columns]
    df = df.dropna().reset_index(drop=True)
    validate_features(df)
    return df


def validate_features(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Feature engineering sonrasi veri bos kaldi.")

    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"Eksik feature kolonlari: {missing_features}")

    if TARGET_COLUMN in FEATURE_COLUMNS:
        raise ValueError("Target olan ptf feature listesine eklenemez.")

    if df[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Feature veya target kolonlarinda NaN deger var.")
