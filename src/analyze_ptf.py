from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from src.config import PROJECT_ROOT


RAW_PTF_PATH = PROJECT_ROOT / "data" / "raw" / "ptf_2025_to_today.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROCESSED_DIR / "figures"
CACHE_DIR = PROCESSED_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class AnalysisPaths:
    raw_csv: Path = RAW_PTF_PATH
    processed_dir: Path = PROCESSED_DIR
    figures_dir: Path = FIGURES_DIR
    clean_csv: Path = PROCESSED_DIR / "ptf_clean.csv"
    summary_json: Path = PROCESSED_DIR / "ptf_summary.json"
    hourly_avg_csv: Path = PROCESSED_DIR / "ptf_hourly_avg.csv"
    daily_avg_csv: Path = PROCESSED_DIR / "ptf_daily_avg.csv"
    monthly_avg_csv: Path = PROCESSED_DIR / "ptf_monthly_avg.csv"


def run_ptf_analysis(paths: AnalysisPaths | None = None) -> dict[str, Any]:
    paths = paths or AnalysisPaths()
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    print("PTF analiz pipeline basladi")
    print(f"Ham veri okunuyor: {paths.raw_csv}")

    raw_df = load_ptf_data(paths.raw_csv)
    df = prepare_ptf_data(raw_df)

    duplicate_count = count_duplicate_records(df)
    missing_hours = find_missing_hours(df)
    outliers = detect_outliers_iqr(df)

    hourly_avg = calculate_hourly_average(df)
    daily_avg = calculate_daily_average(df)
    monthly_avg = calculate_monthly_average(df)

    summary = build_summary(
        df=df,
        duplicate_count=duplicate_count,
        missing_hours=missing_hours,
        outliers=outliers,
    )

    save_outputs(
        df=df,
        summary=summary,
        hourly_avg=hourly_avg,
        daily_avg=daily_avg,
        monthly_avg=monthly_avg,
        paths=paths,
    )
    save_figures(df=df, hourly_avg=hourly_avg, daily_avg=daily_avg, paths=paths)
    print_terminal_summary(summary)

    print("\nKaydedilen dosyalar")
    print("-" * 40)
    print(paths.clean_csv)
    print(paths.summary_json)
    print(paths.hourly_avg_csv)
    print(paths.daily_avg_csv)
    print(paths.monthly_avg_csv)
    print(paths.figures_dir)

    return summary


def load_ptf_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"PTF ham veri dosyasi bulunamadi: {csv_path}\n"
            "Once EPİAŞ veri cekme adimini calistirin."
        )

    return pd.read_csv(csv_path)


def prepare_ptf_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"datetime", "date", "hour", "ptf"}
    missing_columns = sorted(required_columns.difference(raw_df.columns))
    if missing_columns:
        raise ValueError(f"Eksik kolonlar: {missing_columns}")

    df = raw_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["ptf"] = pd.to_numeric(df["ptf"], errors="coerce")

    invalid_rows = df[df[["datetime", "date", "hour", "ptf"]].isna().any(axis=1)]
    if not invalid_rows.empty:
        print(f"[WARN] Gecersiz/eksik alan iceren satir sayisi: {len(invalid_rows)}")

    df = df.dropna(subset=["datetime", "date", "hour", "ptf"])
    df["hour"] = df["hour"].astype(int)
    df = df.sort_values("datetime").reset_index(drop=True)
    validate_ptf_data(df)
    return df


def validate_ptf_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Analiz edilecek veri bos.")

    invalid_hours = ~df["hour"].between(0, 23)
    if invalid_hours.any():
        bad_values = sorted(df.loc[invalid_hours, "hour"].unique().tolist())
        raise ValueError(f"Saat kolonu 0-23 disinda deger iceriyor: {bad_values}")

    if df["ptf"].isna().any():
        raise ValueError("PTF kolonunda bos deger var.")


def count_duplicate_records(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=["datetime"]).sum())


def find_missing_hours(df: pd.DataFrame) -> pd.DatetimeIndex:
    expected_range = pd.date_range(
        start=df["datetime"].min(),
        end=df["datetime"].max(),
        freq="h",
    )
    actual_range = pd.DatetimeIndex(df["datetime"].drop_duplicates())
    return expected_range.difference(actual_range)


def detect_outliers_iqr(df: pd.DataFrame) -> dict[str, Any]:
    q1 = float(df["ptf"].quantile(0.25))
    q3 = float(df["ptf"].quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = (df["ptf"] < lower_bound) | (df["ptf"] > upper_bound)

    return {
        "method": "IQR",
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "count": int(mask.sum()),
        "percentage": float(mask.mean() * 100),
    }


def calculate_hourly_average(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour", as_index=False)
        .agg(ptf_avg=("ptf", "mean"), ptf_min=("ptf", "min"), ptf_max=("ptf", "max"), count=("ptf", "size"))
        .sort_values("hour")
    )


def calculate_daily_average(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.assign(day=df["datetime"].dt.date.astype("string"))
        .groupby("day", as_index=False)
        .agg(ptf_avg=("ptf", "mean"), ptf_min=("ptf", "min"), ptf_max=("ptf", "max"), count=("ptf", "size"))
        .sort_values("day")
    )
    return daily


def calculate_monthly_average(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.assign(month=df["datetime"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(ptf_avg=("ptf", "mean"), ptf_min=("ptf", "min"), ptf_max=("ptf", "max"), count=("ptf", "size"))
        .sort_values("month")
    )
    return monthly


def build_summary(
    df: pd.DataFrame,
    duplicate_count: int,
    missing_hours: pd.DatetimeIndex,
    outliers: dict[str, Any],
) -> dict[str, Any]:
    total_records = int(len(df))

    return {
        "total_records": total_records,
        "start_datetime": df["datetime"].min().isoformat(),
        "end_datetime": df["datetime"].max().isoformat(),
        "duplicate_count": duplicate_count,
        "missing_hour_count": int(len(missing_hours)),
        "missing_hours_sample": [value.isoformat() for value in missing_hours[:50]],
        "ptf": {
            "min": float(df["ptf"].min()),
            "max": float(df["ptf"].max()),
            "mean": float(df["ptf"].mean()),
            "median": float(df["ptf"].median()),
            "std": float(df["ptf"].std()),
        },
        "outliers": outliers,
        "null_counts": {column: int(value) for column, value in df.isna().sum().items()},
    }


def save_outputs(
    df: pd.DataFrame,
    summary: dict[str, Any],
    hourly_avg: pd.DataFrame,
    daily_avg: pd.DataFrame,
    monthly_avg: pd.DataFrame,
    paths: AnalysisPaths,
) -> None:
    df.to_csv(paths.clean_csv, index=False)
    hourly_avg.to_csv(paths.hourly_avg_csv, index=False)
    daily_avg.to_csv(paths.daily_avg_csv, index=False)
    monthly_avg.to_csv(paths.monthly_avg_csv, index=False)

    with paths.summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def save_figures(
    df: pd.DataFrame,
    hourly_avg: pd.DataFrame,
    daily_avg: pd.DataFrame,
    paths: AnalysisPaths,
) -> None:
    save_ptf_line_chart(df, paths.figures_dir / "ptf_hourly_line.png")
    save_daily_average_chart(daily_avg, paths.figures_dir / "ptf_daily_avg.png")
    save_histogram(df, paths.figures_dir / "ptf_histogram.png")
    save_boxplot(df, paths.figures_dir / "ptf_boxplot.png")
    save_hourly_average_chart(hourly_avg, paths.figures_dir / "ptf_hourly_avg_by_hour.png")

    px.line(df, x="datetime", y="ptf", title="Saatlik PTF").write_html(
        paths.figures_dir / "ptf_hourly_line.html"
    )
    px.line(daily_avg, x="day", y="ptf_avg", title="Gunluk Ortalama PTF").write_html(
        paths.figures_dir / "ptf_daily_avg.html"
    )
    px.bar(hourly_avg, x="hour", y="ptf_avg", title="Saat Bazli Ortalama PTF").write_html(
        paths.figures_dir / "ptf_hourly_avg_by_hour.html"
    )


def save_ptf_line_chart(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(df["datetime"], df["ptf"], linewidth=0.8)
    plt.title("Saatlik PTF")
    plt.xlabel("Tarih")
    plt.ylabel("PTF")
    plt.grid(alpha=0.3)
    save_matplotlib_figure(output_path)


def save_daily_average_chart(daily_avg: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(pd.to_datetime(daily_avg["day"]), daily_avg["ptf_avg"], linewidth=1.2)
    plt.title("Gunluk Ortalama PTF")
    plt.xlabel("Tarih")
    plt.ylabel("Ortalama PTF")
    plt.grid(alpha=0.3)
    save_matplotlib_figure(output_path)


def save_histogram(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["ptf"], bins=50, edgecolor="black", alpha=0.8)
    plt.title("PTF Histogram")
    plt.xlabel("PTF")
    plt.ylabel("Frekans")
    plt.grid(alpha=0.3)
    save_matplotlib_figure(output_path)


def save_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.boxplot(df["ptf"], vert=True)
    plt.title("PTF Boxplot")
    plt.ylabel("PTF")
    plt.grid(axis="y", alpha=0.3)
    save_matplotlib_figure(output_path)


def save_hourly_average_chart(hourly_avg: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_avg["hour"], hourly_avg["ptf_avg"])
    plt.title("Saat Bazli Ortalama PTF")
    plt.xlabel("Saat")
    plt.ylabel("Ortalama PTF")
    plt.xticks(range(24))
    plt.grid(axis="y", alpha=0.3)
    save_matplotlib_figure(output_path)


def save_matplotlib_figure(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def print_terminal_summary(summary: dict[str, Any]) -> None:
    print("\nPTF veri kalite ozeti")
    print("-" * 40)
    print(f"Toplam kayit      : {summary['total_records']:,}")
    print(f"Eksik saat sayisi : {summary['missing_hour_count']:,}")
    print(f"Duplicate sayisi  : {summary['duplicate_count']:,}")
    print(f"PTF min           : {summary['ptf']['min']:,.2f}")
    print(f"PTF max           : {summary['ptf']['max']:,.2f}")
    print(f"PTF mean          : {summary['ptf']['mean']:,.2f}")
    print(f"PTF std           : {summary['ptf']['std']:,.2f}")
    print(f"Aykiri deger      : {summary['outliers']['count']:,}")


def analysis_paths_as_dict(paths: AnalysisPaths) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(paths).items()}
