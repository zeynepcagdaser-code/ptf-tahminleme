from __future__ import annotations

"""
Spot-check EPİAŞ 5Y raw CSVs against live API responses.

Goal: answer "veriler dogru mu cekilmis" with an automated, repeatable check.
This does NOT re-download 5 years. It only queries tiny windows (1 day / 7 days / 1 month).

Outputs:
  data/processed/epias_5y_spotcheck_report.csv
"""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, get_settings
from src.fetch_epias_5y import EPIAS_5Y_FEATURES
from src.fetch_final_selected_features import FinalSelectedFeatureFetcher, FinalFeatureSpec


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_5Y_DIR = PROJECT_ROOT / "data" / "raw" / "epias_5y"
REPORT_CSV = PROCESSED_DIR / "epias_5y_spotcheck_report.csv"
REPORT_JSON = PROCESSED_DIR / "epias_5y_spotcheck_report.json"


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _find_dt_col(df: pd.DataFrame) -> str | None:
    if "datetime" in df.columns:
        return "datetime"
    for c in df.columns:
        lc = c.lower()
        if lc in ("date", "tarih", "timestamp") or "datetime" in lc or "date" in lc:
            return c
    return None


def _pick_sample_windows(freq: str) -> tuple[date, date]:
    # Default windows (used when local file is missing).
    if freq == "hourly":
        d = date(2022, 6, 15)
        return d, d
    if freq == "daily":
        return date(2022, 6, 1), date(2022, 6, 7)
    return date(2022, 6, 1), date(2022, 6, 30)


def _pick_sample_windows_from_local(path: Path, freq: str) -> tuple[date, date] | None:
    if not path.exists() or path.stat().st_size < 100:
        return None
    try:
        df = pd.read_csv(path, usecols=["datetime"])
        dt = pd.to_datetime(df["datetime"], errors="coerce").dropna()
        if dt.empty:
            return None
        # Pick a midpoint date to avoid edge publication quirks.
        mid = dt.iloc[len(dt) // 2].date()
        if freq == "hourly":
            return mid, mid
        if freq == "daily":
            end = mid + pd.Timedelta(days=6)
            return mid, end.date()
        # monthly: take entire month window
        month_start = pd.Timestamp(mid).to_period("M").start_time.date()
        month_end = pd.Timestamp(mid).to_period("M").end_time.date()
        return month_start, month_end
    except Exception:
        return None


def _local_slice(path: Path, start: date, end: date, freq: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    dt_col = _find_dt_col(df)
    if dt_col is None:
        return pd.DataFrame()
    dt = _safe_to_datetime(df[dt_col])
    df = df.assign(datetime=dt).dropna(subset=["datetime"])

    if freq == "hourly":
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        return df[(df["datetime"] >= start_dt) & (df["datetime"] <= end_dt)].copy()
    if freq == "daily":
        s = pd.Timestamp(start).normalize()
        e = pd.Timestamp(end).normalize()
        d = df["datetime"].dt.tz_localize(None, nonexistent="shift_forward", ambiguous="NaT") if hasattr(df["datetime"].dt, "tz_localize") else df["datetime"]
        # normalize to date
        dd = pd.to_datetime(d, errors="coerce").dt.normalize()
        df = df.assign(day=dd)
        return df[(df["day"] >= s) & (df["day"] <= e)].copy()

    # monthly
    s = pd.Timestamp(start).to_period("M").start_time
    e = pd.Timestamp(end).to_period("M").start_time
    ym = df["datetime"].dt.to_period("M").dt.start_time
    df = df.assign(month_start=ym)
    return df[(df["month_start"] >= s) & (df["month_start"] <= e)].copy()


def _api_fetch(fetcher: FinalSelectedFeatureFetcher, spec: FinalFeatureSpec, start: date, end: date) -> pd.DataFrame:
    # Access fetcher's internal request method (keeps payload format consistent).
    items: list[dict[str, Any]] = fetcher._request_epias_chunk(spec, start, end)  # type: ignore[attr-defined]
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    # Normalize date/datetime column to "datetime" where possible
    dt_col = _find_dt_col(df)
    if dt_col is not None and dt_col != "datetime":
        df = df.rename(columns={dt_col: "datetime"})
    if "datetime" in df.columns:
        df["datetime"] = _safe_to_datetime(df["datetime"])
    return df


def _choose_numeric_value_col(df: pd.DataFrame) -> str | None:
    # Prefer obvious names first
    preferred = [
        "price",
        "mcp",
        "ptf",
        "interimMcp",
        "systemMarginalPrice",
        "consumption",
        "lep",
        "bidVolume",
        "offerVolume",
        "grfTl",
        "usd_try",
        "total",
        "totalAmount",
        "amount",
        "quantity",
        "value",
        "volume",
        "weightedAveragePrice",
        "matchedQuantity",
        "clearingQuantity",
    ]
    for c in preferred:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return c
    # Otherwise pick the numeric column with most non-nulls.
    best = None
    best_nn = 0
    for c in df.columns:
        if c in ("datetime", "date", "day", "month_start"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nn = int(s.notna().sum())
        if nn > best_nn:
            best_nn = nn
            best = c
    return best


def run_spot_check_epias_5y_vs_api(*, max_features: int | None = None) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    fetcher = FinalSelectedFeatureFetcher(settings.start_date, settings.end_date)

    specs = list(EPIAS_5Y_FEATURES)
    if max_features is not None:
        specs = specs[: max_features]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        local_path = RAW_5Y_DIR / spec.output_file
        picked = _pick_sample_windows_from_local(local_path, spec.frequency)
        if picked is None:
            start, end = _pick_sample_windows(spec.frequency)
        else:
            start, end = picked  # type: ignore[misc]
        row: dict[str, Any] = {
            "feature_name": spec.feature_name,
            "frequency": spec.frequency,
            "endpoint_path": spec.endpoint_path,
            "sample_start": start.isoformat(),
            "sample_end": end.isoformat(),
            "local_file": str(local_path),
        }
        try:
            local_df = _local_slice(local_path, start, end, spec.frequency)
            if not local_path.exists():
                row["status"] = "local_missing"
                row["verdict"] = "local_missing"
                row["local_rows"] = 0
                row["api_rows"] = ""
                rows.append(row)
                continue

            try:
                api_df = _api_fetch(fetcher, spec, start, end)
            except RuntimeError as exc:
                msg = str(exc)
                # Business \"no data\" responses are expected for some endpoints in some periods.
                if "SEF1181" in msg or "veri bulunmamaktadır" in msg:
                    row["status"] = "api_no_data"
                    row["verdict"] = "api_no_data"
                    row["local_rows"] = int(len(local_df))
                    row["api_rows"] = 0
                    rows.append(row)
                    continue
                raise

            row["local_rows"] = int(len(local_df))
            row["api_rows"] = int(len(api_df))

            # Compare a single numeric value column by summary stats (fast + robust to schema diffs)
            api_val_col = _choose_numeric_value_col(api_df) if not api_df.empty else None
            loc_val_col = _choose_numeric_value_col(local_df) if not local_df.empty else None
            row["api_value_col"] = api_val_col or ""
            row["local_value_col"] = loc_val_col or ""

            if api_val_col and not api_df.empty:
                s = pd.to_numeric(api_df[api_val_col], errors="coerce")
                row["api_min"] = float(s.min()) if s.notna().any() else None
                row["api_max"] = float(s.max()) if s.notna().any() else None
                row["api_mean"] = float(s.mean()) if s.notna().any() else None
            if loc_val_col and not local_df.empty:
                s = pd.to_numeric(local_df[loc_val_col], errors="coerce")
                row["local_min"] = float(s.min()) if s.notna().any() else None
                row["local_max"] = float(s.max()) if s.notna().any() else None
                row["local_mean"] = float(s.mean()) if s.notna().any() else None

            # Lightweight verdict
            verdict = "unknown"
            if row.get("api_rows", 0) == 0:
                verdict = "api_empty"
            elif row.get("local_rows", 0) == 0:
                verdict = "local_empty"
            else:
                # For hourly endpoints in 1-day window we expect 24-ish
                if spec.frequency == "hourly":
                    verdict = "ok" if row["api_rows"] >= 20 and row["local_rows"] >= 20 else "suspicious_rowcount"
                elif spec.frequency == "daily":
                    verdict = "ok" if row["api_rows"] >= 5 and row["local_rows"] >= 5 else "suspicious_rowcount"
                else:
                    verdict = "ok" if row["api_rows"] >= 1 and row["local_rows"] >= 1 else "suspicious_rowcount"

            row["verdict"] = verdict
            row["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_CSV, index=False)
    REPORT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    df = run_spot_check_epias_5y_vs_api()
    print(f"Saved: {REPORT_CSV}")
    print(df[['feature_name','frequency','local_rows','api_rows','verdict','status']].to_string(index=False))
