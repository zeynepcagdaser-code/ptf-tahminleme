from __future__ import annotations

import shutil
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT
from src.dl_5y_config import RAW_5Y_DIR, START_DATE_5Y, end_date_5y
from src.epias_5y_panel import sync_fetch_live_to_app_data, write_fetch_progress
from src.fetch_final_selected_features import (
    FinalFeatureSpec,
    FinalSelectedFeatureFetcher,
    FetchResult,
    LOG_DIR,
)


FALLBACK_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "final_selected_features"
MIN_CSV_ROWS = 48  # en az ~2 gun saatlik
COVERAGE_TOLERANCE_DAYS = 45  # bu kadar gun toleransla 2020 baslangici sayilir


EPIAS_5Y_FEATURES: tuple[FinalFeatureSpec, ...] = (
    FinalFeatureSpec(
        "ptf_interim",
        "electricity",
        "/v1/markets/dam/data/interim-mcp",
        "hourly",
        "ptf_interim.csv",
        chunk_granularity="daily",
    ),
    FinalFeatureSpec(
        "ptf_kesinlesmis",
        "electricity",
        "/v1/markets/dam/data/mcp",
        "hourly",
        "ptf_kesinlesmis.csv",
    ),
    FinalFeatureSpec(
        "load_forecast_plan",
        "electricity",
        "/v1/consumption/data/load-estimation-plan",
        "hourly",
        "load_forecast_plan.csv",
    ),
    FinalFeatureSpec(
        "real_time_consumption",
        "electricity",
        "/v1/consumption/data/realtime-consumption",
        "hourly",
        "real_time_consumption.csv",
    ),
    FinalFeatureSpec(
        "realtime_generation",
        "electricity",
        "/v1/generation/data/realtime-generation",
        "hourly",
        "realtime_generation.csv",
    ),
    FinalFeatureSpec(
        "res_generation_forecast",
        "electricity",
        "/v1/renewables/data/res-generation-and-forecast",
        "hourly",
        "res_generation_forecast.csv",
    ),
    FinalFeatureSpec(
        "generation_forecast",
        "electricity",
        "/v1/renewables/data/generation-forecast",
        "hourly",
        "generation_forecast.csv",
    ),
    FinalFeatureSpec(
        "yekdem_realtime",
        "electricity",
        "/v1/renewables/data/licensed-realtime-generation",
        "hourly",
        "yekdem_realtime.csv",
    ),
    FinalFeatureSpec(
        "unlicensed_generation_total",
        "electricity",
        "/v1/renewables/data/unlicensed-generation-amount",
        "hourly",
        "unlicensed_generation_total.csv",
    ),
    FinalFeatureSpec(
        "smf",
        "electricity",
        "/v1/markets/bpm/data/system-marginal-price",
        "hourly",
        "smf.csv",
    ),
    FinalFeatureSpec(
        "system_direction",
        "electricity",
        "/v1/markets/bpm/data/system-direction",
        "hourly",
        "system_direction.csv",
    ),
    FinalFeatureSpec(
        "gop_fiyattan_bagimsiz_alis",
        "electricity",
        "/v1/markets/dam/data/price-independent-bid",
        "hourly",
        "gop_fiyattan_bagimsiz_alis.csv",
    ),
    FinalFeatureSpec(
        "gop_fiyattan_bagimsiz_satis",
        "electricity",
        "/v1/markets/dam/data/price-independent-offer",
        "hourly",
        "gop_fiyattan_bagimsiz_satis.csv",
    ),
    FinalFeatureSpec(
        "grf_tl",
        "natural_gas",
        "/v1/markets/sgp/data/daily-reference-price",
        "daily",
        "grf_tl.csv",
    ),
    FinalFeatureSpec(
        "dam_supply_demand",
        "electricity",
        "/v1/markets/dam/data/supply-demand",
        "hourly",
        "dam_supply_demand.csv",
        sort_field="price",
        date_mode="single",
    ),
    FinalFeatureSpec(
        "dam_trade_volume",
        "electricity",
        "/v1/markets/dam/data/day-ahead-market-trade-volume",
        "hourly",
        "dam_trade_volume.csv",
    ),
    FinalFeatureSpec(
        "dam_clearing_quantity",
        "electricity",
        "/v1/markets/dam/data/clearing-quantity",
        "hourly",
        "dam_clearing_quantity.csv",
    ),
    FinalFeatureSpec(
        "bpm_order_summary_up",
        "electricity",
        "/v1/markets/bpm/data/order-summary-up",
        "hourly",
        "bpm_order_summary_up.csv",
    ),
    FinalFeatureSpec(
        "bpm_order_summary_down",
        "electricity",
        "/v1/markets/bpm/data/order-summary-down",
        "hourly",
        "bpm_order_summary_down.csv",
    ),
    FinalFeatureSpec(
        "idm_weighted_average_price",
        "electricity",
        "/v1/markets/idm/data/weighted-average-price",
        "hourly",
        "idm_weighted_average_price.csv",
    ),
    FinalFeatureSpec(
        "idm_trade_value",
        "electricity",
        "/v1/markets/idm/data/trade-value",
        "hourly",
        "idm_trade_value.csv",
        sort_field="kontratTuru",
    ),
    FinalFeatureSpec(
        "imbalance_quantity",
        "electricity",
        "/v1/markets/imbalance/data/imbalance-quantity",
        "monthly",
        "imbalance_quantity.csv",
    ),
    FinalFeatureSpec(
        "imbalance_amount",
        "electricity",
        "/v1/markets/imbalance/data/imbalance-amount",
        "monthly",
        "imbalance_amount.csv",
    ),
    FinalFeatureSpec(
        "yek_generation_cost",
        "electricity",
        "/v1/renewables/data/licensed-generation-cost",
        "monthly",
        "yek_generation_cost.csv",
    ),
    FinalFeatureSpec(
        "yek_portfolio_income",
        "electricity",
        "/v1/renewables/data/portfolio-income",
        "monthly",
        "yek_portfolio_income.csv",
    ),
)

# Kesin PTF icin legacy dosya adi
_FALLBACK_ALIASES: dict[str, list[str]] = {
    "ptf_kesinlesmis.csv": ["ptf.csv"],
}


def _csv_is_valid(path: Path, *, min_rows: int = MIN_CSV_ROWS) -> bool:
    if not path.exists() or path.stat().st_size < 100:
        return False
    try:
        return len(pd.read_csv(path, nrows=min_rows + 10)) >= min_rows
    except Exception:
        return False


def _earliest_date_in_csv(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path, usecols=lambda c: c.lower() in {"date", "datetime", "gasday", "day"})
        if df.empty:
            return None
        col = df.columns[0]
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().any():
            parsed = parsed.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
        else:
            parsed = pd.to_datetime(df[col], errors="coerce")
        return parsed.min()
    except Exception:
        return None


def _has_full_history(path: Path) -> bool:
    if not _csv_is_valid(path):
        return False
    earliest = _earliest_date_in_csv(path)
    if earliest is None or pd.isna(earliest):
        return False
    target = pd.Timestamp(START_DATE_5Y) + timedelta(days=COVERAGE_TOLERANCE_DAYS)
    return earliest <= target


def _find_existing_source(output_file: str) -> Path | None:
    """epias_5y veya final_selected_features icinde mevcut dosyayi bul."""
    primary = RAW_5Y_DIR / output_file
    if _csv_is_valid(primary):
        return primary

    for alias in [output_file, *_FALLBACK_ALIASES.get(output_file, [])]:
        candidate = FALLBACK_RAW_DIR / alias
        if _csv_is_valid(candidate):
            return candidate
    return None


def _dedupe_epias_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [c for c in ("date", "hour", "time", "datetime", "gasDay") if c in df.columns]
    if keys:
        return df.drop_duplicates(subset=keys, keep="last")
    return df.drop_duplicates()


def sync_existing_to_epias_5y() -> list[str]:
    """Mevcut CSV'leri API cagirmadan epias_5y'ye kopyala."""
    RAW_5Y_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for spec in EPIAS_5Y_FEATURES:
        dest = RAW_5Y_DIR / spec.output_file
        if _csv_is_valid(dest):
            continue
        src = _find_existing_source(spec.output_file)
        if src is None:
            continue
        shutil.copy2(src, dest)
        copied.append(spec.feature_name)
        print(f"[5Y SYNC] {spec.feature_name} <- {src.name} ({len(pd.read_csv(dest)):,} satir)")

    usd_dest = RAW_5Y_DIR / "usd_try.csv"
    if not _csv_is_valid(usd_dest):
        usd_src = FALLBACK_RAW_DIR / "usd_try.csv"
        if _csv_is_valid(usd_src, min_rows=10):
            shutil.copy2(usd_src, usd_dest)
            copied.append("usd_try")
            print(f"[5Y SYNC] usd_try <- {usd_src.name}")

    return copied


class Epias5yFetcher(FinalSelectedFeatureFetcher):
    """5y ham veri — mevcut dosyalari tekrar cekmez."""

    def __init__(self, start_date, end_date, *, skip_existing: bool = True) -> None:
        super().__init__(start_date, end_date)
        self.skip_existing = skip_existing
        self.timeout = 120

    def fetch_epias_feature(self, spec: FinalFeatureSpec) -> FetchResult:
        output_path = RAW_5Y_DIR / spec.output_file
        prior_df: pd.DataFrame | None = None

        if self.skip_existing:
            existing = _find_existing_source(spec.output_file)
            if existing is not None:
                if existing != output_path:
                    shutil.copy2(existing, output_path)
                if _has_full_history(output_path):
                    rows = len(pd.read_csv(output_path))
                    print(f"[5Y SKIP] {spec.feature_name} — tam kapsam ({rows:,} satir)")
                    return FetchResult(
                        spec.feature_name,
                        True,
                        rows,
                        "cached_local",
                        str(output_path),
                    )
                prior_df = pd.read_csv(output_path)
                earliest = _earliest_date_in_csv(output_path)
                print(
                    f"[5Y EXTEND] {spec.feature_name} — gecmis eksik "
                    f"(ilk tarih={earliest}, hedef>={START_DATE_5Y})"
                )

        import src.fetch_final_selected_features as mod

        old_raw = mod.RAW_DIR
        mod.RAW_DIR = RAW_5Y_DIR
        try:
            result = super().fetch_epias_feature(spec, skip_if_exists=False)
        finally:
            mod.RAW_DIR = old_raw

        if not result.success and prior_df is not None:
            prior_df.to_csv(output_path, index=False)
            return FetchResult(
                spec.feature_name,
                True,
                len(prior_df),
                "cached_partial",
                str(output_path),
                result.error,
            )

        if prior_df is not None and result.success and output_path.exists():
            try:
                new_df = pd.read_csv(output_path)
                merged = _dedupe_epias_frame(pd.concat([prior_df, new_df], ignore_index=True))
                merged.to_csv(output_path, index=False)
                return FetchResult(
                    spec.feature_name,
                    True,
                    len(merged),
                    "api_merged",
                    str(output_path),
                    result.error,
                )
            except Exception:
                pass
        return result

    def fetch_usd_try(self) -> FetchResult:
        output_path = RAW_5Y_DIR / "usd_try.csv"
        if self.skip_existing and _has_full_history(output_path):
            rows = len(pd.read_csv(output_path))
            print(f"[5Y SKIP] usd_try — tam kapsam ({rows:,} satir)")
            return FetchResult("usd_try", True, rows, "cached_local", str(output_path))

        import src.fetch_final_selected_features as mod

        old_raw = mod.RAW_DIR
        mod.RAW_DIR = RAW_5Y_DIR
        try:
            return super().fetch_usd_try()
        finally:
            mod.RAW_DIR = old_raw


def run_fetch_epias_5y(
    *,
    skip_existing: bool = True,
    sync_fallback: bool = True,
    force_refetch: bool = False,
) -> dict[str, Any]:
    RAW_5Y_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    end = end_date_5y()
    synced: list[str] = []
    if sync_fallback and not force_refetch:
        synced = sync_existing_to_epias_5y()

    fetcher = Epias5yFetcher(START_DATE_5Y, end, skip_existing=skip_existing and not force_refetch)
    results: list[FetchResult] = []
    failures: list[dict[str, Any]] = []

    import sys

    total = len(EPIAS_5Y_FEATURES) + 1
    print(f"[5Y FETCH] {START_DATE_5Y} -> {end} (skip_existing={skip_existing and not force_refetch})", flush=True)
    write_fetch_progress(current="başlatılıyor", index=0, total=total, running=True)

    for i, spec in enumerate(EPIAS_5Y_FEATURES, start=1):
        print(f"[FETCH] {spec.feature_name}", flush=True)
        write_fetch_progress(current=spec.feature_name, index=i, total=total, running=True)
        result = fetcher.fetch_epias_feature(spec)
        results.append(result)
        write_fetch_progress(
            current=spec.feature_name,
            index=i,
            total=total,
            running=True,
            last_result=asdict(result),
        )
        print(
            f"   -> {spec.feature_name}: {'OK' if result.success else 'FAIL'} "
            f"rows={result.rows} source={result.source}",
            flush=True,
        )
        if not result.success:
            failures.append({**asdict(spec), "error": result.error})

    print("[FETCH] usd_try", flush=True)
    write_fetch_progress(current="usd_try", index=total, total=total, running=True)
    usd = fetcher.fetch_usd_try()
    results.append(usd)
    write_fetch_progress(
        current="usd_try",
        index=total,
        total=total,
        running=True,
        last_result=asdict(usd),
    )
    if not usd.success:
        failures.append({"feature_name": "usd_try", "error": usd.error})

    write_fetch_progress(current=None, index=total, total=total, running=False, last_result=asdict(usd))
    pd.DataFrame(failures).to_csv(LOG_DIR / "failed_epias_5y.csv", index=False)
    pd.DataFrame([asdict(r) for r in results]).to_csv(LOG_DIR / "epias_5y_fetch_status.csv", index=False)

    skipped = sum(1 for r in results if r.source == "cached_local")
    fetched = sum(1 for r in results if r.success and r.source not in ("cached_local",))
    ok = [r for r in results if r.success]
    return {
        "start_date": START_DATE_5Y.isoformat(),
        "end_date": end.isoformat(),
        "attempted": len(results),
        "successful": len(ok),
        "skipped_existing": skipped,
        "fetched_api": fetched,
        "synced_from_fallback": synced,
        "failed": len(results) - len(ok),
        "raw_dir": str(RAW_5Y_DIR),
    }


def run_fetch_epias_5y_features(
    feature_names: list[str],
    *,
    force_refetch: bool = True,
) -> list[FetchResult]:
    """Yalnızca seçili serileri çeker (API düzeltme / eksik tamamlama)."""
    RAW_5Y_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {n.strip() for n in feature_names if n.strip()}
    specs = [s for s in EPIAS_5Y_FEATURES if s.feature_name in wanted]
    if not specs:
        raise ValueError(f"Bilinmeyen seri: {feature_names}")

    end = end_date_5y()
    fetcher = Epias5yFetcher(START_DATE_5Y, end, skip_existing=not force_refetch)
    results: list[FetchResult] = []
    total = len(specs)
    write_fetch_progress(current="başlatılıyor", index=0, total=total, running=True)

    for i, spec in enumerate(specs, start=1):
        print(f"[FETCH] {spec.feature_name}", flush=True)
        write_fetch_progress(current=spec.feature_name, index=i, total=total, running=True)
        result = fetcher.fetch_epias_feature(spec)
        results.append(result)
        write_fetch_progress(
            current=spec.feature_name,
            index=i,
            total=total,
            running=True,
            last_result=asdict(result),
        )
        print(
            f"   -> {spec.feature_name}: {'OK' if result.success else 'FAIL'} "
            f"rows={result.rows} source={result.source}",
            flush=True,
        )

    write_fetch_progress(current=None, index=total, total=total, running=False)
    pd.DataFrame([asdict(r) for r in results]).to_csv(
        LOG_DIR / "epias_5y_fetch_retry_status.csv", index=False
    )
    sync_fetch_live_to_app_data()
    return results
