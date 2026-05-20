from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.dl_5y_config import HOURLY_5Y_PATH, START_DATE_5Y, end_date_5y


RAW_5Y_DIR = PROJECT_ROOT / "data" / "raw" / "epias_5y"
FETCH_STATUS_PATH = PROJECT_ROOT / "logs" / "epias_5y_fetch_status.csv"
FETCH_PROGRESS_PATH = PROJECT_ROOT / "logs" / "epias_5y_fetch_progress.json"
FETCH_PID_PATH = PROJECT_ROOT / "logs" / "epias_5y_fetch.pid"
FETCH_LIVE_APP_PATH = PROJECT_ROOT / "app_data" / "epias_5y_fetch_live.json"
FAILED_PATH = PROJECT_ROOT / "logs" / "failed_epias_5y.csv"

FEATURE_LABELS = {
    "ptf_interim.csv": "PTF (I-MCP / kesinleşmemiş)",
    "ptf_kesinlesmis.csv": "PTF (MCP / kesinleşmiş)",
    "load_forecast_plan.csv": "Yük tahmin planı",
    "real_time_consumption.csv": "Gerçekleşen tüketim",
    "realtime_generation.csv": "Gerçek zamanlı üretim (kaynak kırılımı)",
    "res_generation_forecast.csv": "RES üretim/tahmin",
    "generation_forecast.csv": "GES üretim tahmini",
    "yekdem_realtime.csv": "YEKDEM gerçek zamanlı üretim",
    "unlicensed_generation_total.csv": "Lisanssız üretim",
    "smf.csv": "SMF",
    "system_direction.csv": "Sistem yönü",
    "gop_fiyattan_bagimsiz_alis.csv": "GOP fiyattan bağımsız alış",
    "gop_fiyattan_bagimsiz_satis.csv": "GOP fiyattan bağımsız satış",
    "grf_tl.csv": "Doğalgaz referans (GRF TL)",
    "usd_try.csv": "USD/TRY",
    "dam_supply_demand.csv": "GOP arz-talep",
    "dam_trade_volume.csv": "GOP işlem hacmi",
    "dam_clearing_quantity.csv": "GOP eşleşme miktarı",
    "bpm_order_summary_up.csv": "YAL talimat",
    "bpm_order_summary_down.csv": "YAT talimat",
    "idm_weighted_average_price.csv": "GİP ağırlıklı ortalama fiyat",
    "idm_trade_value.csv": "GİP işlem hacmi",
    "imbalance_quantity.csv": "Dengesizlik miktarı (aylık)",
    "imbalance_amount.csv": "Dengesizlik tutarı (aylık)",
    "yek_generation_cost.csv": "YEK üretim maliyeti (aylık)",
    "yek_portfolio_income.csv": "YEKDEM portföy geliri (aylık)",
}


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().any():
        return parsed.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None)
    return pd.to_datetime(series, errors="coerce")


def _file_date_range(path: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int]:
    if not path.exists() or path.stat().st_size < 50:
        return None, None, 0
    try:
        df = pd.read_csv(path)
        rows = len(df)
        for col in ("datetime", "date", "gasDay", "day"):
            if col in df.columns:
                d = _parse_dates(df[col])
                if d.notna().any():
                    return d.min(), d.max(), rows
        return None, None, rows
    except Exception:
        return None, None, 0


def build_epias_5y_inventory() -> pd.DataFrame:
    target_start = pd.Timestamp(START_DATE_5Y)
    target_end = pd.Timestamp(end_date_5y()) + pd.Timedelta(hours=23)
    expected_hours = int((target_end - target_start).total_seconds() // 3600) + 1

    status_df = pd.DataFrame()
    if FETCH_STATUS_PATH.exists():
        status_df = pd.read_csv(FETCH_STATUS_PATH)

    rows: list[dict] = []
    for fname, label in FEATURE_LABELS.items():
        path = RAW_5Y_DIR / fname
        dmin, dmax, nrows = _file_date_range(path)
        if nrows == 0:
            coverage = 0.0
            dur = "Yok"
        else:
            span_h = 0
            if dmin is not None and dmax is not None and not pd.isna(dmin):
                span_h = int((dmax - dmin).total_seconds() // 3600) + 1
            coverage = min(100.0, span_h / max(expected_hours, 1) * 100)
            if coverage >= 85:
                dur = "Tam (5y)"
            elif coverage >= 25:
                dur = "Kısmi"
            else:
                dur = "Kısa (~1 yıl)"

        src = ""
        if not status_df.empty and "output_path" in status_df.columns:
            match = status_df[status_df["output_path"].astype(str).str.endswith(fname)]
            if not match.empty:
                src = str(match.iloc[-1].get("source", ""))

        rows.append(
            {
                "Dosya": fname,
                "Seri": label,
                "Satır": nrows,
                "İlk tarih": dmin,
                "Son tarih": dmax,
                "Kapsam %": round(coverage, 1),
                "Durum": dur,
                "Kaynak": src or ("dosya" if nrows else "—"),
            }
        )

    return pd.DataFrame(rows)


def load_epias_5y_timeseries(filename: str, value_candidates: list[str] | None = None) -> pd.DataFrame:
    path = RAW_5Y_DIR / filename
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    dt = None
    for col in ("datetime", "date"):
        if col in df.columns:
            base = _parse_dates(df[col])
            if col == "date" and "hour" in df.columns:
                h = df["hour"].astype(str).str.slice(0, 2)
                dt = pd.to_datetime(base.dt.date.astype(str) + " " + h, errors="coerce")
            elif col == "date" and "time" in df.columns:
                dt = pd.to_datetime(base.dt.date.astype(str) + " " + df["time"].astype(str), errors="coerce")
            else:
                dt = base
            break
    if dt is None:
        return pd.DataFrame()

    val_col = None
    if value_candidates:
        for c in value_candidates:
            if c in df.columns:
                val_col = c
                break
    if val_col is None:
        ignore = {"datetime", "date", "hour", "time", "feature_name", "service", "endpoint_path", "frequency"}
        nums = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
        val_col = nums[0] if nums else None
    if val_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({"datetime": dt, "value": pd.to_numeric(df[val_col], errors="coerce")})
    return out.dropna(subset=["datetime"]).sort_values("datetime")


def is_fetch_process_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "run_5y_dl_pipeline.py --fetch-only"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def load_fetch_progress() -> dict:
    if not FETCH_PROGRESS_PATH.exists():
        return {}
    try:
        return json.loads(FETCH_PROGRESS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_fetch_progress(
    *,
    current: str | None = None,
    index: int = 0,
    total: int = 0,
    running: bool = True,
    last_result: dict | None = None,
) -> None:
    FETCH_PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prev = load_fetch_progress()
    completed = list(prev.get("completed", []))
    if last_result and last_result.get("feature_name"):
        name = last_result["feature_name"]
        if name not in completed:
            completed.append(name)

    inv = build_epias_5y_inventory()
    stats = inventory_summary_stats(inv)
    newest = ""
    newest_mtime = ""
    if RAW_5Y_DIR.exists():
        files = [p for p in RAW_5Y_DIR.glob("*.csv") if p.stat().st_size > 50]
        if files:
            latest = max(files, key=lambda p: p.stat().st_mtime)
            newest = latest.name
            newest_mtime = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "running": running,
        "current": current,
        "index": index,
        "total": total,
        "completed": completed,
        "last_result": last_result,
        "newest_file": newest,
        "newest_file_mtime": newest_mtime,
        "inventory": stats,
    }
    FETCH_PROGRESS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    sync_fetch_live_to_app_data()


def _inventory_to_records(inv: pd.DataFrame) -> list[dict]:
    if inv.empty:
        return []
    out = inv.copy()
    for col in ("İlk tarih", "Son tarih"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out.to_dict(orient="records")


def sync_fetch_live_to_app_data() -> Path:
    """Streamlit Cloud (GitHub) için hafif canlı özet — ham CSV git'e gitmez."""
    progress = load_fetch_progress()
    inv = build_epias_5y_inventory()
    stats = inventory_summary_stats(inv)
    running = is_fetch_process_running()
    payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "running": running,
        "progress": progress,
        "stats": stats,
        "series": _inventory_to_records(inv),
        "github_note": "Bu dosya push edilince Streamlit Cloud paneli güncellenir.",
    }
    FETCH_LIVE_APP_PATH.parent.mkdir(parents=True, exist_ok=True)
    FETCH_LIVE_APP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return FETCH_LIVE_APP_PATH


def load_fetch_live_from_app_data() -> dict:
    if not FETCH_LIVE_APP_PATH.exists():
        return {}
    try:
        return json.loads(FETCH_LIVE_APP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_live_fetch_snapshot() -> dict:
    progress = load_fetch_progress()
    inv = build_epias_5y_inventory()
    stats = inventory_summary_stats(inv)
    running = is_fetch_process_running()
    app_live = load_fetch_live_from_app_data()

    if inv.empty and app_live.get("series"):
        inv = pd.DataFrame(app_live["series"])
        stats = app_live.get("stats") or inventory_summary_stats(inv)
        if not running:
            running = bool(app_live.get("running"))
        if not progress:
            progress = app_live.get("progress") or {}

    return {
        "running": running,
        "progress": progress,
        "stats": stats,
        "inventory": inv,
        "app_live": app_live,
        "app_live_updated": app_live.get("updated_at"),
    }


def inventory_summary_stats(inv: pd.DataFrame) -> dict[str, int]:
    if inv.empty:
        return {"total": 0, "full": 0, "partial": 0, "short": 0, "missing": 0}
    return {
        "total": len(inv),
        "full": int((inv["Durum"] == "Tam (5y)").sum()),
        "partial": int(inv["Durum"].isin(["Kısmi"]).sum()),
        "short": int(inv["Durum"].str.contains("Kısa", na=False).sum()),
        "missing": int((inv["Satır"] == 0).sum()),
    }
