from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TIMEZONE = ZoneInfo("Europe/Istanbul")


@dataclass(frozen=True)
class EpiasSettings:
    base_url: str
    auth_url: str
    mcp_endpoint: str
    username: str | None
    password: str | None
    tgt: str | None
    timeout_seconds: int
    page_size: int
    start_date: date
    end_date: date
    csv_path: Path
    xlsx_path: Path
    debug: bool


def get_settings() -> EpiasSettings:
    load_dotenv(PROJECT_ROOT / ".env")

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    return EpiasSettings(
        base_url=os.getenv(
            "EPIAS_BASE_URL",
            "https://seffaflik.epias.com.tr/electricity-service",
        ).rstrip("/"),
        auth_url=os.getenv(
            "EPIAS_AUTH_URL",
            "https://giris.epias.com.tr/cas/v1/tickets",
        ),
        mcp_endpoint="/v1/markets/dam/data/mcp",
        username=os.getenv("EPIAS_USERNAME"),
        password=os.getenv("EPIAS_PASSWORD"),
        tgt=os.getenv("EPIAS_TGT"),
        timeout_seconds=int(os.getenv("EPIAS_TIMEOUT_SECONDS", "45")),
        page_size=int(os.getenv("EPIAS_PAGE_SIZE", "10000")),
        start_date=date(2025, 1, 1),
        end_date=datetime.now(TIMEZONE).date(),
        csv_path=DATA_RAW_DIR / "ptf_2025_to_today.csv",
        xlsx_path=DATA_RAW_DIR / "ptf_2025_to_today.xlsx",
        debug=os.getenv("DEBUG", "1").lower() in {"1", "true", "yes", "on"},
    )
