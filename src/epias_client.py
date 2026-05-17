from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import EpiasSettings, TIMEZONE


class EpiasAuthError(RuntimeError):
    pass


class EpiasApiError(RuntimeError):
    pass


class EpiasClient:
    def __init__(self, settings: EpiasSettings) -> None:
        self.settings = settings
        self.session = self._build_session()
        self._tgt = settings.tgt

    def _build_session(self) -> Session:
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch_ptf_monthly(self, start_date: date, end_date: date) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for chunk_start, chunk_end in iter_monthly_ranges(start_date, end_date):
            debug_print(
                self.settings.debug,
                f"Fetching PTF: {chunk_start.isoformat()} -> {chunk_end.isoformat()}",
            )
            items = self.fetch_mcp(chunk_start, chunk_end)
            debug_print(self.settings.debug, f"  rows received: {len(items)}")
            if items:
                frames.append(pd.DataFrame(items))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def fetch_mcp(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        url = f"{self.settings.base_url}{self.settings.mcp_endpoint}"
        payload = {
            "startDate": to_epias_datetime(start_date),
            "endDate": to_epias_datetime(end_date),
            "page": {
                "number": 1,
                "size": self.settings.page_size,
                "sort": {"field": "date", "direction": "ASC"},
            },
        }
        response = self.session.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=self.settings.timeout_seconds,
        )
        self._raise_for_response(response)
        data = response.json()
        return extract_items(data)

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "TGT": self._get_tgt(),
        }

    def _get_tgt(self) -> str:
        if self._tgt:
            return self._tgt

        if not self.settings.username or not self.settings.password:
            raise EpiasAuthError(
                "EPİAŞ API TGT bilgisi yok. .env dosyasına EPIAS_TGT ekleyin "
                "veya EPIAS_USERNAME / EPIAS_PASSWORD tanımlayın."
            )

        debug_print(self.settings.debug, "Requesting EPİAŞ TGT token...")
        response = self.session.post(
            self.settings.auth_url,
            data={
                "username": self.settings.username,
                "password": self.settings.password,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "text/plain",
            },
            timeout=self.settings.timeout_seconds,
        )

        if response.status_code != 201:
            raise EpiasAuthError(
                f"TGT alınamadı. HTTP {response.status_code}: {response.text[:500]}"
            )

        self._tgt = response.text.strip()
        return self._tgt

    def _raise_for_response(self, response: Response) -> None:
        if response.ok:
            return

        message = response.text[:1000]
        if response.status_code == 401:
            raise EpiasAuthError(
                "EPİAŞ kimlik doğrulama hatası. TGT süresi dolmuş olabilir "
                f"veya kimlik bilgisi eksik olabilir. HTTP 401: {message}"
            )

        raise EpiasApiError(f"EPİAŞ API hatası. HTTP {response.status_code}: {message}")


def iter_monthly_ranges(start_date: date, end_date: date) -> list[tuple[date, date]]:
    if start_date > end_date:
        raise ValueError("start_date end_date degerinden buyuk olamaz.")

    ranges: list[tuple[date, date]] = []
    cursor = start_date

    while cursor <= end_date:
        last_day = monthrange(cursor.year, cursor.month)[1]
        month_end = date(cursor.year, cursor.month, last_day)
        chunk_end = min(month_end, end_date)
        ranges.append((cursor, chunk_end))

        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    return ranges


def to_epias_datetime(value: date) -> str:
    dt = datetime(value.year, value.month, value.day, tzinfo=TIMEZONE)
    return dt.isoformat(timespec="seconds")


def extract_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    body = data.get("body", data)

    if isinstance(body, dict) and isinstance(body.get("items"), list):
        return body["items"]

    found = find_first_items_list(body)
    if found is None:
        raise EpiasApiError(f"EPİAŞ yanıtında items listesi bulunamadı: {str(data)[:500]}")

    return found


def find_first_items_list(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value

    if isinstance(value, dict):
        for key in ("items", "mcpList", "ptfList", "data"):
            candidate = value.get(key)
            result = find_first_items_list(candidate)
            if result is not None:
                return result

        for candidate in value.values():
            result = find_first_items_list(candidate)
            if result is not None:
                return result

    return None


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[DEBUG] {message}")
