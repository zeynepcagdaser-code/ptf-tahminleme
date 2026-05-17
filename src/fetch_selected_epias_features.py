from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import PROJECT_ROOT, get_settings
from src.epias_client import EpiasAuthError, extract_items, iter_monthly_ranges, to_epias_datetime
from src.manual_epias_feature_registry import get_selected_feature_registry, save_selected_feature_registry


RAW_DIRS = {
    "electricity": PROJECT_ROOT / "data" / "raw" / "electricity_features",
    "natural_gas": PROJECT_ROOT / "data" / "raw" / "natural_gas_features",
}
DEBUG_DIR = PROJECT_ROOT / "data" / "raw" / "debug_responses"
LOG_DIR = PROJECT_ROOT / "logs"
FAILED_LOG = LOG_DIR / "failed_selected_features.csv"
DIAGNOSIS_LOG = LOG_DIR / "failed_endpoint_diagnosis.csv"
RETRIED_LOG = LOG_DIR / "retried_failed_features.csv"
RECOVERED_LOG = LOG_DIR / "recovered_features.csv"
STILL_FAILED_LOG = LOG_DIR / "still_failed_features.csv"


@dataclass(frozen=True)
class FetchSummary:
    total_features: int
    success_count: int
    failed_count: int
    empty_count: int


class SelectedEpiasFeatureFetcher:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.session = self._build_session()
        self._tgt = self.settings.tgt
        self.base_urls = {
            "electricity": "https://seffaflik.epias.com.tr/electricity-service",
            "natural_gas": "https://seffaflik.epias.com.tr/natural-gas-service",
        }

    def _build_session(self) -> Session:
        retry = Retry(
            total=1,
            connect=2,
            read=2,
            backoff_factor=1,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        return session

    def fetch_all(self, registry: pd.DataFrame) -> FetchSummary:
        for directory in [*RAW_DIRS.values(), DEBUG_DIR, LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        failed: list[dict[str, Any]] = []
        empty: list[dict[str, Any]] = []
        success_count = 0

        for row in registry.to_dict(orient="records"):
            feature_name = row["feature_name"]
            print(f"[FETCH] {feature_name}", flush=True)
            try:
                test_items = self.fetch_feature(row, self.settings.start_date, min(self.settings.end_date, date(2025, 1, 3)))
                if not test_items:
                    empty.append({**row, "reason": "test_range_empty"})
                    continue

                full_items: list[dict[str, Any]] = []
                for start, end in iter_monthly_ranges(self.settings.start_date, self.settings.end_date):
                    full_items.extend(self.fetch_feature(row, start, end))
                    time.sleep(0.15)

                if not full_items:
                    empty.append({**row, "reason": "full_range_empty"})
                    continue

                df = pd.DataFrame(full_items)
                for key in ["feature_name", "market_type", "service", "endpoint_path", "frequency"]:
                    df[key] = row.get(key)
                output_path = RAW_DIRS[row["service"]] / f"{feature_name}.csv"
                df.to_csv(output_path, index=False)
                success_count += 1
            except Exception as exc:
                failed.append(self._failure_record(row, exc))
                print(f"  [WARN] {feature_name} cekilemedi: {exc}", flush=True)

        pd.DataFrame(failed).to_csv(FAILED_LOG, index=False)
        pd.DataFrame(empty).to_csv(LOG_DIR / "empty_selected_features.csv", index=False)
        return FetchSummary(len(registry), success_count, len(failed), len(empty))

    def fetch_feature(self, row: dict[str, Any], start_date: date, end_date: date) -> list[dict[str, Any]]:
        service = row["service"]
        url = f"{self.base_urls[service]}{row['endpoint_path']}"
        method = str(row.get("method", "POST")).upper()
        headers = self._headers()

        if method == "GET":
            payload = {}
            response = self._request_with_backoff("GET", url, headers=headers, payload=payload)
        else:
            payload = self._build_payload(row, start_date, end_date)
            response = self._request_with_backoff("POST", url, headers=headers, payload=payload)

        if response.status_code == 401:
            raise EpiasAuthError("EPİAŞ TGT/kullanici bilgisi gerekli veya gecersiz.")
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

        data = response.json()
        self._save_debug(row["feature_name"], data)
        try:
            return extract_items(data)
        except Exception:
            body = data.get("body", data) if isinstance(data, dict) else data
            if isinstance(body, list):
                return body
            if isinstance(body, dict):
                return [body]
            return []

    def retry_failed_only(self, failed_path: Path = FAILED_LOG) -> FetchSummary:
        if not failed_path.exists():
            raise FileNotFoundError(f"Basarisiz endpoint logu bulunamadi: {failed_path}")

        failed_df = pd.read_csv(failed_path)
        registry = save_selected_feature_registry()
        retry_registry = registry[registry["feature_name"].isin(failed_df["feature_name"].tolist())].copy()
        diagnosis = diagnose_failed_endpoints(failed_df)
        diagnosis.to_csv(DIAGNOSIS_LOG, index=False)

        retried: list[dict[str, Any]] = []
        recovered: list[dict[str, Any]] = []
        still_failed: list[dict[str, Any]] = []

        for row in retry_registry.to_dict(orient="records"):
            feature_name = row["feature_name"]
            print(f"[RETRY] {feature_name}", flush=True)
            try:
                test_items = self.fetch_feature(row, self.settings.start_date, min(self.settings.end_date, date(2025, 1, 3)))
                if not test_items:
                    still_failed.append({**row, "fail_reason": "empty_response_after_retry"})
                    continue

                full_items: list[dict[str, Any]] = []
                for start, end in iter_monthly_ranges(self.settings.start_date, self.settings.end_date):
                    full_items.extend(self.fetch_feature(row, start, end))
                    time.sleep(1.5)

                if not full_items:
                    still_failed.append({**row, "fail_reason": "empty_full_range_after_retry"})
                    continue

                df = pd.DataFrame(full_items)
                for key in ["feature_name", "market_type", "service", "endpoint_path", "frequency"]:
                    df[key] = row.get(key)
                output_path = RAW_DIRS[row["service"]] / f"{feature_name}.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                recovered.append({**row, "rows": len(df), "output_path": str(output_path)})
                retried.append({**row, "retry_status": "recovered"})
            except Exception as exc:
                record = self._failure_record(row, exc)
                still_failed.append(record)
                retried.append({**row, "retry_status": "failed", "error": type(exc).__name__, "message": str(exc)[:1000]})
                print(f"  [WARN] retry basarisiz: {exc}", flush=True)
            time.sleep(3)

        pd.DataFrame(retried).to_csv(RETRIED_LOG, index=False)
        pd.DataFrame(recovered).to_csv(RECOVERED_LOG, index=False)
        pd.DataFrame(still_failed).to_csv(STILL_FAILED_LOG, index=False)
        return FetchSummary(len(retry_registry), len(recovered), len(still_failed), 0)

    def _build_payload(self, row: dict[str, Any], start_date: date, end_date: date) -> dict[str, Any]:
        sort_field = row.get("sort_field") if isinstance(row.get("sort_field"), str) and row.get("sort_field") else "date"
        date_mode = row.get("date_mode") if isinstance(row.get("date_mode"), str) and row.get("date_mode") else "range"
        payload: dict[str, Any] = {
            "page": {
                "number": 1,
                "size": self.settings.page_size,
                "sort": {"field": sort_field, "direction": "ASC"},
            }
        }
        if date_mode == "range":
            payload["startDate"] = to_epias_datetime(start_date)
            payload["endDate"] = to_epias_end_datetime(end_date)
        elif date_mode == "period":
            payload["period"] = to_epias_datetime(start_date)
        elif date_mode == "none":
            pass

        extra_body = row.get("extra_body")
        if isinstance(extra_body, float) and math.isnan(extra_body):
            extra_body = None
        if isinstance(extra_body, str) and extra_body:
            extra_body = json.loads(extra_body.replace("'", '"'))
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        return payload

    def _request_with_backoff(self, method: str, url: str, headers: dict[str, str], payload: dict[str, Any]) -> requests.Response:
        max_attempts = 3
        last_response: requests.Response | None = None
        for attempt in range(1, max_attempts + 1):
            if method == "GET":
                response = self.session.get(url, headers=headers, timeout=min(max(self.settings.timeout_seconds, 20), 30))
            else:
                response = self.session.post(url, json=payload, headers=headers, timeout=min(max(self.settings.timeout_seconds, 20), 30))
            last_response = response
            if response.status_code != 429:
                return response
            wait_seconds = min(30, 10 * attempt)
            print(f"  [RATE_LIMIT] 429 alindi, {wait_seconds} sn bekleniyor...", flush=True)
            time.sleep(wait_seconds)
        return last_response  # type: ignore[return-value]

    def _failure_record(self, row: dict[str, Any], exc: Exception) -> dict[str, Any]:
        payload = {}
        try:
            payload = self._build_payload(row, self.settings.start_date, min(self.settings.end_date, date(2025, 1, 3)))
        except Exception:
            payload = {}
        message = str(exc)
        return {
            **row,
            "error": type(exc).__name__,
            "message": message[:1000],
            "status_code": extract_status_code(message),
            "fail_reason": classify_failure(message),
            "request_body": json.dumps(payload, ensure_ascii=False),
            "response_sample": message[:500],
        }

    def _headers(self) -> dict[str, str]:
        return {"Accept": "application/json", "Content-Type": "application/json", "TGT": self._get_tgt()}

    def _get_tgt(self) -> str:
        if self._tgt:
            return self._tgt
        if not self.settings.username or not self.settings.password:
            raise EpiasAuthError(".env icinde EPIAS_TGT veya EPIAS_USERNAME/EPIAS_PASSWORD gerekli.")
        response = self.session.post(
            self.settings.auth_url,
            data={"username": self.settings.username, "password": self.settings.password},
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "text/plain"},
            timeout=self.settings.timeout_seconds,
        )
        if response.status_code != 201:
            raise EpiasAuthError(f"TGT alinamadi: HTTP {response.status_code} {response.text[:300]}")
        self._tgt = response.text.strip()
        return self._tgt

    def _save_debug(self, feature_name: str, data: Any) -> None:
        path = DEBUG_DIR / f"{feature_name}.json"
        if not path.exists():
            with path.open("w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)


def run_selected_feature_fetch() -> FetchSummary:
    registry = save_selected_feature_registry()
    fetcher = SelectedEpiasFeatureFetcher()
    summary = fetcher.fetch_all(registry)
    print(f"Fetch ozeti: total={summary.total_features}, success={summary.success_count}, failed={summary.failed_count}, empty={summary.empty_count}")
    return summary


def run_retry_failed_selected_features() -> FetchSummary:
    fetcher = SelectedEpiasFeatureFetcher()
    summary = fetcher.retry_failed_only()
    print(
        "Retry ozeti: "
        f"failed_initial={summary.total_features}, retried={summary.total_features}, "
        f"recovered={summary.success_count}, still_failed={summary.failed_count}"
    )
    return summary


def to_epias_end_datetime(value: date) -> str:
    return f"{value.isoformat()}T23:59:59+03:00"


def extract_status_code(message: str) -> int | None:
    if "HTTP " in message:
        try:
            return int(message.split("HTTP ", 1)[1][:3])
        except ValueError:
            return None
    if "429" in message:
        return 429
    if "Read timed out" in message or "Timeout" in message:
        return None
    return None


def classify_failure(message: str) -> str:
    lower = message.lower()
    if "429" in lower or "too many" in lower:
        return "429_rate_limit"
    if "http 400" in lower:
        if "sıralama" in lower or "sort" in lower:
            return "unsupported_sort_field"
        if "periyot" in lower or "period" in lower:
            return "missing_request_field"
        return "400_bad_request"
    if "http 401" in lower or "http 403" in lower:
        return "auth"
    if "http 404" in lower or "method not found" in lower:
        return "404_endpoint_wrong"
    if "timed out" in lower or "timeout" in lower:
        return "timeout"
    if "empty" in lower:
        return "empty_response"
    return "unknown"


def diagnose_failed_endpoints(failed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in failed_df.to_dict(orient="records"):
        message = str(row.get("message", ""))
        rows.append(
            {
                "feature_name": row.get("feature_name"),
                "endpoint_path": row.get("endpoint_path"),
                "http_status_code": extract_status_code(message),
                "error_message": message,
                "response_sample": message[:500],
                "request_body": "",
                "fail_reason": classify_failure(message),
                "suggested_action": suggested_action(classify_failure(message)),
            }
        )
    return pd.DataFrame(rows)


def suggested_action(reason: str) -> str:
    return {
        "429_rate_limit": "retry_with_backoff_and_slower_spacing",
        "unsupported_sort_field": "fix_page_sort_field_from_api_error",
        "missing_request_field": "add_required_endpoint_specific_field",
        "404_endpoint_wrong": "fix_endpoint_path_from_official_docs",
        "timeout": "increase_timeout_and_retry_smaller_chunks",
        "auth": "check_tgt_or_credentials",
    }.get(reason, "manual_review")
