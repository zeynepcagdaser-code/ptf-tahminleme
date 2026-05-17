from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timedelta
from typing import Any

import pandas as pd
import requests

from src.config import PROJECT_ROOT, TIMEZONE, get_settings
from src.epias_client import EpiasClient, extract_items, iter_monthly_ranges


RAW_DIR = PROJECT_ROOT / "data" / "raw" / "final_selected_features"
DEBUG_DIR = PROJECT_ROOT / "data" / "raw" / "debug_responses" / "final_selected_features"
LOG_DIR = PROJECT_ROOT / "logs"
FAILED_LOG_PATH = LOG_DIR / "failed_final_selected_features.csv"
FETCH_STATUS_PATH = LOG_DIR / "final_selected_features_fetch_status.csv"


@dataclass(frozen=True)
class FinalFeatureSpec:
    feature_name: str
    service: str
    endpoint_path: str
    frequency: str
    output_file: str
    fallback_file: str | None = None
    sort_field: str = "date"


@dataclass
class FetchResult:
    feature_name: str
    success: bool
    rows: int
    source: str
    output_path: str
    error: str = ""


FINAL_EPIAS_FEATURES: tuple[FinalFeatureSpec, ...] = (
    FinalFeatureSpec(
        "ptf",
        "electricity",
        "/v1/markets/dam/data/mcp",
        "hourly",
        "ptf.csv",
        None,
    ),
    FinalFeatureSpec(
        "gop_fiyattan_bagimsiz_alis",
        "electricity",
        "/v1/markets/dam/data/price-independent-bid",
        "hourly",
        "gop_fiyattan_bagimsiz_alis.csv",
        None,
    ),
    FinalFeatureSpec(
        "gop_fiyattan_bagimsiz_satis",
        "electricity",
        "/v1/markets/dam/data/price-independent-offer",
        "hourly",
        "gop_fiyattan_bagimsiz_satis.csv",
        None,
    ),
    FinalFeatureSpec(
        "load_forecast_plan",
        "electricity",
        "/v1/consumption/data/load-estimation-plan",
        "hourly",
        "load_forecast_plan.csv",
        None,
    ),
    FinalFeatureSpec(
        "realtime_generation",
        "electricity",
        "/v1/generation/data/realtime-generation",
        "hourly",
        "realtime_generation.csv",
        None,
    ),
    FinalFeatureSpec(
        "real_time_consumption",
        "electricity",
        "/v1/consumption/data/realtime-consumption",
        "hourly",
        "real_time_consumption.csv",
        None,
    ),
    FinalFeatureSpec(
        "grf_tl",
        "natural_gas",
        "/v1/markets/sgp/data/daily-reference-price",
        "daily",
        "grf_tl.csv",
        None,
    ),
    FinalFeatureSpec(
        "unlicensed_generation_total",
        "electricity",
        "/v1/renewables/data/unlicensed-generation-amount",
        "hourly",
        "unlicensed_generation_total.csv",
        None,
    ),
    FinalFeatureSpec(
        "smf",
        "electricity",
        "/v1/markets/bpm/data/system-marginal-price",
        "hourly",
        "smf.csv",
        None,
    ),
)


def run_fetch_final_selected_features() -> dict[str, Any]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    fetcher = FinalSelectedFeatureFetcher(settings.start_date, settings.end_date)

    results: list[FetchResult] = []
    failures: list[dict[str, Any]] = []

    for spec in FINAL_EPIAS_FEATURES:
        result = fetcher.fetch_epias_feature(spec)
        results.append(result)
        if not result.success:
            failures.append({**asdict(spec), "error": result.error})

    usd_result = fetcher.fetch_usd_try()
    results.append(usd_result)
    if not usd_result.success:
        failures.append(
            {
                "feature_name": "usd_try",
                "service": "external_fx",
                "endpoint_path": "https://api.frankfurter.app/{start}..{end}?from=USD&to=TRY",
                "frequency": "daily",
                "output_file": "usd_try.csv",
                "fallback_file": "data/raw/final_selected_features/usd_try_manual.csv",
                "sort_field": "date",
                "error": usd_result.error,
            }
        )

    pd.DataFrame(failures).to_csv(FAILED_LOG_PATH, index=False)
    pd.DataFrame([asdict(item) for item in results]).to_csv(FETCH_STATUS_PATH, index=False)

    successful = [item for item in results if item.success]
    return {
        "attempted_features": len(results),
        "successful_features": len(successful),
        "failed_features": len(results) - len(successful),
        "results": [asdict(item) for item in results],
        "failed_log_path": str(FAILED_LOG_PATH),
        "start_date": settings.start_date.isoformat(),
        "end_date": settings.end_date.isoformat(),
        "fetch_status_path": str(FETCH_STATUS_PATH),
    }


class FinalSelectedFeatureFetcher:
    def __init__(self, start_date: date, end_date: date) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.settings = get_settings()
        self.client = EpiasClient(self.settings)
        self.session = requests.Session()
        self.timeout = self.settings.timeout_seconds
        self.page_size = self.settings.page_size

    def fetch_epias_feature(self, spec: FinalFeatureSpec) -> FetchResult:
        output_path = RAW_DIR / spec.output_file
        frames: list[pd.DataFrame] = []
        last_error = ""

        print(f"[FETCH] {spec.feature_name}")

        for chunk_start, chunk_end in iter_monthly_ranges(self.start_date, self.end_date):
            try:
                items = self._request_epias_chunk_with_backoff_window(spec, chunk_start, chunk_end)
                if items:
                    frame = pd.DataFrame(items)
                    frame["feature_name"] = spec.feature_name
                    frame["service"] = spec.service
                    frame["endpoint_path"] = spec.endpoint_path
                    frame["frequency"] = spec.frequency
                    frames.append(frame)
                time.sleep(0.25)
            except Exception as exc:  # noqa: BLE001 - pipeline must continue endpoint by endpoint.
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"  chunk failed {chunk_start} -> {chunk_end}: {last_error}")
                continue

        if frames:
            data = pd.concat(frames, ignore_index=True)
            data = data.drop_duplicates()
            data.to_csv(output_path, index=False)
            source = "api_partial" if last_error else "api"
            return FetchResult(spec.feature_name, True, len(data), source, str(output_path), last_error)

        if output_path.exists():
            output_path.unlink()
        return FetchResult(spec.feature_name, False, 0, "failed", str(output_path), last_error or "empty response")

    def _request_epias_chunk_with_backoff_window(
        self,
        spec: FinalFeatureSpec,
        chunk_start: date,
        chunk_end: date,
    ) -> list[dict[str, Any]]:
        trial_end = chunk_end
        last_error: Exception | None = None
        while trial_end >= chunk_start:
            try:
                return self._request_epias_chunk(spec, chunk_start, trial_end)
            except Exception as exc:  # noqa: BLE001 - try older end dates for not-yet-published current windows.
                last_error = exc
                message = str(exc)
                if "SEF1124" not in message and "geçmiş zaman" not in message and "gecmis zaman" not in message:
                    raise
                trial_end = trial_end - timedelta(days=1)

        if last_error is not None:
            raise last_error
        return []

    def _request_epias_chunk(
        self,
        spec: FinalFeatureSpec,
        chunk_start: date,
        chunk_end: date,
    ) -> list[dict[str, Any]]:
        base_url = self._base_url(spec.service)
        url = f"{base_url}{spec.endpoint_path}"
        payload = {
            "startDate": _epias_datetime(chunk_start, end_of_day=False),
            "endDate": _epias_datetime(chunk_end, end_of_day=True),
            "page": {
                "number": 1,
                "size": self.page_size,
                "sort": {"field": spec.sort_field, "direction": "ASC"},
            },
        }

        response = self._post_with_retry(url, payload)
        if not response.ok:
            sample = response.text[:500]
            self._save_debug(spec.feature_name, {"payload": payload, "status": response.status_code, "response": sample})
            raise RuntimeError(f"HTTP {response.status_code}: {sample}")

        data = response.json()
        items = extract_items(data)
        self._save_debug(spec.feature_name, data)
        return items

    def _post_with_retry(self, url: str, payload: dict[str, Any], max_retries: int = 3) -> requests.Response:
        last_response: requests.Response | None = None
        for attempt in range(max_retries + 1):
            response = self.session.post(
                url,
                json=payload,
                headers=self.client._headers(),
                timeout=self.timeout,
            )
            last_response = response
            if response.status_code != 429 and response.status_code < 500:
                return response

            wait_seconds = min(30, 5 * (2**attempt))
            print(f"  rate/server limit HTTP {response.status_code}; waiting {wait_seconds}s")
            time.sleep(wait_seconds)

        if last_response is None:
            raise RuntimeError("no response")
        return last_response

    def fetch_usd_try(self) -> FetchResult:
        output_path = RAW_DIR / "usd_try.csv"
        url = f"https://api.frankfurter.app/{self.start_date.isoformat()}..{self.end_date.isoformat()}"
        params = {"from": "USD", "to": "TRY"}

        print("[FETCH] usd_try")
        try:
            response = requests.get(url, params=params, timeout=30)
            if not response.ok:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            data = response.json()
            rows = [
                {"date": day, "usd_try": values.get("TRY")}
                for day, values in sorted(data.get("rates", {}).items())
            ]
            frame = pd.DataFrame(rows)
            if frame.empty:
                raise RuntimeError("empty USD response")
            frame.to_csv(output_path, index=False)
            return FetchResult("usd_try", True, len(frame), "frankfurter", str(output_path))
        except Exception as exc:  # noqa: BLE001
            manual_path = RAW_DIR / "usd_try_manual.csv"
            if manual_path.exists():
                data = pd.read_csv(manual_path)
                data.to_csv(output_path, index=False)
                return FetchResult("usd_try", True, len(data), "manual_csv_fallback", str(output_path))
            return FetchResult("usd_try", False, 0, "failed", str(output_path), f"{type(exc).__name__}: {exc}")

    def _base_url(self, service: str) -> str:
        if service == "natural_gas":
            return "https://seffaflik.epias.com.tr/natural-gas-service"
        return "https://seffaflik.epias.com.tr/electricity-service"

    def _save_debug(self, feature_name: str, data: Any) -> None:
        try:
            pd.Series([data]).to_json(DEBUG_DIR / f"{feature_name}.json", force_ascii=False, indent=2)
        except Exception:
            pass


def _epias_datetime(value: date, end_of_day: bool) -> str:
    clock = dt_time(23, 59, 59) if end_of_day else dt_time(0, 0, 0)
    return datetime.combine(value, clock, tzinfo=TIMEZONE).isoformat(timespec="seconds")
