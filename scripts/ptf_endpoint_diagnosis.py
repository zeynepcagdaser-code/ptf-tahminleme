from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import PROJECT_ROOT, get_settings
from src.epias_client import EpiasClient, extract_items

BASE = 'https://seffaflik.epias.com.tr/electricity-service'
LOG_CSV = PROJECT_ROOT / 'logs' / 'ptf_endpoint_diagnosis.csv'
LOG_JSON = PROJECT_ROOT / 'logs' / 'ptf_endpoint_diagnosis.json'
DOC_URL = 'https://seffaflik.epias.com.tr/electricity-service/technical/tr/index.html'

DATE_FORMATS = [
    ('iso_colon', '2026-05-15T00:00:00+03:00', '2026-05-15T23:59:59+03:00'),
    ('millis_no_colon', '2026-05-15T00:00:00.000+0300', '2026-05-15T23:59:59.000+0300'),
    ('iso_no_colon', '2026-05-15T00:00:00+0300', '2026-05-15T23:59:59+0300'),
    ('space', '2026-05-15 00:00:00', '2026-05-15 23:59:59'),
    ('date_only', '2026-05-15', '2026-05-15'),
]

POST_ENDPOINTS = [
    ('mcp', '/v1/markets/dam/data/mcp'),
    ('interim_mcp', '/v1/markets/dam/data/interim-mcp'),
    ('kptf_guess_1', '/v1/markets/dam/data/final-mcp'),
    ('kptf_guess_2', '/v1/markets/dam/data/kptf'),
]
GET_ENDPOINTS = [
    ('interim_mcp_published_status', '/v1/markets/dam/data/interim-mcp-published-status'),
]


def main() -> None:
    settings = get_settings()
    client = EpiasClient(settings)
    headers = client._headers()
    rows: list[dict[str, Any]] = []

    doc_paths = discover_doc_paths()
    rows.append({
        'test_name': 'technical_doc_discovery',
        'endpoint': DOC_URL,
        'method': 'GET',
        'date_format': '',
        'request_body': '',
        'status_code': 200,
        'response_keys': ','.join(doc_paths),
        'row_count': len(doc_paths),
        'last_datetime': '',
        'first_5_rows': json.dumps(doc_paths[:5], ensure_ascii=False),
        'error': '',
    })

    for endpoint_name, endpoint_path in POST_ENDPOINTS:
        for fmt_name, start_date, end_date in DATE_FORMATS:
            payload = {
                'startDate': start_date,
                'endDate': end_date,
                'page': {
                    'number': 1,
                    'size': 100,
                    'sort': {'field': 'date', 'direction': 'ASC'},
                },
            }
            rows.append(test_post(headers, endpoint_name, endpoint_path, fmt_name, payload))

    for endpoint_name, endpoint_path in GET_ENDPOINTS:
        rows.append(test_get(headers, endpoint_name, endpoint_path))
        # Some GET-like endpoints in EPİAŞ docs still expect POST body. Test that too.
        payload = {'date': '2026-05-15'}
        rows.append(test_post(headers, endpoint_name + '_as_post_date', endpoint_path, 'date_payload', payload))

    frame = pd.DataFrame(rows)
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(LOG_CSV, index=False)
    LOG_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(frame[['test_name','endpoint','method','date_format','status_code','row_count','last_datetime','error']].to_string(index=False))
    print('\nWrote')
    print(LOG_CSV)
    print(LOG_JSON)


def discover_doc_paths() -> list[str]:
    try:
        html = requests.get(DOC_URL, timeout=30).text
    except Exception as exc:
        return [f'doc_fetch_error:{type(exc).__name__}:{exc}']
    paths = sorted(set(re.findall(r'/v1/[a-zA-Z0-9_./{}?=&:-]+', html)))
    filtered = [p for p in paths if any(term in p.lower() for term in ['mcp', 'ptf', 'kptf', 'interim'])]
    # Also search visible text for nearby snippets when no paths contain Turkish KPTF.
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text('\n')
    snippets = []
    for term in ['PTF', 'KPTF', 'Kesinleşmemiş', 'MCP', 'interim']:
        idx = text.lower().find(term.lower())
        if idx >= 0:
            snippets.append('snippet:' + text[max(0, idx-120):idx+240].replace('\n', ' ')[:500])
    return filtered + snippets[:8]


def test_post(headers: dict[str, str], endpoint_name: str, endpoint_path: str, fmt_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = BASE + endpoint_path
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        return parse_response(endpoint_name, endpoint_path, 'POST', fmt_name, payload, response)
    except Exception as exc:
        return base_row(endpoint_name, endpoint_path, 'POST', fmt_name, payload, None, f'{type(exc).__name__}: {exc}')


def test_get(headers: dict[str, str], endpoint_name: str, endpoint_path: str) -> dict[str, Any]:
    url = BASE + endpoint_path
    try:
        response = requests.get(url, headers=headers, timeout=45)
        return parse_response(endpoint_name, endpoint_path, 'GET', '', {}, response)
    except Exception as exc:
        return base_row(endpoint_name, endpoint_path, 'GET', '', {}, None, f'{type(exc).__name__}: {exc}')


def parse_response(endpoint_name: str, endpoint_path: str, method: str, fmt_name: str, payload: dict[str, Any], response: requests.Response) -> dict[str, Any]:
    error = ''
    response_keys = ''
    row_count = 0
    last_datetime = ''
    first_5_rows = ''
    try:
        data = response.json()
        response_keys = keys_summary(data)
        try:
            items = extract_items(data)
        except Exception:
            items = []
        row_count = len(items)
        if items:
            first_5_rows = json.dumps(items[:5], ensure_ascii=False, default=str)
            last_datetime = detect_last_datetime(items)
    except Exception as exc:
        data = response.text[:1000]
        first_5_rows = str(data)[:1000]
        error = f'parse_error:{type(exc).__name__}:{exc}'
    if not response.ok and not error:
        error = response.text[:1000]
    return base_row(endpoint_name, endpoint_path, method, fmt_name, payload, response.status_code, error, response_keys, row_count, last_datetime, first_5_rows)


def base_row(test_name: str, endpoint_path: str, method: str, fmt_name: str, payload: dict[str, Any], status_code: int | None, error: str, response_keys: str = '', row_count: int = 0, last_datetime: str = '', first_5_rows: str = '') -> dict[str, Any]:
    return {
        'test_name': test_name,
        'endpoint': endpoint_path,
        'method': method,
        'date_format': fmt_name,
        'request_body': json.dumps(payload, ensure_ascii=False),
        'status_code': status_code,
        'response_keys': response_keys,
        'row_count': row_count,
        'last_datetime': last_datetime,
        'first_5_rows': first_5_rows,
        'error': error,
    }


def keys_summary(value: Any, prefix: str = '') -> str:
    keys = []
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            keys.append(path)
            if isinstance(nested, dict):
                keys.extend(keys_summary(nested, path).split(','))
    return ','.join(k for k in keys if k)


def detect_last_datetime(items: list[dict[str, Any]]) -> str:
    candidates = []
    for item in items:
        for key in ['date', 'datetime', 'deliveryDay', 'period']:
            if key in item:
                parsed = pd.to_datetime(item[key], errors='coerce', utc=True)
                if not pd.isna(parsed):
                    candidates.append(parsed.tz_convert('Europe/Istanbul').tz_localize(None))
    if not candidates:
        return ''
    return max(candidates).strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    main()
