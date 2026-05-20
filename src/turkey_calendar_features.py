from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


# Sabit resmi tatiller (ay-gün)
FIXED_HOLIDAYS_MM_DD = {
    (1, 1),
    (4, 23),
    (5, 1),
    (5, 19),
    (7, 15),
    (8, 30),
    (10, 29),
}

# Dini bayramlar (yaklaşık — resmi takvim kayması olabilir)
RAMADAN_BAYRAM = {
    2020: [(date(2020, 5, 24), date(2020, 5, 26))],
    2021: [(date(2021, 5, 13), date(2021, 5, 15))],
    2022: [(date(2022, 5, 2), date(2022, 5, 4))],
    2023: [(date(2023, 4, 21), date(2023, 4, 23))],
    2024: [(date(2024, 4, 10), date(2024, 4, 12))],
    2025: [(date(2025, 3, 30), date(2025, 4, 1))],
    2026: [(date(2026, 3, 20), date(2026, 3, 22))],
}

KURBAN_BAYRAM = {
    2020: [(date(2020, 7, 31), date(2020, 8, 3))],
    2021: [(date(2021, 7, 20), date(2021, 7, 23))],
    2022: [(date(2022, 7, 9), date(2022, 7, 12))],
    2023: [(date(2023, 6, 28), date(2023, 7, 1))],
    2024: [(date(2024, 6, 16), date(2024, 6, 19))],
    2025: [(date(2025, 6, 6), date(2025, 6, 9))],
    2026: [(date(2026, 5, 27), date(2026, 5, 30))],
}


def _in_ranges(d: date, ranges: list[tuple[date, date]]) -> bool:
    return any(start <= d <= end for start, end in ranges)


def attach_turkey_calendar(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12)
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    cal_date = dt.dt.date
    out["is_holiday"] = cal_date.map(
        lambda d: (d.month, d.day) in FIXED_HOLIDAYS_MM_DD
    ).astype(int)

    out["is_ramadan_bayram"] = cal_date.map(
        lambda d: _in_ranges(d, RAMADAN_BAYRAM.get(d.year, []))
    ).astype(int)
    out["is_kurban_bayram"] = cal_date.map(
        lambda d: _in_ranges(d, KURBAN_BAYRAM.get(d.year, []))
    ).astype(int)

    out["month_start"] = (dt.dt.day <= 3).astype(int)
    out["month_end"] = (dt.dt.day >= (dt + pd.offsets.MonthEnd(0)).dt.day - 2).astype(int)
    return out
