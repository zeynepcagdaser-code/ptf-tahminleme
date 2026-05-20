#!/usr/bin/env python3
"""Ham 5y CSV'leri tam 43.800 saatlik ızgaraya getirir."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dl_5y_config import HOURS_5Y
from src.epias_5y_normalize import normalize_all_epias_5y_raw


def main() -> None:
    rows = normalize_all_epias_5y_raw()
    ok = sum(1 for n in rows.values() if n == HOURS_5Y)
    print(f"Hedef: {HOURS_5Y:,} saatlik satır/seri")
    for name, n in sorted(rows.items()):
        mark = "OK" if n == HOURS_5Y else ("—" if n == 0 else f"{n:,}")
        print(f"  {mark:>8}  {name}")
    print(f"\nTam 43.800: {ok}/{len(rows)} seri")


if __name__ == "__main__":
    main()
