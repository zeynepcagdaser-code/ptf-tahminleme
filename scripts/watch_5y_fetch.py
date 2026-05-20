#!/usr/bin/env python3
"""5y EPİAŞ fetch canlı izleme — log tamponlu olsa da dosya/envanter üzerinden güncellenir."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.epias_5y_panel import (  # noqa: E402
    FETCH_LIVE_APP_PATH,
    FETCH_PROGRESS_PATH,
    build_epias_5y_inventory,
    get_live_fetch_snapshot,
    inventory_summary_stats,
    is_fetch_process_running,
    sync_fetch_live_to_app_data,
)


def _clear() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def render_once() -> None:
    snap = get_live_fetch_snapshot()
    inv = snap["inventory"]
    stats = snap["stats"]
    prog = snap["progress"]
    running = snap["running"]

    print("=" * 72)
    print("  EPİAŞ 5Y FETCH İZLEME")
    print("=" * 72)
    if running:
        print("  Durum: ÇALIŞIYOR")
    else:
        print("  Durum: DURDU (veya henüz başlamadı)")
    if prog:
        cur = prog.get("current") or "—"
        idx = prog.get("index", 0)
        tot = prog.get("total", 0)
        print(f"  İlerleme: {idx}/{tot}  |  Şu an: {cur}")
        if prog.get("newest_file"):
            print(f"  Son güncellenen dosya: {prog['newest_file']} ({prog.get('newest_file_mtime', '')})")
        print(f"  Güncelleme: {prog.get('updated_at', '—')}")
    print(
        f"  Envanter: tam5y={stats['full']}  kısmi={stats['partial']}  "
        f"kısa={stats['short']}  eksik={stats['missing']} / {stats['total']}"
    )
    print(f"  Progress: {FETCH_PROGRESS_PATH}")
    print(f"  GitHub özet: {FETCH_LIVE_APP_PATH}")
    print("-" * 72)

    if inv.empty:
        print("  Henüz veri yok.")
        return

    show = inv[["Seri", "Satır", "İlk tarih", "Kapsam %", "Durum"]].copy()
    show["İlk tarih"] = show["İlk tarih"].astype(str).str.slice(0, 10)
    for _, row in show.iterrows():
        mark = ">>>" if running and prog.get("newest_file", "").replace(".csv", "") in str(row["Seri"]).lower() else "   "
        print(
            f"{mark} {row['Durum']:12} {row['Kapsam %']:5.1f}%  "
            f"{int(row['Satır']):>8,}  {row['İlk tarih']:10}  {row['Seri']}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="5y fetch canlı izleme")
    parser.add_argument("--once", action="store_true", help="Tek seferlik özet")
    parser.add_argument("--interval", type=float, default=5.0, help="Yenileme saniyesi")
    parser.add_argument(
        "--push-github",
        action="store_true",
        help="Her döngüde app_data/epias_5y_fetch_live.json dosyasını GitHub'a push et",
    )
    args = parser.parse_args()

    if args.once:
        render_once()
        return

    try:
        while True:
            _clear()
            render_once()
            if not is_fetch_process_running():
                print("\nFetch süreci yok. Çıkmak için Ctrl+C.")
                break
            sync_fetch_live_to_app_data()
            if args.push_github:
                subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "scripts" / "push_fetch_live_to_github.py")],
                    cwd=PROJECT_ROOT,
                    check=False,
                )
            print(f"\n{args.interval:.0f} sn sonra yenilenir (Ctrl+C çıkış)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nİzleme durduruldu.")


if __name__ == "__main__":
    main()
