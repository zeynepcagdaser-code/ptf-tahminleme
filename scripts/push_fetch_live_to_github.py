#!/usr/bin/env python3
"""Canlı fetch özetini GitHub'a gönder — Streamlit Cloud paneli güncellenir."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.epias_5y_panel import FETCH_LIVE_APP_PATH, sync_fetch_live_to_app_data  # noqa: E402


def main() -> None:
    path = sync_fetch_live_to_app_data()
    rel = path.relative_to(PROJECT_ROOT)
    subprocess.run(["git", "add", str(rel)], cwd=PROJECT_ROOT, check=True)
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
    if status.returncode == 0:
        print("Değişiklik yok — push atlandı.")
        return
    msg = "chore: EPİAŞ 5y fetch canlı durum özeti (Streamlit panel)"
    subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
    subprocess.run(["git", "push", "origin", "main"], cwd=PROJECT_ROOT, check=True)
    print(f"GitHub güncellendi: {rel}")


if __name__ == "__main__":
    main()
