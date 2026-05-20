#!/usr/bin/env python3
"""5y veri ile en iyi PTF tahmini: CatBoost + ensemble + hybrid."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build_best_ptf_pipeline import run_best_ptf_pipeline


def main() -> None:
    summary = run_best_ptf_pipeline(include_hybrid=True, include_spike_gru=True)
    print("\n" + "=" * 56)
    print("EN İYİ PTF TAHMİNİ (5 yıl veri)")
    print("=" * 56)
    print(f"CatBoost MAE   : {summary.catboost_mae:,.1f} TL/MWh")
    print(f"Ensemble MAE   : {summary.ensemble_mae:,.1f} TL/MWh")
    print(f"Hybrid MAE     : {summary.hybrid_mae:,.1f} TL/MWh")
    print(f"Seçilen        : {summary.best_track.upper()} (MAE {summary.best_mae:,.1f})")
    print(f"Tahminler      : {summary.predictions_path}")
    print(f"Özet           : {summary.metrics_path}")


if __name__ == "__main__":
    main()
