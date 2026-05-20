#!/usr/bin/env python3
"""Hibrit pipeline — mevcut final ensemble dosyalarina dokunmaz."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.build_hybrid_forecast import build_hybrid_forecast
    from src.train_lstm_hybrid import train_lstm_hybrid
    from src.train_spike_classifier import train_spike_classifier

    print("1/3 Spike classifier...")
    spike = train_spike_classifier()
    print(f"   spike rate={spike.spike_rate:.2%} test_auc={spike.test_auc:.3f}")

    print("2/3 Hybrid GRU (24h window)...")
    gru = train_lstm_hybrid(input_window_hours=24, max_epochs=10, patience=3)
    print(f"   GRU MAE={gru.mae:.1f} R2={gru.r2:.3f}")

    print("3/3 Hybrid ensemble + volatility calibration...")
    hybrid = build_hybrid_forecast()
    print(f"   Ham MAE={hybrid.raw_mae:.1f} -> Kalibre MAE={hybrid.overall_mae:.1f} R2={hybrid.overall_r2:.3f}")
    print(f"   Spike MAE={hybrid.spike_hours_mae:.1f} | Non-spike MAE={hybrid.non_spike_hours_mae:.1f}")
    print(f"   Yon dogrulugu={hybrid.direction_accuracy:.1%} | Trend corr={hybrid.trend_correlation:.3f}")
    print(f"\nCikti: {hybrid.predictions_path}")
    print(f"Grafik: {hybrid.calibration_chart_path}")


if __name__ == "__main__":
    main()
