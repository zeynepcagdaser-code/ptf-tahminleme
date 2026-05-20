#!/usr/bin/env python3
"""5 yıllık DL veri hattı — mevcut final ensemble'a dokunmaz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="5y EPİAŞ fetch + DL dataset + model baselines")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--skip-fetch", action="store_true", help="Fetch adimini tamamen atla")
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Mevcut CSV olsa bile API'den yeniden cek",
    )
    parser.add_argument("--build-only", action="store_true", help="Fetch ve train atla")
    parser.add_argument("--train-only", action="store_true", help="NPZ hazırsa yalnız model eğit")
    parser.add_argument(
        "--use-fallback-raw",
        action="store_true",
        help="(eski) Dosya bazli cozumleme artik varsayilan; bu bayrak istege bagli log icin",
    )
    parser.add_argument("--quick-train", action="store_true", default=True)
    parser.add_argument(
        "--only-features",
        type=str,
        default="",
        help="Virgülle ayrılmış seri adları (örn. dam_supply_demand,idm_trade_value)",
    )
    args = parser.parse_args()

    if args.only_features.strip():
        from src.fetch_epias_5y import run_fetch_epias_5y_features

        names = [x.strip() for x in args.only_features.split(",") if x.strip()]
        print(f"=== Seçili EPİAŞ serileri: {names} ===")
        run_fetch_epias_5y_features(names, force_refetch=True)
        return

    if args.use_fallback_raw:
        from src.build_hourly_dataset_5y import enable_fallback_raw_dir

        enable_fallback_raw_dir()

    if not args.skip_fetch and not args.build_only and not args.train_only:
        from src.fetch_epias_5y import run_fetch_epias_5y

        print("=== 1/4 EPİAŞ 5y fetch (mevcut dosyalar atlanir) ===")
        stats = run_fetch_epias_5y(force_refetch=args.force_refetch)
        print(
            f"   OK={stats['successful']}/{stats['attempted']} "
            f"(atlanan={stats.get('skipped_existing', 0)}, api={stats.get('fetched_api', 0)}) "
            f"-> {stats['raw_dir']}"
        )
        if args.fetch_only:
            return

    if not args.train_only:
        from src.build_hourly_dataset_5y import build_hourly_dataset_5y
        from src.build_dl_sequence_dataset_5y import build_dl_sequence_dataset_5y

        print("=== 2/4 Saatlik panel + feature engineering ===")
        df, quality = build_hourly_dataset_5y()
        print(f"   {len(df):,} satır -> {quality['output_path']}")
        print(f"   Eksik saat (reindex öncesi): {quality['missing_hours_before_reindex']:,}")

        print("=== 3/4 Sequence dataset (168h -> 12h) ===")
        seq = build_dl_sequence_dataset_5y()
        print(f"   {seq['n_samples']:,} örnek x {seq['n_features']} feature")
        print(f"   train/val/test = {seq['train_samples']}/{seq['val_samples']}/{seq['test_samples']}")

        if args.build_only:
            return

    print("=== 4/4 DL baselines (GRU/LSTM/CNN-LSTM/PatchTST) ===")
    from src.train_dl_models_5y import train_dl_baselines_5y

    report = train_dl_baselines_5y(quick=args.quick_train)
    print(f"\nEn iyi model: {report.get('best_model')}")
    print(f"Metrikler: data/processed/dl_models_metrics_5y.json")


if __name__ == "__main__":
    main()
