from __future__ import annotations

import argparse

from src.run_ptf_pipeline import run_full_ptf_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="PTF 12 saatlik tahmin pipeline")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="EPİAŞ'tan canli veri cek (kimlik bilgisi gerekir)",
    )
    args = parser.parse_args()

    try:
        result = run_full_ptf_pipeline(fetch_live=args.fetch)

        print("\nPTF 12 Saatlik Tahmin Pipeline")
        print("-" * 56)
        print(f"Egitim ornek sayisi : {result['dataset_rows']:,}")
        print(f"Ensemble MAE          : {result['mae']:,.2f} TL/MWh")
        print(f"Ensemble RMSE         : {result['rmse']:,.2f} TL/MWh")
        print(f"Ensemble SMAPE        : {result['smape']:,.2f}%")
        print(f"Ensemble R2           : {result['r2']:,.4f}")
        print(f"CatBoost MAE          : {result['catboost_mae']:,.2f} TL/MWh")
        print(f"CatBoost R2           : {result['catboost_r2']:,.4f}")
        print(f"MAE iyilestirme       : {result['mae_improvement_pct']:,.1f}%")
        print(f"Ufuk metrikleri       : {result['horizon_metrics_path']}")
        print("\nCiktilar")
        print(f"- {result['dataset_path']}")
        print(f"- {result['predictions_path']}")
        print(f"- {result['live_forecast_path']}")
        print("\nPanel: streamlit run streamlit_app.py")
    except Exception as exc:
        print("\nPipeline hatasi")
        print("-" * 56)
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
