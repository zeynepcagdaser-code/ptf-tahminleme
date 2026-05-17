from __future__ import annotations

from src.build_d_plus_2_forecast_datasets import build_d_plus_2_forecast_datasets
from src.train_d_plus_2_catboost import train_d_plus_2_catboost_models


def main() -> None:
    try:
        before, after, dataset_summary = build_d_plus_2_forecast_datasets()
        training_summary = train_d_plus_2_catboost_models()

        print("\nD+2 PTF forecast dataset ozeti")
        print("-" * 56)
        print(f"Forecast issue date araligi : {dataset_summary.first_forecast_issue_date} -> {dataset_summary.last_forecast_issue_date}")
        print(f"Before 14 dataset shape     : {before.shape}")
        print(f"After 14 dataset shape      : {after.shape}")
        print(f"Before 14 path              : {dataset_summary.before_path}")
        print(f"After 14 path               : {dataset_summary.after_path}")

        print("\nCatBoost D+2 performans karsilastirmasi")
        print("-" * 56)
        for metrics in [training_summary.before_metrics, training_summary.after_metrics]:
            print(f"\nSenaryo: {metrics['scenario']}")
            print(f"Rows       : {metrics['rows']:,}")
            print(f"Features   : {metrics['feature_count']:,}")
            print(f"Train rows : {metrics['train_rows']:,}")
            print(f"Test rows  : {metrics['test_rows']:,}")
            print(f"MAE        : {metrics['MAE']:,.4f}")
            print(f"RMSE       : {metrics['RMSE']:,.4f}")
            print(f"SMAPE      : {metrics['SMAPE']:,.4f}%")
            print(f"R2         : {metrics['R2']:,.6f}")

        print("\nCiktilar")
        print(f"- {dataset_summary.before_path}")
        print(f"- {dataset_summary.after_path}")
        print(f"- {training_summary.before_predictions_path}")
        print(f"- {training_summary.after_predictions_path}")
        print(f"- {training_summary.comparison_path}")
    except Exception as exc:
        print("\nD+2 forecast pipeline hatasi")
        print("-" * 56)
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
