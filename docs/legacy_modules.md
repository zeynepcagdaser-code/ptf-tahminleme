# Legacy moduller

Bu dosyalar repoda duruyor ancak **`python main.py` pipeline'ina dahil degildir**.
Aktif urun: 12 saatlik kesinlesmis PTF tahmini (I-MCP kesiminden sonra).

## D+2 tahmin (eski tez yonu)

| Dosya | Aciklama |
|-------|----------|
| `src/build_d_plus_2_forecast_datasets.py` | D+2 veri setleri |
| `src/train_d_plus_2_catboost.py` | D+2 CatBoost |
| `src/train_d_plus_2_lstm.py` | D+2 LSTM |
| `src/export_d_plus_2_metrics.py` | D+2 metrik export |

## 24 saat / diger ML denemeleri

| Dosya | Aciklama |
|-------|----------|
| `src/build_24h_forecast_dataset.py` | 24h veri seti |
| `src/train_catboost_24h_forecast.py` | 24h CatBoost |
| `src/train_catboost_final.py` | Eski final CatBoost |
| `src/train_multiple_models.py` | Coklu model karsilastirma |
| `src/train_baseline_xgboost.py` | XGBoost baseline |
| `src/feature_selection_and_retrain.py` | Ozellik secimi |
| `src/feature_engineering.py` | Eski feature engineering |
| `src/feature_engineering_final_dataset.py` | Final dataset FE raporu |

## 12h deneme modelleri (ensemble disinda)

| Dosya | Aciklama |
|-------|----------|
| `src/train_ptf_12h_forecast_enhanced.py` | Genis HP aramali CatBoost |
| `src/train_ptf_12h_lstm.py` | LSTM/GRU/CNN denemeleri |
| `data/processed/ptf_12h_lstm_runs/` | LSTM run ciktilari |

## Diger

| Dosya | Aciklama |
|-------|----------|
| `src/fetch_selected_epias_features.py` | Eski EPİAŞ fetch |
| `src/standardize_selected_features.py` | Eski standardizasyon |
| `src/analyze_ptf.py` | Analiz scripti |
| `scripts/ptf_endpoint_diagnosis.py` | EPİAŞ endpoint teshis |

## Aktif pipeline dosyalari

- `main.py` / `src/run_ptf_pipeline.py`
- `src/fetch_final_selected_features.py`
- `src/build_final_hourly_dataset.py`
- `src/build_12h_forecast_dataset.py`
- `src/train_ptf_12h_forecast.py`
- `src/build_final_ensemble.py`
- `src/live_forecast.py`
- `scripts/update_dashboard_snapshot.py`
- `streamlit_app.py`
