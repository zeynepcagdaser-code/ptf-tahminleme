# 5 Yıllık Deep Learning Pipeline — Teknik Plan

## Kapsam
- Tarih: `2020-01-01` → bugün (Europe/Istanbul), saatlik panel
- Mevcut `final_hourly_dataset.csv` / `main.py` ensemble **dokunulmaz**
- Tüm çıktılar `*_5y` suffix ile ayrı dosyalarda

## Aşamalar

### 1. EPİAŞ fetch (`fetch_epias_5y.py`)
- Ham veri: `data/raw/epias_5y/`
- PTF/MCP, I-MCP, yük planı, tüketim, gerçek zamanlı üretim (kaynak kırılımı), SMF, RES/GES tahmin, YEKDEM, sistem yönü, USD/TRY
- I-MCP: günlük chunk; diğerleri: aylık chunk + backoff

### 2. Saatlik panel + feature engineering (`build_hourly_dataset_5y.py`)
- Eksiksiz saatlik index, duplicate drop
- `net_load`, `renewable_*`, `thermal_*`, lag/rolling, takvim, bayram bayrakları
- `spike_flag` (geçmiş rolling 168h — leakage yok)
- Kalite raporu: eksik saatler, NaN oranları, doldurma stratejisi

### 3. Sequence dataset (`build_dl_sequence_dataset_5y.py`)
- `input_window=168`, `output_horizon=12`
- Kronolojik train/val/test; scaler **yalnız train**
- `forecast_12h_sequence_dataset_5y.npz` + tabular CSV + `scalers_5y.pkl`

### 4. DL baselines (`train_dl_models_5y.py`)
- GRU → LSTM → CNN-LSTM → (opsiyonel) küçük Transformer
- Hızlı epoch + early stopping; en iyi model seçimi

## Çalıştırma
```bash
python scripts/run_5y_dl_pipeline.py              # tam
python scripts/run_5y_dl_pipeline.py --skip-fetch   # ham CSV hazırsa
python scripts/run_5y_dl_pipeline.py --train-only   # npz hazırsa
```

Fetch 5 yıl için uzun sürebilir (EPİAŞ rate limit); kimlik bilgisi `.env` gerekli.
