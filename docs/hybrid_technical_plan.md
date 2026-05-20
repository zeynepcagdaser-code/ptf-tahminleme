# Hibrit PTF Tahmin — Teknik Plan

## Amaç
CatBoost + mevsimsel baseline + GRU + spike classifier birleşimi; mevcut `ptf_12h_final_*` çıktılarına dokunulmaz.

## Bileşenler
1. **Spike classifier** (`train_spike_classifier.py`): HistGradientBoosting; hedef = rolling 168h median + 2σ veya üst %90 quantile; çıktı `spike_probability`.
2. **GRU** (`train_lstm_hybrid.py`): 24h pencere → 12h; düşük epoch + early stopping; CPU.
3. **Hibrit ensemble** (`build_hybrid_forecast.py`): validation grid ile ağırlıklar; spike ile genlik artışı.

## Formül
- `core = w_cb·CatBoost + w_seas·Seasonal + w_gru·GRU` (ramp-down’da GRU↑ CatBoost↓)
- `hybrid_raw = seasonal + (core − seasonal) × (1 + spike_prob × 0.45)`

## Volatility calibration (`hybrid_volatility_calibration.py`)
- Rejim: `rolling_24_std`, `rolling_168_std`, `ramp_abs_1h/3h`, `intraday_range`
- Clip band: `lower=rolling_24_q10`, `upper=rolling_24_q95×1.15`
- `spike_prob < 0.35`: dip çekme (baseline’a %55)
- `spike_prob > 0.7`: dinamik genlik + overshoot tavanı
- Orta rejim: ramp-down → GRU blend

## Çıktılar
- `ptf_12h_hybrid_predictions.csv`, `ptf_12h_hybrid_metrics.json`
- Ara: `ptf_12h_spike_*`, `ptf_12h_hybrid_gru_*`

## Çalıştırma
```bash
python scripts/run_hybrid_pipeline.py
```
