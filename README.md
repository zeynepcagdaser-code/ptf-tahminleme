# PTF Tahmin Uygulamasi

EPİAŞ verileri ile çalışan makine öğrenmesi ve derin öğrenme tabanlı PTF tahmin sistemi için Python 3.11 geliştirme ortamı.

## Ortam Kurulumu

Homebrew yolunu aktif hale getirin:

```zsh
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Python 3.11 kurulu değilse:

```zsh
brew install python@3.11
python3.11 --version
```

## Projeye Giris

```zsh
cd ~/ptf_tahmin_uygulamasi
```

## Virtual Environment

Sanal ortami olusturmak:

```zsh
python3.11 -m venv venv
```

Sanal ortami aktif etmek:

```zsh
source venv/bin/activate
```

Terminal basinda `(venv)` gorunuyorsa ortam aktiftir.

Pip guncelleme:

```zsh
pip install --upgrade pip
```

Paketleri kurma:

```zsh
pip install -r requirements.txt
```

`requirements.txt` Streamlit panelinin bulutta hızlı kurulması için hafif tutulur.
Model eğitimi ve notebook çalışmaları için tam ortamı kurmak isterseniz:

```zsh
pip install -r requirements-full.txt
```

Apple Silicon icin bu ortamda TensorFlow Metal destegi kullanilir. `tensorflow-metal` en yeni `tensorflow` surumuyle uyumsuzluk gosterebildigi icin ortam calisan surume sabitlenmistir:

```zsh
pip install "tensorflow==2.19.0" "tensorboard==2.19.0" tensorflow-metal
```

## VS Code Interpreter Secimi

1. VS Code ile proje klasorunu acin:

```zsh
code ~/ptf_tahmin_uygulamasi
```

2. `Cmd+Shift+P` tuslarina basin.
3. `Python: Select Interpreter` yazin.
4. Proje icindeki interpreter'i secin:

```text
./venv/bin/python
```

## Gerekli VS Code Extensionlari

- Python
- Pylance
- Jupyter

## Calistirma

Tam pipeline (veri seti + delta CatBoost + ensemble + canli snapshot):

```zsh
source venv/bin/activate
python main.py
```

EPİAŞ'tan canli veri cekerek:

```zsh
python main.py --fetch
```

Streamlit testi:

```zsh
source venv/bin/activate
streamlit run streamlit_app.py
```

## Aktif mimari (12 saat PTF)

```
EPİAŞ (I-MCP + MCP) -> final_hourly_dataset.csv
  -> forecast_12h_dataset.csv (hedef: kesinlesmis, delta: kesin - interim)
  -> CatBoost (12 ufuk, delta hedef)
  -> build_final_ensemble.py (mevsimsel baseline + CatBoost)
  -> ptf_12h_live_bundle.csv + app_data/dashboard_data.json
```

Panel (`streamlit_app.py`) final ensemble metriklerini ve canli karsilastirmayi (I-MCP / naive / CatBoost / ensemble) gosterir.

## Legacy moduller

Asagidaki dosyalar **ana pipeline'a bagli degildir** (D+2, 24h, deneme LSTM vb.). Liste: `docs/legacy_modules.md`.

## 5Y fetch — panel + GitHub canlı özet

Yerel çekim sırasında Streamlit paneli **5Y EPİAŞ Verileri** sekmesinde ilerlemeyi gösterir.
Streamlit Cloud ham CSV okuyamaz; `app_data/epias_5y_fetch_live.json` push edilir:

```zsh
python scripts/push_fetch_live_to_github.py
# veya çekim + otomatik push:
python scripts/watch_5y_fetch.py --push-github
```

## Streamlit Cloud Deploy

Streamlit Community Cloud'da Python surumu repo icindeki `runtime.txt` ile degil,
uygulama olusturulurken `Advanced settings` ekranindan secilir. Bu proje icin
Python `3.11` secilmelidir.

Mevcut app yanlis Python surumuyle olustuysa:

1. Streamlit Cloud'da mevcut app ayarlarini ve Secrets degerlerini not alin.
2. App'i silin.
3. Ayni GitHub repo ile yeniden deploy edin:

```text
Repository: zeynepcagdaser-code/ptf-tahminleme
Branch: main
Main file path: streamlit_app.py
Python version: 3.11
```

4. Secrets alanina EPİAŞ bilgilerini tekrar ekleyin.

## 5 Yillik Deep Learning Pipeline (ayri hat)

Mevcut `python main.py` final ensemble'a **dokunmaz**. Tum ciktilar `*_5y` dosya adlariyla tutulur.

```zsh
# Tam hat (2020-01-01 -> bugun EPİAŞ fetch — uzun surer)
python scripts/run_5y_dl_pipeline.py

# Ham CSV hazirsa
python scripts/run_5y_dl_pipeline.py --skip-fetch

# Gecici test: mevcut final_selected_features ile
python scripts/run_5y_dl_pipeline.py --skip-fetch --use-fallback-raw
```

Ciktilar: `data/processed/final_hourly_dataset_5y.csv`, `forecast_12h_sequence_dataset_5y.npz`, `data_quality_report_5y.json`, `data/models/scalers_5y.pkl`, `dl_models_metrics_5y.json`

Teknik plan: `docs/dl_5y_technical_plan.md`

## Hata Cozumleri

`python3.11: command not found` hatasi:

```zsh
eval "$(/opt/homebrew/bin/brew shellenv)"
brew install python@3.11
```

`pip: command not found` hatasi:

```zsh
source venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

Apple Silicon TensorFlow Metal kontrolu:

```zsh
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

PyTorch MPS kontrolu:

```zsh
python -c "import torch; print(torch.backends.mps.is_available())"
```
