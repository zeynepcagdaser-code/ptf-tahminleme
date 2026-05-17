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

Ortam testi:

```zsh
source venv/bin/activate
python main.py
```

Streamlit testi:

```zsh
source venv/bin/activate
streamlit hello
```

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
