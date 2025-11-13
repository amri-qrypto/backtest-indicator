# BTCUSDT Strategy Backtesting (EMA, MACD, ATR & Mean Reversion)

## Ringkasan
Proyek ini mengevaluasi berbagai strategi trading BTCUSDT harian menggunakan Python, pandas,
dan metrik performa QF-Lib. Setiap strategi menghasilkan sinyal long/cash yang kemudian
diuji melalui _vectorized backtest_ untuk memperoleh Sharpe ratio, CAGR, volatilitas tahunan,
dan statistik risiko lain.

## Fitur Utama
- 📈 **Koleksi strategi**: EMA trend following (50, 112, hasil optimasi 45), MACD crossover,
  mean reversion oversold, serta filter tren berbasis ATR.
- 🧮 **Evaluasi kuantitatif**: semua strategi dinilai dengan `qflib_metrics_from_returns`
  yang mengekstrak statistik TimeseriesAnalysis.
- 🔍 **Optimasi parameter**: notebook `ema_optimization.ipynb` menjalankan grid search
  EMA 20–200; hasil terbaik (EMA 45) disertakan dalam perbandingan final.
- 📊 **Perbandingan menyeluruh**: `strategy_comparison.ipynb` menggabungkan Sharpe,
  CAGR, drawdown, dan volatilitas dari seluruh strategi.

## Struktur Folder
```
project-root/
├─ data/
│  └─ OKX_BTCUSDT, 1D.csv
├─ notebooks/
│  ├─ ema_112.ipynb
│  ├─ ema_optimization.ipynb
│  ├─ ema_vs_price_backtest.ipynb
│  ├─ strategy_atr_filter.ipynb
│  ├─ strategy_comparison.ipynb
│  ├─ strategy_ema.ipynb
│  ├─ strategy_macd.ipynb
│  └─ strategy_oversold_mean_rev.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ backtest.py
│  ├─ data_loader.py
│  ├─ indicators.py
│  ├─ qflib_adapters.py
│  ├─ qflib_metrics.py
│  ├─ strategy.py
│  ├─ strategy_atr_filter.py
│  ├─ strategy_macd.py
│  └─ strategy_oversold.py
├─ requirements.txt
└─ README.md
```

## Instalasi
1. Pastikan Python 3.10+ terinstal.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan JupyterLab
1. Jalankan perintah berikut:
   ```bash
   jupyter lab
   ```
2. Buka notebook apa pun di folder `notebooks/` untuk mengeksplorasi strategi.

## Alur Notebook
- `ema_vs_price_backtest.ipynb`: pengantar strategi EMA dasar (EMA 50).
- `ema_112.ipynb`: grid search sekitar EMA 112 dan evaluasi performa QF-Lib.
- `ema_optimization.ipynb`: grid search luas EMA 20–200 untuk menemukan periode terbaik.
- `strategy_ema.ipynb`: contoh langkah penuh EMA trend following.
- `strategy_macd.ipynb`: sinyal MACD fast/slow + signal line.
- `strategy_oversold_mean_rev.ipynb`: pendekatan mean reversion berbasis indikator oversold.
- `strategy_atr_filter.ipynb`: filter tren dengan ATR dan median volatilitas.
- `strategy_comparison.ipynb`: final dashboard Sharpe, CAGR, drawdown, dan volatilitas dari seluruh strategi.

## Data
Letakkan file CSV historis BTCUSDT harian pada folder `data/` dengan nama `OKX_BTCUSDT, 1D.csv`.
Kolom wajib mencakup `time` (UNIX detik), `open`, `high`, `low`, `close`, serta kolom tambahan
apabila diperlukan oleh strategi (mis. sinyal oversold).

## Cara Kerja Backtest
1. Muat data melalui `load_ohlcv_csv`.
2. Hitung indikator (EMA, MACD, ATR, dsb.) dari modul `src/`.
3. Bentuk sinyal long/cash menggunakan fungsi strategi terkait.
4. Jalankan `run_backtest` untuk menghasilkan _equity curve_, _strategy return_, dan biaya trading.
5. Evaluasi performa via `qflib_metrics_from_returns` atau `performance_metrics`.
6. Visualisasikan harga, indikator, posisi, dan kurva ekuitas.

## Keterbatasan
- Tidak memasukkan slippage, likuiditas, maupun biaya trading realistis.
- Hanya mendukung posisi long (tidak ada short/leverage).
- Backtest bersifat historis; validasi out-of-sample tetap diperlukan sebelum implementasi live.
