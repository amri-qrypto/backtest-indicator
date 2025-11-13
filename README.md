# BTCUSDT EMA vs Price Backtest (QF-Lib + Python)

## Deskripsi
Proyek ini menampilkan backtest strategi sederhana berbasis hubungan harga penutupan BTCUSDT dan Exponential Moving Average (EMA). Ketika harga penutupan berada di atas EMA, strategi mengambil posisi long. Jika harga penutupan di bawah EMA, strategi beralih ke posisi kas. Backtest dilakukan menggunakan Python, pandas, dan adaptor ringan ke struktur data QF-Lib.

## Struktur Folder
```
project-root/
├─ data/
│  └─ OKX_BTCUSDT, 1D.csv
├─ notebooks/
│  ├─ ema_vs_price_backtest.ipynb
│  ├─ strategy_ema.ipynb
│  ├─ strategy_macd.ipynb
│  ├─ strategy_oversold_mean_rev.ipynb
│  ├─ strategy_atr_filter.ipynb
│  └─ strategy_comparison.ipynb
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
2. Buka notebook `notebooks/ema_vs_price_backtest.ipynb` melalui antarmuka JupyterLab.

## Data
Letakkan file CSV historis BTCUSDT harian pada folder `data/` dengan nama `OKX_BTCUSDT, 1D.csv`. Kolom wajib mencakup `time` (UNIX detik), `open`, `high`, `low`, `close`, serta kolom EMA (opsional jika hendak dihitung di notebook).

## Cara Menggunakan Notebook
Notebook `ema_vs_price_backtest.ipynb` memandu langkah-langkah berikut:
1. Memuat data menggunakan `load_ohlcv_csv`.
2. Menampilkan informasi data awal.
3. Menghitung EMA (periode default 50, dapat diubah pada sel terkait).
4. Menghasilkan sinyal strategi melalui `ema_vs_price_signals`.
5. Menjalankan backtest dengan `run_backtest`.
6. Menghitung metrik performa menggunakan `performance_metrics`.
7. Membuat plot harga + EMA dan kurva ekuitas.

## 🔬 Multi-Strategy Analysis

Project ini sekarang mendukung beberapa strategi tambahan:
- EMA Trend Following
- MACD Cross
- Oversold Mean Reversion
- ATR Volatility Filter

Setiap strategi memiliki notebook masing-masing di `/notebooks/`.

Notebook final `strategy_comparison.ipynb` membandingkan:
- Sharpe Ratio
- CAGR
- Max Drawdown
- Volatility

Menggunakan QF-Lib TimeseriesAnalysis.

## Strategi
- **Long** ketika `close > EMA`.
- **Cash** (tidak ada posisi) ketika `close < EMA`.
- Transisi posisi dapat dikonfigurasi dengan biaya transaksi (basis poin) pada fungsi `run_backtest`.

## Metrik Output
Fungsi `performance_metrics` menghasilkan:
- `total_return`
- `annualized_return`
- `annualized_volatility`
- `sharpe_ratio`
- `max_drawdown`
- `max_drawdown_duration`

## Modifikasi Periode EMA
Ganti nilai `span` saat memanggil `calculate_ema` di notebook untuk menggunakan periode EMA yang berbeda.

## Keterbatasan
- Tidak mempertimbangkan slippage realistis atau likuiditas pasar.
- Tidak memperhitungkan leverage atau posisi short.
- Backtest bersifat historis dan tidak menjamin performa masa depan.

### 🔎 EMA Optimization

Notebook `notebooks/ema_optimization.ipynb` melakukan grid search EMA (20–200)
untuk strategi:

- Long ketika close > EMA
- Cash ketika close < EMA

Setiap nilai EMA dievaluasi dengan **QF-Lib TimeseriesAnalysis**, menghasilkan:
Sharpe ratio, CAGR, volatilitas tahunan, max drawdown, dll.

Hasil akhirnya:
- menampilkan EMA span terbaik berdasarkan Sharpe (dengan tie-breaker CAGR),
- grafik Sharpe vs EMA span,
- grafik CAGR vs EMA span,
- serta backtest khusus untuk EMA terbaik.

