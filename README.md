# BTCUSDT Strategy Backtesting (EMA, MACD, ATR & Mean Reversion)

## Ringkasan
Proyek ini sekarang dipusatkan pada pipeline `run_single_asset_pipeline` di
`src/pipelines/single_asset.py`. Pipeline tersebut memuat data OHLCV, menghitung indikator
opsional, menjalankan strategi dari registry TradingView (`src/strategy_backtest/`), dan
mengembalikan metrik performa lengkap. Seluruh notebook telah disederhanakan sehingga cukup
mengubah sel konfigurasi `CONFIG` untuk menjalankan eksperimen baru. Selain itu tersedia CLI
`python -m src.cli.run_single_asset` dan dataset mini agar eksperimen dapat dijalankan tanpa
notebook sekaligus dapat dites otomatis lewat pytest.

## Fitur Utama
- 📦 **Pipeline tunggal** – `SingleAssetPipelineConfig` merangkum seluruh parameter (lokasi data,
  strategi, indikator tambahan, dan horizon). Pipeline mengembalikan objek `BacktestOutputs`
  berisi metrik, trade log, kurva ekuitas, dan ringkasan statistik siap pakai.
- 🗒️ **Notebook parametrik** – setiap notebook di `notebooks/` hanya berisi tiga sel: import,
  konfigurasi `CONFIG`, dan eksekusi pipeline. Mengganti path data atau nama strategi tidak lagi
  membutuhkan modifikasi kode manual.
- 💻 **CLI untuk eksperimen cepat** – `python -m src.cli.run_single_asset CONFIG.json` membaca file
  konfigurasi JSON/YAML (contoh tersedia di `configs/ema50_daily.json`) dan otomatis menyimpan
  metrik, log trade, serta grafik ekuitas ke folder yang ditentukan.
- ✅ **Dataset mini + pytest** – file `tests/fixtures/mini_ohlcv.csv` menyediakan OHLCV sintetis
  sehingga pipeline dapat diuji otomatis. Test `tests/test_pipeline.py` memastikan jumlah trade,
  Sharpe ratio, dan artefak ekspor valid.
- 📊 **Evaluasi kuantitatif** – metrik dihitung via `qflib_metrics_from_returns` (dengan fallback
  pandas/numpy) sehingga Sharpe, CAGR, drawdown, dan volatilitas tersedia di mana pun pipeline
  dipanggil.

## Struktur Folder
```
project-root/
├─ configs/                 # Contoh file konfigurasi untuk CLI
├─ data/                    # Dataset mentah (OKX, sample 1H, dsb.)
├─ notebooks/               # Notebook parametris dengan sel CONFIG
├─ outputs/                 # Folder default penyimpanan artefak
├─ src/
│  ├─ cli/run_single_asset.py
│  ├─ pipelines/single_asset.py
│  ├─ strategy_backtest/...
│  └─ strategi lainnya (ema, macd, atr, oversold)
├─ tests/
│  ├─ fixtures/mini_ohlcv.csv
│  └─ test_pipeline.py
├─ requirements.txt
└─ README.md
```

## Instalasi
1. Pastikan Python 3.10+ tersedia.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Workflow Notebook
1. Jalankan `jupyter lab` dari root proyek.
2. Buka notebook apa pun di `notebooks/`. Sel pertama menyiapkan path & import pipeline.
3. Ubah nilai pada dictionary `CONFIG` di sel kedua (mis. `data_path`, `strategy_name`, parameter
   strategi, indikator tambahan, horizon data, atau lokasi penyimpanan artefak).
4. Jalankan sel eksekusi untuk melihat metrik, ringkasan trade, dan file artefak yang tersimpan.

## Menjalankan Pipeline via CLI
1. Siapkan file konfigurasi JSON/YAML mengikuti struktur `configs/ema50_daily.json`.
2. Jalankan perintah:
   ```bash
   python -m src.cli.run_single_asset configs/ema50_daily.json
   ```
3. Gunakan opsi tambahan bila diperlukan:
   - `--output-dir <path>` untuk mengganti folder tujuan.
   - `--prefix <nama>` untuk mengganti prefix file output.
   - `--skip-save` jika hanya ingin melihat ringkasan tanpa menyimpan artefak.

Output CLI akan mencetak jumlah trade, Sharpe ratio, CAGR, dan lokasi file `*_metrics.json`,
`*_trades.csv`, serta `*_equity.png` yang dihasilkan oleh `save_backtest_outputs`.

## Pengujian
Dataset sintetis pada `tests/fixtures/mini_ohlcv.csv` memudahkan verifikasi pipeline. Jalankan:
```bash
pytest
```
Test `tests/test_pipeline.py` memanggil pipeline dengan strategi EMA50, mengecek jumlah trade,
Sharpe ratio tidak NaN, dan memastikan fungsi `save_backtest_outputs` membuat file metrik, trade
log, serta grafik ekuitas.

## Data
Letakkan file CSV historis pada folder `data/` dengan kolom minimal `time`, `open`, `high`, `low`,
`close`, dan `volume`. File bawaan repo mencakup `OKX_BTCUSDT, 1D.csv`, `OKX_ETHUSDT.P, 1D.csv`,
dan `sample_1h_data.csv`. Untuk unit test, gunakan `tests/fixtures/mini_ohlcv.csv` yang sudah
mengandung kolom indikator dasar (EMA, MACD, ATR) sehingga strategi apa pun dapat dijalankan.

## Cara Kerja Pipeline
1. `load_ohlcv_csv` membersihkan data dan memastikan kolom numerik siap pakai.
2. `_apply_indicators` (opsional) menghitung indikator tambahan berdasarkan daftar
   `IndicatorConfig` pada `SingleAssetPipelineConfig`.
3. `get_strategy` dari `src/strategy_backtest/registry.py` menginisialisasi strategi TradingView.
4. Strategi menghasilkan DataFrame sinyal (`long_entry`, `long_exit`, dll.) yang dieksekusi oleh
   `SignalBacktester`.
5. `qflib_metrics_from_returns` menghitung metrik performa, sedangkan `save_backtest_outputs`
   menyimpan JSON metrik, CSV trade log, dan grafik kurva ekuitas.

## Keterbatasan
- Tidak memperhitungkan slippage, likuiditas, maupun biaya trading riil.
- Mayoritas strategi bersifat long-only.
- Backtest tetap memerlukan validasi out-of-sample sebelum implementasi live.
