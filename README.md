# Crypto Backtesting & Practical ML Pipeline

## Ringkasan
Repositori ini menyediakan **dua jalur eksperimen utama** yang bisa dijalankan secara terpisah
atau berurutan:

1. `run_single_asset_pipeline` (`src/pipelines/single_asset.py`) untuk backtest strategi
   deterministik berbasis indikator (EMA, MACD, ATR, mean reversion, dsb.). Pipeline ini
   memuat data OHLCV, menerapkan indikator opsional, menjalankan strategi TradingView, dan
   mengembalikan metrik performa serta artefak kurva ekuitas.
2. `run_practical_crypto_ml_pipeline` (`src/pipelines/practical_crypto_ml.py`) yang menyiapkan
   pondasi **practical crypto trading ML pipeline**: multi-source data ingestion (funding,
   order book, on-chain, sentiment), feature engineering & normalisasi, label builder untuk
   horizon yang bisa dikonfigurasi, model stack (logistic L1 + gradient boosting) dengan
   walk-forward CV, ensemble signal, hingga konstruksi portofolio beserta guardrails.

Notebook `notebooks/backtest-strategy.ipynb` tetap menjadi entry point visual untuk pipeline
single-asset, sementara pipeline ML dapat dipanggil langsung lewat skrip Python/CLI custom.

## Fitur Utama
### Single-Asset Backtesting
- 📦 **Pipeline terintegrasi** – `SingleAssetPipelineConfig` merangkum seluruh parameter (lokasi
  data, strategi, indikator tambahan, dan horizon) dan mengembalikan `BacktestOutputs` berisi
  metrik, trade log, kurva ekuitas, serta statistik lainnya.
- 🎛️ **Modulasi sinyal eksternal** – pipeline bisa menerima file CSV prediksi/guardrail (mis. skor
  ML) untuk **scale in/out** posisi atau memblokir entry ketika kepercayaan rendah. File cukup
  berisi kolom waktu dan sinyal yang akan di-align ke OHLCV.
- 🗒️ **Notebook parametrik** – setiap notebook di `notebooks/` hanya berisi tiga sel: import,
  konfigurasi `CONFIG`, dan eksekusi pipeline. Mengganti path data atau strategi tidak lagi
  membutuhkan modifikasi kode manual.
- 💻 **CLI untuk eksperimen cepat** – `python -m src.cli.run_single_asset CONFIG.json` membaca file
  konfigurasi JSON/YAML (contoh `configs/ema50_daily.json`, `configs/ema112_hourly.json`, atau
  `configs/vwap_hourly.json`) dan otomatis menyimpan metrik, log
  trade, serta grafik ekuitas ke folder yang ditentukan.
- ✅ **Dataset mini + pytest** – `tests/fixtures/mini_ohlcv.csv` memungkinkan regresi otomatis via
  `pytest` untuk memastikan Sharpe, jumlah trade, dan artefak ekspor konsisten.

### Practical Crypto ML Pipeline
- 🔗 **Multi-source ingestion** – `MultiSourceDataConfig` menggabungkan OHLCV utama dengan data
  funding, order-book depth, on-chain activity, dan sentiment (setiap sumber punya rule resample,
  kolom pilihan, prefix, serta batas forward-fill).
- 🧮 **Feature store & normalisasi** – `_engineer_base_features` membangun blok teknikal
  (return multi-horizon, momentum, volatilitas, volume, jarak ke EMA) dan kini dapat menambahkan
  fitur opsional seperti rolling funding skew, order-book imbalance, serta realized volatility
  intrabar lewat `FeatureEngineeringConfig`. Seluruh sumber kemudian digabung, di-clip dengan
  Z-score, di-imputasi, dan dinormalisasi oleh `FeatureStore` berbasis `StandardScaler`.
- 🎯 **Label builder fleksibel** – `LabelConfig` menentukan horizon (mis. 24 bar), jenis tugas
  (binary/regression), threshold (statis atau berbasis volatilitas rolling), dan kolom harga untuk
  membentuk target ML.
- 🤖 **Model stack + walk-forward CV** – `ModelStackConfig` mengaktifkan Logistic Regression L1 dan
  Gradient Boosting, masing-masing dievaluasi dengan `TimeSeriesSplit` agar tidak terjadi data
  leakage. Out-of-fold prediction digabung menjadi ensembel sinyal [-1, 1].
- 📈 **Portfolio & guardrails** – `PortfolioConfig` mengatur top-K selection, opsi long-short,
  leverage maksimum, target volatilitas, dan batas turnover; fungsi `_build_portfolio` otomatis
  mengecilkan bobot jika guardrails terlanggar dan `_evaluate_realized_performance` menghitung
  directional accuracy maupun information ratio sinyal.

## Struktur Folder
```
project-root/
├─ configs/                 # Contoh file konfigurasi untuk CLI
├─ data/                    # Dataset mentah (OKX, sample 1H, dsb.)
├─ notebooks/               # Notebook parametris dengan sel CONFIG
├─ outputs/                 # Folder default penyimpanan artefak
├─ src/
│  ├─ cli/run_single_asset.py
│  ├─ pipelines/
│  │  ├─ practical_crypto_ml.py   # Multi-source ML pipeline
│  │  └─ single_asset.py          # Backtest strategi deterministik
│  ├─ strategy_backtest/...
│  └─ strategi lainnya (ema, macd, atr, oversold)
├─ tests/
│  ├─ fixtures/mini_ohlcv.csv
│  └─ test_pipeline.py
├─ requirements.txt
└─ README.md
```

## Tahap-Tahap Kerja
Bagian berikut merapikan seluruh proses menjadi tahapan berurutan. Ikuti langkahnya dari atas
ke bawah agar setup, data, backtest, hingga eksperimen ML berjalan mulus.

### Tahap 0 – Siapkan Lingkungan Kerja
1. **Clone repo dan masuk ke folder proyek**
   ```bash
   git clone <repo-url>
   cd backtest
   ```
2. **(Opsional) buat virtual environment** supaya dependensi isolasi.
3. **Instal paket yang dibutuhkan** untuk menjalankan pipeline dan notebook:
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan tes regresi cepat** menggunakan dataset mini untuk memastikan pipeline inti
   (`src/pipelines/single_asset.py`) berfungsi:
   ```bash
   pytest
   ```

### Tahap 1 – Siapkan Data & Notebook
1. Taruh file OHLCV mentah (min. kolom `time`, `open`, `high`, `low`, `close`, `volume`) di
   folder `data/`. File contoh tersedia: `OKX_BTCUSDT, 1D.csv`, `OKX_ETHUSDT.P, 1D.csv`, dan
   `sample_1h_data.csv`.
2. Untuk workflow notebook yang konsisten, jalankan `notebooks/features_target_pipeline.ipynb`
   terlebih dahulu. Notebook ini memanggil utilitas pada `src/features/` untuk:
   - memuat OHLCV via `load_ohlcv_csv`;
   - membangun fitur teknikal (termasuk ADX/±DI untuk mengukur kekuatan tren);
   - menulis dataset hasil (`data/processed/*.parquet` atau `.csv`).
   > Jika `pyarrow` belum terinstal, notebook otomatis jatuh ke CSV. Instal `pyarrow` lewat
   > `pip install -r requirements.txt` agar bisa menyimpan Parquet yang lebih efisien.
3. Saat menjalankan notebook apa pun di `notebooks/`, cukup perbarui dictionary `CONFIG` pada
   sel kedua untuk menunjuk file data, strategi, dan lokasi output.

### Tahap 2 – Backtest Single Asset
1. **Pahami komponen utama**:
   - `SingleAssetPipelineConfig` dan `IndicatorConfig` (di `src/pipelines/single_asset.py`) untuk
     mendeskripsikan data, strategi, indikator, dan horizon yang ingin diuji.
   - `run_single_asset_pipeline` memanggil loader `load_ohlcv_csv`, menghitung indikator,
     menjalankan strategi dari `src/strategy_backtest/registry.py`, lalu mengembalikan
     `BacktestOutputs` yang berisi metrik, ringkasan trade, dan hasil eksekusi.
   - `save_backtest_outputs` menyimpan `*_metrics.json`, `*_trades.csv`, dan `*_equity.png`.
     File JSON kini menyertakan blok `standard_metrics` dengan skema konsisten untuk
     `total_return`, `cagr`, `max_drawdown`, dan `hit_rate` sehingga artefak lintas strategi
     dapat langsung dibandingkan.
2. **Pilih modus eksekusi**:
   - **Notebook**: jalankan `jupyter lab`, buka `notebooks/backtest-strategy.ipynb`, ubah `CONFIG`,
     lalu jalankan semua sel untuk melihat metrik dan grafik.
   - **CLI**: buat file konfigurasi mirip `configs/ema50_daily.json`, kemudian jalankan
     ```bash
     python -m src.cli.run_single_asset configs/ema50_daily.json \
         --output-dir outputs/ema50 --prefix btcusdt_daily
     ```
     CLI (`src/cli/run_single_asset.py`) akan mencetak Sharpe, CAGR, total trade, dan lokasi
     artefak yang disimpan. Opsional, tambahkan filter sinyal eksternal (skala 0–1) lewat config
     atau argumen CLI:
     ```bash
     python -m src.cli.run_single_asset configs/ema112_hourly_external.json \
         --external-signal configs/sample_external_signal.csv \
         --output-dir outputs/ema112_modulated --prefix btcusdt_modulated
     ```
     Nilai sinyal `<= block_threshold` akan menutup akses entry (block trade), sementara nilai lain
     mengalikan bobot posisi berjalan (scale in/out) tanpa mengubah log trade asli.
3. **Tambahkan indikator opsional** dengan menaruh array `IndicatorConfig` di konfigurasi
   (mis. `{"name": "ema", "source_column": "close", "target_column": "ema_fast", "params": {"span": 25}}`).
4. **Eksperimen multi-strategi** dengan membuat beberapa file konfigurasi di `configs/` dan
   menjalankan CLI dalam loop/Makefile untuk mengisi `outputs/` dengan hasil yang siap dibandingkan.
   Contoh preset siap pakai:
  - `configs/ema112_hourly.json` → EMA112 trend-following pada `data/BINANCE_ETHUSDT.P, 60.csv`.
  - `configs/vwap_hourly.json` → Strategi VWAP mean-reversion pada file yang sama.
  - Notebook perbandingan lintas timeframe: `notebooks/strategy_comparison_tf1h.ipynb`
    (data Binance ETHUSDT perpetual 1H) dan `notebooks/strategy_comparison_tf1d.ipynb`
    (data Binance ETHUSD 1D) sudah menyiapkan `OUTPUT_DIR` serta file ekspor Excel terpisah
    sehingga artefak tidak bercampur. Ringkasan data/output tiap timeframe beserta insight
    tren (bullish vs sideways/bearish) yang sudah diuji tersedia di
    `notebooks/strategy_timeframe_matrix.ipynb`. Artefak sinyal ML hourly yang berasal dari
    `outputs/result-test/ml_baseline.xlsx` kini diturunkan ke
    `outputs/strategy_comparison/ml_hourly_metrics.json` agar bisa dibaca langsung oleh
    matriks timeframe.

### Tahap 3 – Practical Crypto ML Pipeline
1. **Konfigurasikan sumber data** menggunakan `MultiSourceDataConfig` pada
   `src/pipelines/practical_crypto_ml.py`.
   - `ohlcv_path` wajib; sumber tambahan (`funding_rates`, `order_book_depth`, `on_chain_activity`,
     `sentiment_scores`) bersifat opsional tetapi dapat diberi prefix, daftar kolom, dan aturan
     resampling.
   - Fungsi `_load_auxiliary_source` memastikan setiap CSV memiliki kolom `time` dan di-align ke
     index OHLCV.
2. **Bangun fitur** dengan `_engineer_base_features` (return multi-horizon, momentum, volatilitas,
   sinyal volume, jarak ke EMA). Gunakan `FeatureEngineeringConfig` untuk mengaktifkan fitur
   tambahan seperti rolling funding skew (memanfaatkan sumber `funding_rates`), order-book
   imbalance (menggunakan kolom bid/ask volume), atau realized volatility intrabar dari rentang
   high-low. Seluruh sumber kemudian digabung lewat `_merge_sources`, di-clip berdasarkan Z-score,
   dan diimputasi/`dropna` sesuai konfigurasi.
   ```python
   feature_cfg = FeatureEngineeringConfig(
       realized_vol_window=24,
       funding_skew_window=72,
       funding_skew_column="funding_rate",
       order_book_imbalance_window=12,
       order_book_bid_column="bid_volume",
       order_book_ask_column="ask_volume",
   )
   data_cfg = MultiSourceDataConfig(
       ohlcv_path="data/sample_1h_data.csv",
       funding_rates=AuxiliarySourceConfig(path="data/funding.csv", prefix="fund_"),
       order_book_depth=AuxiliarySourceConfig(path="data/order_book.csv", prefix="ob_"),
       feature_engineering=feature_cfg,
   )
   ```
3. **Buat label** memakai `LabelConfig` (mendukung threshold statis atau dinamis berbasis
   volatilitas rolling):
   ```python
   labels = build_labels(
       ohlcv,
       LabelConfig(
           horizon_bars=24,
           task="binary",
           threshold=0.0,
           rolling_vol_window=48,
           rolling_vol_multiplier=0.5,
       ),
   )
   labels_reg = build_labels(ohlcv, LabelConfig(horizon_bars=24, task="regression"))
   ```
   Label akan sejajar dengan index fitur setelah `dropna()`. Gunakan `task="regression"` bila ingin
   memprediksi forward returns kontinu alih-alih klasifikasi 0/1.
4. **Normalisasi & simpan metadata fitur** dengan `FeatureStore.fit`, yang memanfaatkan
   `StandardScaler` agar pipeline inference bisa menggunakan scaler yang sama.
5. **Latih model stack** lewat `train_model_stack`. Secara default pipeline menjalankan Logistic
   Regression L1 dan Gradient Boosting (`sklearn`) dengan walk-forward CV (`TimeSeriesSplit`). Kamu
   bisa menyalakan varian linear lain seperti ElasticNet Logistic (grid `C` + `l1_ratio`), Probit
   (butuh `statsmodels`), atau `SGDClassifier` (loss `log_loss` atau `hinge` dengan grid `alpha`).
   Aktifkan baseline deep learning (MLPClassifier) dengan mengatur
   `ModelStackConfig(train_deep_learning=True, mlp_hidden_layers=(128, 64))` sehingga tiga blok
   model (linear, tree, dan neural network) tersedia sekaligus.
   - **Praktik terbaik tuning**:
     - Pertahankan `TimeSeriesSplit` (walk-forward) agar evaluasi tidak bocor ke masa depan.
     - Lakukan grid search sederhana untuk regularisasi Logistic Regression (mis. `logistic_l1_cs`
       atau `logistic_elasticnet_cs` + `logistic_elasticnet_l1_ratios`) dan atur `logistic_tol`
       serta `logistic_max_iter` agar konvergen.
     - Aktifkan early stopping pada SGD (`sgd_early_stopping=True`, `sgd_n_iter_no_change=5`) atau
       batasi `sgd_max_iter` supaya training tidak berlarut.
     - Untuk MLP, gunakan regularisasi `alpha` dan parameter optimizer `beta_1`/`beta_2` atau
       `mlp_early_stopping=True` + `mlp_validation_fraction` sebagai alternatif "dropout-like"
       (validasi hold-out otomatis memotong training bila tidak ada perbaikan).
   - Untuk `task="binary"`, `_train_and_score_model` mencatat rata-rata `accuracy` dan `roc_auc`.
   - Untuk `task="regression"`, pipeline otomatis memakai LinearRegression/Ridge/Lasso (serta
     opsional GradientBoostingRegressor jika `train_tree_based=True`) dan mencatat `mae` + `r2`.
   - Prediksi out-of-fold setiap model disimpan di DataFrame `predictions`.
6. **Buat sinyal & portofolio**:
   - `_combine_predictions` mengubah rata-rata probabilitas ke rentang [-1, 1] (tugas klasifikasi)
     atau mengambil sign prediksi dan mengalikannya dengan quantile absolut (tugas regresi), sehingga
     sinyal terjaga di kisaran [-1, 1] dan membawa informasi kekuatan.
   - `_build_portfolio` menerjemahkan sinyal menjadi bobot sesuai `PortfolioConfig` (top-K,
     long-short optional, batas leverage & turnover). Guardrail yang terpicu ditandai di
     `guardrail_flags`.
7. **Evaluasi performa realisasi** dengan `_evaluate_realized_performance`, yang menghitung
   directional accuracy atau korelasi sinyal-label serta information ratio tahunan.
8. **Jalankan semuanya** melalui `run_practical_crypto_ml_pipeline` dan simpan objek
   `MLPipelineResult` untuk dianalisis lebih lanjut (mis. `result.cv_metrics`,
   `result.portfolio_weights.tail()`). Pipeline sekarang otomatis menulis
   tabel per-fold dan grafik stabilitas PSI ke `outputs/cv_report.json` serta
   `outputs/cv_stability.png` (path bisa dioverride). File JSON memuat distribusi
   label train/test, statistik mean/std fitur, serta skor stability (Population
   Stability Index) per fold sehingga drift dapat dilacak tanpa membuka notebook.
9. **Simpan artefak backtest sinyal ML** dengan `save_ml_backtest_outputs` (dari
   `src/pipelines/ml_signals.py`) yang menulis `*_metrics.json` berisi `standard_metrics`
   (`total_return`, `cagr`, `max_drawdown`, `hit_rate`) serta ringkasan trade log ke CSV.

### Langkah 7 – Baseline Linear / Tree / Deep Learning

Blok ini berada pada `src/pipelines/practical_crypto_ml.py` fungsi `train_model_stack`.
Konfigurasinya dikendalikan oleh `ModelStackConfig` sehingga kamu bisa menentukan kombinasi model
apa saja yang ingin dilatih tanpa menyentuh kode:

```python
from src.pipelines.practical_crypto_ml import ModelStackConfig

model_cfg = ModelStackConfig(
    train_linear=True,             # Logistic Regression L1 (baseline reguler)
    train_logistic_elasticnet=True,# Grid ElasticNet Logistic (variasi C & l1_ratio)
    train_probit=False,            # Probit (aktif jika statsmodels terpasang)
    train_sgd=True,                # SGDClassifier (log_loss & hinge) dengan grid alpha
    train_tree_based=True,         # Gradient Boosting / tree-based workhorse
    train_deep_learning=True,      # Aktifkan MLPClassifier (feedforward NN)
    mlp_hidden_layers=(128, 64),   # Atur arsitektur hidden layer
    mlp_alpha=1e-3,                # Regularisasi L2 untuk menstabilkan MLP
    mlp_beta1=0.9,                 # Hyperparameter Adam (momentum)
    mlp_beta2=0.999,               # Hyperparameter Adam (RMS)
    mlp_validation_fraction=0.1,   # Split hold-out untuk early stopping
    mlp_early_stopping=True,
    mlp_max_iter=400,              # Iterasi training agar konvergen untuk data 1H
    logistic_l1_cs=(0.5, 1.0, 2.0),
    logistic_max_iter=1500,
    logistic_elasticnet_cs=(0.2, 1.0, 2.0),
    logistic_elasticnet_l1_ratios=(0.2, 0.5, 0.8),
    logistic_elasticnet_max_iter=2000,
    sgd_alphas=(0.0001, 0.001, 0.01),
    sgd_max_iter=800,
    sgd_early_stopping=True,
    sgd_n_iter_no_change=5,
)
```

Pipeline akan menjalankan walk-forward CV (`TimeSeriesSplit`) untuk ketiga model sekaligus, dan
mencatat metrik `accuracy` + `roc_auc` (klasifikasi) atau `mae` + `r2` (regresi) di `result.cv_metrics`
dengan kunci sesuai nama varian (mis. `logistic_en_C1.0_l1r0.5`, `sgd_hinge_alpha0.001`). Untuk setiap
model, `result.cv_report` berisi log per-fold (label balance, rangkuman mean/std fitur, serta PSI train
vs test) dan disalin ke `outputs/cv_report.json`. Grafik `outputs/cv_stability.png` menampilkan rata-rata
PSI per fold sehingga pergeseran distribusi fitur mudah dipantau tanpa memuat DataFrame besar. Semua
path artefak dapat diubah lewat argumen `cv_report_path` / `cv_plot_path` ketika memanggil
`run_practical_crypto_ml_pipeline`. Prediksi out-of-fold setiap model disimpan di DataFrame `predictions`.
menggabungkan prediksi menjadi sinyal ensemble. Untuk regresi, gunakan `LabelConfig(task="regression")`
agar stack berisi LinearRegression/Ridge/Lasso (+GradientBoostingRegressor opsional) dan sinyal diubah
menjadi arah * sign(prediksi) * kekuatan berdasarkan quantile absolut. Kamu bisa mematikan salah satu
blok (misal hanya linear + tree) dengan menyetel flag ke `False`.

### Tahap 4 – Analisis & Dokumentasi Hasil
1. Gunakan notebook `notebooks/strategy_comparison.ipynb` untuk menggabungkan output di
   `outputs/` dan membuat ranking Sharpe/CAGR.
2. Arsipkan artefak penting yang dibuat `save_backtest_outputs` ke penyimpanan eksternal.
3. Catat konfigurasi yang menghasilkan performa baik (mis. commit file JSON/YAML di `configs/`).
4. Jalankan `pytest` setiap kali menambahkan strategi baru untuk memastikan tidak ada regresi.

> **Tips cepat**: Bila ingin loncat langsung ke pipeline tertentu, gunakan daftar isi berikut
> untuk menemukan instruksi spesifiknya:
> - [Workflow Notebook (Single Asset)](#workflow-notebook-single-asset)
> - [Menjalankan Pipeline via CLI](#menjalankan-pipeline-via-cli)
> - [Practical Crypto ML Pipeline](#practical-crypto-ml-pipeline)
> - [Pengujian](#pengujian)

## Workflow Notebook (Single Asset)
1. Jalankan `jupyter lab` dari root proyek.
2. Buka notebook apa pun di `notebooks/`. Sel pertama menyiapkan path & import pipeline.
3. Ubah nilai pada dictionary `CONFIG` di sel kedua (mis. `data_path`, `strategy_name`, parameter
   strategi, indikator tambahan, horizon data, atau lokasi penyimpanan artefak).
4. Jalankan sel eksekusi untuk melihat metrik, ringkasan trade, dan file artefak yang tersimpan.

## Menjalankan Pipeline via CLI
1. Siapkan file konfigurasi JSON/YAML mengikuti struktur `configs/ema50_daily.json`.
   Path apa pun di dalam konfigurasi akan dianggap relatif terhadap lokasi file konfigurasi.
   Karena folder preset berada di `configs/`, gunakan prefix `../data/` jika file data Anda
   berada di direktori `data/` pada root repo.
2. Jalankan perintah:
   ```bash
   python -m src.cli.run_single_asset configs/ema50_daily.json
   ```
   Untuk file hourly terbaru, gunakan preset:
   ```bash
   python -m src.cli.run_single_asset configs/ema112_hourly.json
   python -m src.cli.run_single_asset configs/vwap_hourly.json
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
### Single-Asset Backtest
1. `load_ohlcv_csv` membersihkan data dan memastikan kolom numerik siap pakai.
2. `_apply_indicators` (opsional) menghitung indikator tambahan berdasarkan daftar
   `IndicatorConfig` pada `SingleAssetPipelineConfig`.
3. `get_strategy` dari `src/strategy_backtest/registry.py` menginisialisasi strategi TradingView.
4. Strategi menghasilkan DataFrame sinyal (`long_entry`, `long_exit`, dll.) yang dieksekusi oleh
   `SignalBacktester`.
5. `qflib_metrics_from_returns` menghitung metrik performa, sedangkan `save_backtest_outputs`
   menyimpan JSON metrik, CSV trade log, dan grafik kurva ekuitas.

### Practical Crypto ML Pipeline
1. `MultiSourceDataConfig` menentukan lokasi OHLCV utama serta sumber tambahan (funding,
   order book, on-chain, sentiment) lengkap dengan aturan resampling dan prefix kolom.
2. `_merge_sources` memanggil `_engineer_base_features`, melakukan join terhadap seluruh sumber,
   clipping outlier berbasis Z-score, kemudian forward-fill & drop missing sesuai konfigurasi.
3. `build_labels` membentuk target (binary/regression) sesuai horizon `LabelConfig` dan price
   column yang dipilih.
4. `FeatureStore.fit` menormalkan fitur dan menyimpan `StandardScaler` agar dapat dipakai ulang.
5. `train_model_stack` melatih Logistic Regression L1 dan/atau Gradient Boosting dengan
   `TimeSeriesSplit`, sekaligus menghasilkan prediksi out-of-fold untuk ensemble.
6. `_combine_predictions` mengubah probabilitas menjadi sinyal [-1, 1] lalu `_build_portfolio`
   menerjemahkannya menjadi bobot dengan guardrails leverage/turnover.
7. `_evaluate_realized_performance` menghitung directional accuracy atau korelasi sinyal-label dan
   information ratio tahunan sebagai sanity check.

Contoh penggunaan minimal:
```python
from src.pipelines.practical_crypto_ml import (
    AuxiliarySourceConfig,
    LabelConfig,
    ModelStackConfig,
    MultiSourceDataConfig,
    PortfolioConfig,
    run_practical_crypto_ml_pipeline,
)

data_cfg = MultiSourceDataConfig(
    ohlcv_path="data/BINANCE_ETHUSDT.P, 60.csv",
    funding_rates=AuxiliarySourceConfig(path="data/funding_rates.csv", prefix="fund_"),
)
result = run_practical_crypto_ml_pipeline(
    data_cfg=data_cfg,
    label_cfg=LabelConfig(horizon_bars=24, task="binary"),
    model_cfg=ModelStackConfig(cv_splits=4),
    portfolio_cfg=PortfolioConfig(top_k=1, long_short=False),
)
print(result.cv_metrics)
print(result.portfolio_weights.tail())
```

## Strategi & Sinyal yang Tersedia

### VWAP Mean Reversion (`vwap`)
- **Entry** – Mengikuti `src/strategy_backtest/strategies/vwap.py`: harga penutupan harus berada di bawah VWAP sesi berjalan, RSI > level oversold (default 30) dan *cross up* di atas 50 untuk membuka **long**. Sebaliknya, posisi **short** hanya dibuka ketika harga di atas VWAP, RSI < level overbought (default 70) dan *cross down* di bawah 50.
- **Exit & manajemen risiko** – Stop loss berbasis ATR (`atr_length=14`) dengan multiplier default 1.5. Posisi long keluar jika harga menyentuh `entry_price - ATR*multiplier`, posisi short keluar di `entry_price + ATR*multiplier`. Bila sinyal berlawanan muncul saat posisi aktif, strategi akan melakukan flip langsung sehingga konteks `position`/`stop_level` ikut diperbarui pada frame sinyal.
- **Kapan dipakai** – Cocok untuk pasar sideways atau mean-reverting intraday karena syarat entry menunggu konfirmasi momentum dari RSI namun tetap membeli diskon terhadap VWAP. Dalam data ETHUSDT 1 jam, strategi ini menjalankan 41 trade (26 long, 15 short) dengan Sharpe 1.50 dan drawdown maksimum -14.8% sekaligus profit factor >2.3.

### EMA112 + ATR Auto-Flip (`ema112_atr`)
- **Entry** – Harga harus menembus EMA112 dari bawah ke atas agar membuka long; penembusan dari atas ke bawah memicu short (`src/strategy_backtest/strategies/ema112_atr.py`). Strategi memeriksa `close` vs `ema` bar sebelumnya agar hanya *true breakout* yang dieksekusi.
- **Exit & auto re-entry** – Stop loss dinamis memakai ATR14 * 1.2. Jika stop kena dan `reentry_enabled=True`, posisi otomatis dibalik sehingga tetap mengikuti momentum dominan tanpa menunggu sinyal EMA berikutnya. Trade log menyimpan `entry_price`, `long_stop`, `short_stop`, serta status posisi pada setiap bar.
- **Kapan dipakai** – Dirancang untuk tren kuat karena selalu *in the market* (time-in-market ~52% long-only) dan membiarkan profit besar mengalir (avg win +6.6% vs avg loss -0.8%). Pada dataset ETHUSDT 1 jam strategi menghasilkan CAGR 0.61, Sharpe 1.34 namun menanggung drawdown -24.9% karena whipsaw saat sideways.

### ML LightGBM (`ml_lightgbm`)
- **Dataset & fitur** – Notebook `notebooks/ml_baseline.ipynb` memuat dataset hasil `notebooks/features_target_pipeline.ipynb` (ret 1h/4h/24h, momentum 7d, volatilitas 24h/7d, volume z-score, jarak ke EMA 24/96). Target-nya adalah tanda (sign) return 5 jam ke depan sehingga cocok untuk strategi multi-asset intraday.
- **Model stack** – Dua baseline linear (Lasso & ElasticNet Logistic Regression) dibandingkan dengan `lightgbm.LGBMClassifier` (800 tree, learning rate 0.03, leaves 31). Seluruh model dievaluasi memakai walk-forward `TimeSeriesSplit` sehingga CV dan window out-of-sample (2023) konsisten.
- **Entry/exit sinyal ML** – Fungsi `fit_and_evaluate` mengubah probabilitas menjadi sinyal kontinu `signal = 2 * prob - 1`. Pada tahap deployment, posisi = `sign(signal)`: nilai >0 membuka long, <0 membuka short, dan 0 berarti flat hingga update jam berikutnya.Profit realisasi dihitung sebagai `position * future_return`, setara memegang posisi selama horizon 5 jam yang sama dengan target label.
- **Kinerja ringkas** – LightGBM mencatat CV accuracy 0.510/AUC 0.524 dan out-of-sample accuracy 0.542 dengan ROC-AUC 0.548 serta Sharpe sinyal 5.47.Set prediksi yang diekspor (`outputs/predictions/lightgbm_ml_baseline_predictions.csv`) menunjukkan total return +119.7%, Sharpe 6.16, CAGR terannualisasi tinggi (8,173%) karena penggabungan 759 bar 1 jam menjadi satu tahun ekuivalen, serta hit rate 54.1% dengan drawdown terburuk -48.6%. Notebook yang sama kini juga menulis baseline baru untuk `ml_logreg` dan `ml_linreg` ke `outputs/predictions/ml_logreg_baseline_predictions.csv` dan `outputs/predictions/ml_linreg_baseline_predictions.csv` agar bisa dipakai ulang oleh notebook perbandingan.

### ML Logistic Regression (`ml_logreg`)
- **Pipeline** – Memanggil `run_practical_crypto_ml_pipeline` (tugas klasifikasi) dengan konfigurasi linear di `train_model_stack` (Logistic Regression L1/elasticnet). Hasil probabilitas diubah menjadi sinyal [-1, 1] lalu diterjemahkan ke entri/exit `SignalBacktester` dengan guardrail leverage (`max_leverage`) dan turnover (`turnover_limit`).
- **Konfigurasi penting** – Selaraskan `strategy_kwargs.data_path` dengan `data_path` pipeline agar loader ML dan backtester membaca berkas yang sama. Opsi lain mengikuti `LabelConfig`/`PortfolioConfig`: `label_horizon` (default 24 bar), `top_k`, `long_short=True`, `max_leverage=1.0`, `turnover_limit=1.5`, serta `cv_splits` untuk `TimeSeriesSplit`.
- **Contoh CLI** – Gunakan preset `configs/ml_logreg_hourly.json` lalu jalankan `python -m src.cli.run_single_asset configs/ml_logreg_hourly.json`. Artefak akan ditulis ke `outputs/ml_logreg/ethusdt_ml_logreg_*` sesuai `output_prefix`.
- **Notebook** – Di `notebooks/backtest-strategy.ipynb`, set `CONFIG["strategy_name"] = "ml_logreg"` dan isi `CONFIG["strategy_kwargs"]["data_path"]` agar sama dengan path OHLCV. Pipeline notebook akan menyimpan metrik dan trade log sembari memuat konteks `ml_signal`, `ml_probability`, serta flag guardrail pada frame sinyal.

### ML Linear Regression (`ml_linreg`)
- **Pipeline** – Menjalankan `run_practical_crypto_ml_pipeline` dengan `LabelConfig(task="regression")` sehingga target adalah forward returns kontinu. Stack default mencakup LinearRegression, Ridge, dan Lasso (opsional GradientBoostingRegressor), masing-masing dievaluasi via walk-forward CV dengan metrik `mae`/`r2`.
- **Interpretasi sinyal** – Ensemble prediksi dikonversi ke [-1, 1] dengan `sign(prediksi) * quantile(|prediksi|)`: sign menentukan arah (long/short), sedangkan quantile absolut menjadi skala kekuatan sehingga sinyal 0.8 berarti konfidensi tinggi untuk long.
- **Contoh CLI** – Preset regresi single-asset dapat langsung dieksekusi lewat `python -m src.cli.run_single_asset configs/ml_linreg_hourly.json`; artefak disimpan di `outputs/ml_linreg/ethusdt_ml_linreg_*`.
- **Notebook** – Atur `CONFIG["strategy_name"] = "ml_linreg"` beserta `CONFIG["strategy_kwargs"]["data_path"]` agar notebook memakai berkas OHLCV yang sama. Notebook akan menampilkan `ml_signal` (arah & kekuatan), `predicted_return`, serta flag guardrail.

#### Ringkasan cepat strategi ML & opsi Logistic
- `ml_logreg` menjalankan Logistic Regression L1 (bisa ditambah ElasticNet/SGD/Probit via `ModelStackConfig`) untuk tugas klasifikasi, mengonversi probabilitas menjadi sinyal [-1, 1].
- `ml_linreg` menjalankan LinearRegression/Ridge/Lasso (opsional GradientBoostingRegressor) untuk tugas regresi, lalu memetakan prediksi ke arah & kekuatan sinyal.
- Varian Logistic ElasticNet bisa diaktifkan dengan mengubah `ModelStackConfig(train_logistic_elasticnet=True, logistic_elasticnet_cs=..., logistic_elasticnet_l1_ratios=...)` pada `src/pipelines/practical_crypto_ml.py` atau menambahkan argumen serupa jika memanggil pipeline secara langsung.

| Strategi | Horizon label default | Penalti reguler / model utama | Metode konversi sinyal | File konfigurasi yang diedit |
| --- | --- | --- | --- | --- |
| `ml_logreg` | 24 bar (LabelConfig) | Logistic Regression L1; opsi ElasticNet/SGD/Probit via `ModelStackConfig` | Probabilitas → `2 * p - 1`, lalu `sign` jadi posisi | `configs/ml_logreg_hourly.json` (CLI) atau `notebooks/backtest-strategy.ipynb` (`CONFIG` blok ML) |
| `ml_linreg` | 24 bar (LabelConfig, tugas regresi) | LinearRegression/Ridge/Lasso (opsional GradientBoostingRegressor) | `sign(prediksi) * quantile(|prediksi|)` untuk arah + kekuatan | `configs/ml_linreg_hourly.json` (CLI) atau `notebooks/backtest-strategy.ipynb` (`CONFIG` blok ML) |
| Logistic ElasticNet varian | 24 bar (tugas klasifikasi) | Logistic Regression ElasticNet (`saga`) dengan grid `logistic_elasticnet_cs` & `logistic_elasticnet_l1_ratios` | Probabilitas → sinyal [-1, 1] sama seperti `ml_logreg` | `src/pipelines/practical_crypto_ml.py` → `ModelStackConfig(train_logistic_elasticnet=True, ...)` atau override saat memanggil `run_practical_crypto_ml_pipeline` |

## Hasil Evaluasi & Pemilihan Strategi Sesuai Kondisi Pasar

| Strategi | Total Return | CAGR | Annualised Vol | Sharpe | Max DD | Catatan Regime |
| --- | --- | --- | --- | --- | --- | --- |
| EMA112 | 24.4% | 0.61 | 0.42 | 1.34 | -24.9% | Long-only trend following; unggul saat bullish berkepanjangan namun rawan whipsaw ketika volatilitas mendatar. |
| VWAP | 30.6% | 0.79 | 0.46 | 1.50 | -14.8% | Mean-reversion yang memadukan long & short sehingga lebih stabil di range-bound market dengan profit factor >2. |
| ml_lightgbm | 75.1% | 1.89 | 0.50 | 2.38 | -19.7% | Sinyal probabilistik long/short per jam; performa tinggi berasal dari diferensiasi posisi cepat namun perlu guardrail risiko untuk menahan drawdown. |

- **Tren bullish kuat** – EMA112 unggul karena selalu mengikuti breakout EMA dan membiarkan posisi long berjalan ratusan bar (avg win 98 bar). Gunakan ATR stop ketat serta re-entry otomatis supaya tidak tertinggal saat momentum lanjut.
- **Tren bearish atau range volatil** – VWAP menjadi pilihan karena mampu membalik posisi (26 long vs 15 short) dan disiplin menunggu konfirmasi RSI, sehingga drawdown hanya -14.8% meski volatilitas tahunan 0.46.
- **Pasar choppy dengan banyak katalis makro** – Gunakan `ml_lightgbm` karena probabilitasnya dapat diarahkan menjadi portofolio market-neutral atau long-short cepat (hit rate 54%). Kombinasikan dengan guardrail `PortfolioConfig` (max leverage, turnover limit) supaya sinyal yang sering flip tidak menambah biaya transaksi berlebihan.
- **Sideways panjang** – VWAP + filter RSI paling stabil. EMA112 bisa tetap dipakai bila menambah filter ATR atau menurunkan leverage. ML stack dapat dijadikan overlay: hanya jalankan trade rule-based ketika probabilitas ML searah untuk mengurangi whipsaw.

### Menggabungkan sinyal ML dengan strategi indikator
- **Overlay** – Memakai probabilitas ML sebagai penguat posisi indikator. Contoh: ketika EMA112 memberi sinyal long, ukuran posisi bisa ditimbang oleh `p_long = prob_ml - 0.5` sehingga posisi penuh hanya terjadi jika konfidensi ML tinggi (misal `prob_ml > 0.65`).
- **Filter** – Menahan eksekusi indikator sampai ML setuju arahnya. Praktis untuk mengurangi whipsaw EMA112 saat pasar choppy: jalankan long/short hanya bila `prob_ml` melewati ambang yang sama arah dengan sinyal indikator; jika di bawah threshold, bar dilewatkan.
- **Referensi threshold** – Notebook perbandingan 1 jam (`notebooks/strategy_comparison_tf1h.ipynb`) menyimpan metrik ML ke `outputs/strategy_comparison/ml_hourly_metrics.json`. Nilai seperti `best_threshold` atau `roc_auc` bisa dipakai sebagai acuan awal sebelum grid-search ulang di dataset terbaru.

Contoh pseudo-code sederhana (adaptasi cell pemuatan metrik di notebook perbandingan):

```python
import json

# Muat ambang ML dari artefak agar konsisten dengan evaluasi sebelumnya
with open("outputs/strategy_comparison/ml_hourly_metrics.json") as f:
    ml_metrics = json.load(f)

prob_th = ml_metrics["ml_lightgbm"]["best_threshold"]  # misal 0.55
ml_prob = signals["ml_probability"].shift(1)  # hindari look-ahead

# Filter: EMA112 hanya dieksekusi jika ML searah & yakin
signals["long_entry"] = signals["long_entry"] & (ml_prob > prob_th)
signals["short_entry"] = signals["short_entry"] & (ml_prob < 1 - prob_th)

# Overlay: skala posisi long/short menurut kekuatan ML
signals["position"] = signals["position"] * (ml_prob - 0.5) * 2
```

## Notebook Referensi

- **`notebooks/features_target_pipeline.ipynb`** – Loader OHLCV → rekayasa fitur teknikal → label builder. Notebook ini memanggil utilitas `src/features` untuk menghitung return multi-horizon, momentum, volatilitas, volume anomaly dan menyimpan dataset ke `data/processed/*.csv` untuk dipakai ulang oleh pipeline ML.
- **`notebooks/ml_baseline.ipynb`** – Menyusun baseline ML lengkap: pembagian walk-forward, normalisasi (`StandardScaler`), regresi logistik L1 (Lasso) & ElasticNet, plus LightGBM. Notebook ini juga mengekspor model `.pkl`, prediksi (LightGBM + baseline `ml_logreg` & `ml_linreg`), serta laporan Excel sehingga eksperimen bisa dilacak lintas commit sekaligus siap dibaca oleh dashboard perbandingan.
- **`notebooks/backtest-strategy.ipynb`** – Front-end minimalis untuk memanggil `run_single_asset_pipeline`. Setelah konfigurasi `CONFIG`, notebook ini otomatis mencetak Sharpe, CAGR, serta trade log dan menyediakan sel tambahan untuk analisis MAE/MFE guna memahami distribusi PnL strategi yang sedang diuji.
- **`notebooks/strategy_comparison.ipynb`** – Membaca output CLI/notebook (EMA112, VWAP, ML) lalu menyusun ranking metrik, cuplikan trade, distribusi PnL, hingga ekspor Excel multi-sheet (`comparison_metrics`, `trade_summary`, dan prediksi ML dari `ml_lightgbm`/`ml_logreg`/`ml_linreg`). Notebook ini menjadi dashboard final untuk memutuskan strategi apa yang dipakai live.

## Glosarium Teknis

- **OHLCV** – Singkatan dari *Open, High, Low, Close, Volume*; kolom wajib pada loader `load_ohlcv_csv` dan menjadi basis perhitungan indikator seperti EMA, ATR, maupun VWAP.
- **VWAP (Volume-Weighted Average Price)** – Harga rata-rata tertimbang volume per sesi; digunakan sebagai patokan fair value agar entry dilakukan di diskon/premium tertentu. Implementasi `_session_vwap` mengakumulasi typical price × volume lalu membagi dengan cumulative volume pada sesi aktif.
- **EMA & ATR** – *Exponential Moving Average* memberikan bobot lebih besar pada data terbaru (contoh EMA112), sedangkan *Average True Range* mengukur volatilitas absolut untuk memasang stop dinamis.
- **RSI** – *Relative Strength Index* 14 bar digunakan sebagai filter momentum pada strategi VWAP agar entry hanya terjadi saat terjadi *momentum shift* nyata.
- **Lasso Logistic Regression** – Regresi logistik dengan penalti L1 (`penalty="l1"`) sehingga koefisien sparing (banyak nol) dan cocok untuk feature selection cepat. Pada dataset ini memberikan Sharpe negatif (-4.53) sehingga ditandai sebagai `reject_negative_sharpe`.
- **ElasticNet Logistic Regression** – Kombinasi penalti L1/L2 (`l1_ratio=0.5`) yang menyeimbangkan sparsity dan stabilitas koefisien. Hasilnya juga negatif (-4.54 Sharpe) sehingga lebih cocok sebagai baseline sanity check ketimbang strategi live.
- **LightGBM** – Gradient boosting berbasis histogram yang efisien untuk dataset besar. Parameter default kita (800 trees, leaves 31) memberi sinyal probabilistik yang kemudian diubah menjadi posisi long/short setiap jam.
- **CAGR (Compound Annual Growth Rate)** – Tingkat pertumbuhan tahunan rata-rata yang mempertimbangkan efek compounding; diambil dari rasio `last_equity / first_equity` yang kemudian dipangkatkan `365 / days_between` agar mengikuti selisih hari riil antar bar sehingga cocok untuk data 1 jam maupun time frame lain.
- **Sharpe Ratio** – Rasio excess return terhadap volatilitas. Kita mengukur return rata-rata per bar, annualised return = `mean_r * bars_per_year`, annualised vol = `std_r * sqrt(bars_per_year)`, lalu Sharpe = `annualised_ret / annualised_vol` sehingga ML, EMA112, dan VWAP dibandingkan dengan rumus identik.
- **Annualised Volatility** – Standar deviasi return yang telah diskalakan tahunan (`std * sqrt(bars_per_year)`); untuk data 1 jam konstanta `bars_per_year = 24 * 365` sehingga output Sharpe/vol tidak lagi under-scale ketika masuk ke `backtest-strategy.xlsx` maupun `strategy_comparison.ipynb`.
- **MAE/MFE (Maximum Adverse/Favourable Excursion)** – Statistik trade-level pada `notebooks/strategy_comparison.ipynb` yang memplot seberapa jauh posisi sempat rugi/profit sebelum exit. Panel kiri histogram PnL% menyoroti distribusi outcome per trade, sedangkan panel kanan scatter MAE vs MFE membantu melihat trade mana yang segera membalik arah dan mana yang sempat rally.
  - Lonjakan frekuensi di bin `pnl_pct = 0` (sekitar ~70 trade) datang dari posisi yang ditutup sangat cepat setelah sinyal berbalik sehingga perubahan harga di bar entry sama dengan exit (praktis scratch trade). Ini bukan error: pola tersebut menunjukkan banyak entry yang dibatalkan segera ketika konfirmasi hilang, sehingga risiko terjaga meski tidak menghasilkan profit.
- **Max Drawdown** – Penurunan relatif dari puncak equity; LightGBM menunjukkan -48.6% sehingga wajib dipasangkan dengan manajemen risiko tambahan sebelum produksi.
- **Portfolio Guardrails** – Parameter `PortfolioConfig` (`top_k`, `max_leverage`, `turnover_limit`) yang memastikan sinyal ML tidak menghasilkan leverage berlebih serta tetap sesuai batas risiko saat diterjemahkan menjadi bobot portofolio.

## Keterbatasan
- Tidak memperhitungkan slippage, likuiditas, maupun biaya trading riil.
- Mayoritas strategi bersifat long-only.
- Backtest tetap memerlukan validasi out-of-sample sebelum implementasi live.
