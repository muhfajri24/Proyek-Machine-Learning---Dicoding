# Submission Machine Learning: Clustering dan Klasifikasi Transaksi

Project ini dibuat untuk memenuhi submission machine learning dengan alur lengkap mulai dari EDA, preprocessing, clustering, interpretasi cluster, sampai klasifikasi label cluster menggunakan dataset Kaggle bertema fraud detection.

## Hasil Utama

- Dataset diambil dari Kaggle `aryan208/financial-transactions-dataset-for-fraud-detection`
- Project men-stage sample kerja sebanyak `30.000` baris agar runtime lokal tetap stabil
- Elbow Method dengan `KElbowVisualizer` digunakan untuk evaluasi jumlah cluster, dengan hasil cluster terbaik pipeline saat ini `2`
- Model clustering disimpan sebagai `models/model_clustering.joblib` dan `models/model_clustering.h5`
- Data training hasil clustering diekspor dengan kolom target bernama `Target`
- Model klasifikasi `Decision Tree` disimpan sebagai `models/decision_tree_model.h5`
- Akurasi klasifikasi pada data uji: `0.9993`

## Checklist Submission yang Sudah Dicakup

1. EDA
   - `head()`
   - `info()`
   - `describe()`
2. Pembersihan dan preprocessing
   - `isnull().sum()`
   - `duplicated().sum()`
   - `dropna()`
   - `drop_duplicates()`
   - drop kolom ID, address, dan date
   - `LabelEncoder` untuk fitur kategorikal
3. Clustering
   - preprocessing dataset
   - Elbow Method dengan `KElbowVisualizer`
   - `KMeans`
   - `joblib.dump()` ke `model_clustering.joblib`
4. Interpretasi clustering
   - ringkasan mean, min, max fitur numerik
   - interpretasi karakteristik tiap cluster
   - export data training dengan kolom `Target`
5. Klasifikasi
   - `train_test_split()`
   - `DecisionTreeClassifier`
   - `joblib.dump()` ke `decision_tree_model.h5`

## Struktur Submission Yang Dikirim

Folder yang perlu di-upload ke reviewer adalah `BMLP_Muhammad-Fajri`, bukan seluruh folder `Machine Learning`.
Folder submission ini sengaja dibersihkan otomatis oleh pipeline agar hanya menyisakan file yang diminta reviewer.

```text
BMLP_Muhammad-Fajri/
|-- [Clustering]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb
|-- [Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb
|-- model_clustering.h5
|-- decision_tree_model.h5
`-- data_clustering.csv
```

## Checklist File Reviewer

- `[Clustering]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb`
- `[Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb`
- `model_clustering.h5`
- `decision_tree_model.h5`
- `data_clustering.csv`

## File Penting

- `run_pipeline.py` untuk menjalankan pipeline end-to-end
- `src/transaction_ml_pipeline.py` untuk logika utama project
- `notebooks/submission_walkthrough.ipynb` untuk notebook submission
- `data/processed/transactions_training_with_target.csv` untuk data hasil preprocessing + target cluster
- `reports/cluster_profile_summary.csv` untuk ringkasan karakteristik cluster
- `BMLP_Muhammad-Fajri/` untuk folder submission final yang siap dikirim ulang

## Cara Menjalankan

```powershell
cd "D:\code\Machine Learning"
python run_pipeline.py
python -m unittest discover -s tests
```

## Output yang Dihasilkan

- Dataset mentah: `data/raw/financial_transactions.csv`
- Dataset preprocessing: `data/processed/transactions_preprocessed.csv`
- Dataset training dengan target: `data/processed/transactions_training_with_target.csv`
- Model clustering: `models/model_clustering.joblib`
- Model clustering reviewer-ready: `models/model_clustering.h5`
- Model klasifikasi: `models/decision_tree_model.h5`
- Grafik elbow: `reports/figures/elbow_method.png`
- Distribusi cluster: `reports/figures/cluster_distribution.png`
- Paket submission final:
  - `BMLP_Muhammad-Fajri/[Clustering]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb`
  - `BMLP_Muhammad-Fajri/[Klasifikasi]_Submission_Akhir_BMLP_Muhammad_Fajri.ipynb`
  - `BMLP_Muhammad-Fajri/model_clustering.h5`
  - `BMLP_Muhammad-Fajri/decision_tree_model.h5`
  - `BMLP_Muhammad-Fajri/data_clustering.csv`

## Sumber Dataset

- Kaggle: `aryan208/financial-transactions-dataset-for-fraud-detection`
- File sumber asli: `financial_fraud_detection_dataset.csv`

## Catatan

- Dataset Kaggle aslinya berisi sekitar `5.000.000` baris, sehingga project ini memakai sample kerja `30.000` baris yang di-stage ulang ke folder project agar pipeline tetap cepat dijalankan lokal.
- Kolom dataset di-rename ke format yang lebih dekat dengan checklist submission, misalnya `transaction_id` menjadi `TransactionID` dan `timestamp` menjadi `TransactionDate`.
- Kolom `FraudType` dan `TimeSinceLastTransaction` dihapus sebelum `dropna()` karena missing value-nya sangat besar dan akan membuat terlalu banyak data hilang bila dipertahankan apa adanya.
- Pipeline sekarang juga otomatis menyalin notebook submission dan memvalidasi artefak wajib di folder `BMLP_Muhammad-Fajri`.
