# UAS Pembelajaran Mesin — Klasifikasi Tingkat Obesitas
**Universitas Dian Nuswantoro | Semester Genap 2025/2026**  
Nama: Anza Ali S. | NIM: A11.2024.15791 | Kelompok: A11.4401

---

## Deskripsi Project
Aplikasi Machine Learning untuk memprediksi **tingkat obesitas** individu berdasarkan
kebiasaan makan dan kondisi fisik, menggunakan KNN, Naive Bayes, dan SVM (model wajib)
serta 8 model pembanding tambahan (total 11 model).

**Dataset:** UCI Obesity Levels (2111 baris, 17 fitur, 7 kelas target)

---

## Struktur Folder
```
ml-uas-A11.2024.15791-Anza_Ali_S/
├── data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv   # dataset mentah
│   ├── obesity_cleaned_data.csv                    # dataset setelah preprocessing
│   ├── data_dictionary.md                          # kamus data
│   └── source_dataset.md                           # sumber & lisensi dataset
├── notebooks/
│   └── UAS_ML_Obesitas_A11_2024_15791_Anza_Ali_S.ipynb
├── src/
│   ├── preprocessing.py       # pipeline preprocessing
│   ├── inference.py           # modul inferensi terpusat
│   ├── predict.py             # CLI prediksi & modul predict_single/predict_batch
│   ├── ml_core.py             # definisi model & fungsi evaluasi
│   ├── data_generator.py      # load & audit dataset
│   ├── evaluate.py            # evaluasi & visualisasi komparatif
│   ├── feature_selection.py   # seleksi fitur (ExtraTrees, SelectKBest, RFE)
│   ├── train.py               # training semua 11 model sekaligus
│   └── train_optimization.py  # GridSearchCV untuk 3 model wajib
├── models/
│   ├── best_model.joblib           # model terbaik keseluruhan
│   ├── best_obesity_model.joblib   # alias model terbaik
│   ├── knn_optimized.joblib
│   ├── naivebayes_optimized.joblib
│   ├── svm_optimized.joblib
│   ├── lightgbm.joblib
│   ├── xgboost.joblib
│   ├── random_forest.joblib
│   ├── extra_trees.joblib
│   ├── decision_tree.joblib
│   ├── logistic_regression_results.joblib
│   ├── labelpropagation.joblib
│   ├── labelspreading.joblib
│   ├── scaler.joblib
│   └── ordinal_encoder.joblib
├── reports/
│   ├── audit_dataset.json
│   ├── all_experiment_results.csv         # tabel baseline vs optimized 3 model wajib
│   ├── eleven_model_comparison.csv        # perbandingan seluruh 11 model
│   ├── classification_reports.json
│   ├── soal03_baseline_metrics.csv
│   ├── soal03_confusion_matrices.png
│   ├── boxplot_before.png
│   └── boxplot_after.png
├── presentation/
│   └── Presentasi_uas_ml.pdf
├── report/
│   └── Laporan_uas_ml_obesitas.pdf
├── app_streamlit.py      # Aplikasi Streamlit utama
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi Streamlit (model sudah tersedia)
```bash
streamlit run app_streamlit.py
```

### 3. Prediksi via CLI
```bash
python src/predict.py --gender L --age 25 --height 1.70 --weight 80
```

### 4. Training ulang model (opsional — model .joblib sudah tersedia)
```bash
# Training + optimasi 3 model wajib (KNN, NaiveBayes, SVM)
python src/train_optimization.py

# Training semua 11 model sekaligus
python src/train.py
python src/train.py --model KNN
```

### 5. Jalankan notebook
```bash
jupyter notebook notebooks/UAS_ML_Obesitas_A11_2024_15791_Anza_Ali_S.ipynb
```

---

## Hasil Utama

| Model | Stage | F1-macro | Balanced Accuracy |
|-------|-------|----------|-------------------|
| KNN | Baseline | 0.7942 | 0.8004 |
| KNN | Optimized | 0.8574 | 0.8585 |
| Naive Bayes | Baseline | 0.4305 | 0.5356 |
| Naive Bayes | Optimized | 0.5294 | 0.5721 |
| SVM | Baseline | 0.8520 | 0.8515 |
| **SVM** | **Optimized** | **0.9615** | **0.9611** |
| LightGBM | — | 0.9637 | 0.9628 |

**Model terbaik (wajib):** SVM optimized (kernel linear, C=10)  
**Model terbaik (keseluruhan 11 model):** LightGBM (F1-macro 0.9637)

---

## Catatan Etika
Aplikasi ini adalah **decision support**, bukan alat keputusan medis final.
Data bersifat anonim dan tidak boleh digunakan untuk profiling individu.

**Random seed:** 42 | **Split:** 80:20 stratified
