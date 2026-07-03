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
uas-ml-kelulusan-A11.2024.15791-Anza_Ali_S/
├── data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv   # dataset mentah
│   ├── obesity_cleaned_data.csv                    # dataset setelah preprocessing
│   ├── data_dictionary.md                          # kamus data
│   └── source_dataset.md                           # sumber & lisensi dataset
├── notebooks/
│   ├── Soal02_Audit_Dataset_Preprocessing_Pipeline.ipynb
│   ├── Soal03_Baseline_KNN_NaiveBayes_SVM.ipynb
│   ├── Soal04_Optimasi_Perbandingan_Model.ipynb
│   └── Soal05_Capstone_Aplikasi_Laporan.ipynb
├── src/
│   ├── preprocessing.py          # pipeline preprocessing
│   ├── inference.py              # modul inferensi terpusat
│   ├── feature_selection.py      # seleksi fitur (ExtraTrees, SelectKBest, RFE)
│   ├── evaluate.py               # evaluasi & visualisasi komparatif
│   ├── train_knn.py              # training KNN baseline
│   ├── train_naive_bayes.py      # training Naive Bayes baseline
│   ├── train_svm.py              # training SVM baseline
│   ├── train_optimization.py     # GridSearchCV untuk 3 model wajib
│   ├── train_decision_tree.py
│   ├── train_extra_trees.py
│   ├── train_random_forest.py
│   ├── train_lightgbm.py
│   ├── train_xgboost.py
│   ├── train_logistic_regression.py
│   ├── train_semi_supervised.py  # LabelPropagation & LabelSpreading
│   └── obesity_prediction_gradio.py  # Gradio app (versi src)
├── models/
│   ├── best_obesity_model.joblib       # model terbaik (SVM optimized)
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
│   ├── all_experiment_results.csv
│   ├── eleven_model_comparison.csv
│   ├── error_analysis.csv
│   ├── classification_reports.json
│   └── *.png  (grafik eksperimen)
├── presentation/
│   └── presentasi_uas_ml.pdf
├── report/
│   └── laporan_uas_ml_kelulusan.pdf
├── app_streamlit.py      # Aplikasi Streamlit (jalankan: streamlit run app_streamlit.py)
├── app_gradio.py         # Aplikasi Gradio   (jalankan: python app_gradio.py)
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Training ulang model (opsional, model .joblib sudah tersedia)
```bash
# Training baseline 3 model wajib
python src/train_knn.py
python src/train_naive_bayes.py
python src/train_svm.py

# Optimasi dengan GridSearchCV
python src/train_optimization.py

# Training semua 11 model
python src/train_decision_tree.py
python src/train_extra_trees.py
python src/train_random_forest.py
python src/train_lightgbm.py
python src/train_xgboost.py
python src/train_logistic_regression.py
python src/train_semi_supervised.py
```

### 3. Jalankan aplikasi
```bash
# Streamlit (direkomendasikan)
streamlit run app_streamlit.py

# Gradio
python app_gradio.py
```

### 4. Jalankan notebook (urut)
Buka Jupyter dan jalankan berurutan:
1. `Soal02_Audit_Dataset_Preprocessing_Pipeline.ipynb`
2. `Soal03_Baseline_KNN_NaiveBayes_SVM.ipynb`
3. `Soal04_Optimasi_Perbandingan_Model.ipynb`
4. `Soal05_Capstone_Aplikasi_Laporan.ipynb`

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

**Model terbaik (wajib):** SVM optimized (kernel RBF, GridSearchCV macro-F1)  
**Model terbaik (keseluruhan 11 model):** LightGBM

---

## Catatan Etika
Aplikasi ini adalah **decision support**, bukan alat keputusan medis final.
Data bersifat anonim dan tidak boleh digunakan untuk profiling individu.

**Random seed:** 42 | **Split:** 80:20 stratified
