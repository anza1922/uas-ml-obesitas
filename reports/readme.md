UTS Pembelajaran Mesin — Klasifikasi Tingkat Obesitas

Universitas Dian Nuswantoro | Semester Genap 2025/2026
Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition  
Target: NObeyesdad (7 kelas multiclass)  
Algoritma:KNN, Naive Bayes, Decision Tree  Extra Trees, Random Forest, LightGBM, XGBoost, Logistic Regression, SVM, LabelPropagation, LabelSpreading

 Struktur Proyek

```
ml-uts-<A11.2024.15791>-<Anza_Ali_S>/
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv   # dataset mentah
├── src/
│   ├── preprocessing.py        # load + preprocessing pipeline
│   ├── feature_selection.py    # seleksi atribut (opsional)
│   ├── train_knn.py            # training KNN
│   ├── train_decision_tree.py  # training Decision Tree
│   ├── train_naive_bayes.py    # training Naive Bayes
│   └── evaluate.py             # evaluasi & visualisasi komparatif
├── models/
│   └── best_model.joblib       # model terbaik 
├── notebooks/
│   └── UTS_ML_<nim>.ipynb      # notebook lengkap 11 model
├── reports/
│   ├── laporan.pdf
│   └── readme.md
└── requirements.txt
```


Cara Menjalankan

1. Install dependensi
```bash
pip install -r requirements.txt
```

2. Letakkan dataset
Taruh file `ObesityDataSet_raw_and_data_sinthetic.csv` di folder `data/`.

3. Jalankan preprocessing saja
```bash
python src/preprocessing.py
```

4. Training model individual
```bash
python src/train_knn.py
python src/train_naive_bayes.py
python src/train_decision_tree.py
python src/train_extra_trees.py
python src/train_lightgbm.py
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/train_semi_supervised.py
python src/train_svm.py
python src/train_xgboost.py
```

5. Evaluasi semua model (butuh semua .joblib di models/)
```bash
python src/evaluate.py
```

6. Seleksi fitur (opsional)
```bash

python src/feature_selection.py
```


Dependensi Utama
- scikit-learn ≥ 1.3
- lightgbm ≥ 4.0
- xgboost ≥ 2.0
- pandas, numpy, matplotlib, seaborn, joblib
