"""
train_optimization.py
======================
Optimasi hyperparameter (GridSearchCV) untuk KNN, Naive Bayes, dan SVM
pada klasifikasi tingkat obesitas. Membandingkan performa baseline (parameter
default/manual yang sudah dipakai di train_knn.py, train_naive_bayes.py,
train_svm.py) terhadap versi optimized (parameter terbaik hasil GridSearchCV).

Memenuhi SOAL 04 UAS Pembelajaran Mesin:
- GridSearchCV dengan scoring utama macro-F1
- Stratified k-fold cross-validation
- Tabel ringkas baseline vs optimized
- Error analysis (contoh prediksi salah, pola kelas yang sering keliru)
- Penentuan model terbaik

Cara penggunaan:
    python src/train_optimization.py

Output:
    reports/all_experiment_results.csv   -> tabel baseline vs optimized
    reports/error_analysis.csv           -> contoh prediksi salah tiap model
    models/<nama>_optimized.joblib       -> model hasil tuning terbaik
"""

import os
import sys
import time
import json
import joblib
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# Pastikan src/ ada di sys.path agar bisa dijalankan dari root maupun dari src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report
)

try:
    from preprocessing import run_preprocessing
except ImportError:
    from src.preprocessing import run_preprocessing

# ── Path otomatis ────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(_BASE_DIR, "models")
REPORTS_DIR = os.path.join(_BASE_DIR, "reports")
DATA_PATH   = os.path.join(_BASE_DIR, "data", "ObesityDataSet_raw_and_data_sinthetic.csv")

RANDOM_SEED = 42
ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III"
]

# Hyperparameter baseline (sama persis dengan default di train_knn.py,
# train_naive_bayes.py, train_svm.py) supaya perbandingan adil/apple-to-apple.
BASELINE_PARAMS = {
    "KNN":         {"n_neighbors": 7, "metric": "euclidean"},
    "NaiveBayes":  {"var_smoothing": 1e-9},
    "SVM":         {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
}

# Grid pencarian untuk GridSearchCV
PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    },
    "NaiveBayes": {
        "var_smoothing": np.logspace(-12, -1, 12),
    },
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "linear"],
    },
}

ESTIMATORS = {
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True, random_state=RANDOM_SEED),
}


def evaluate_model(model, X_test, y_test, train_ms):
    """Hitung semua metrik wajib untuk satu model yang sudah di-fit."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "train_ms": train_ms,
        "y_pred": y_pred,
    }


def run_baseline(name, estimator_cls_params, X_train, y_train, X_test, y_test):
    """Latih model dengan parameter baseline (manual, tanpa tuning)."""
    model = ESTIMATORS[name].__class__(**estimator_cls_params)
    if name == "SVM":
        model = SVC(probability=True, random_state=RANDOM_SEED, **estimator_cls_params)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_ms = (time.time() - t0) * 1000
    metrics = evaluate_model(model, X_test, y_test, train_ms)
    metrics["model"] = model
    metrics["params"] = estimator_cls_params
    return metrics


def run_optimized(name, X_train, y_train, X_test, y_test, cv):
    """Cari hyperparameter terbaik dengan GridSearchCV (scoring=macro-F1)."""
    print(f"\n  -> Menjalankan GridSearchCV untuk {name} (scoring=f1_macro) ...")
    grid = GridSearchCV(
        estimator=ESTIMATORS[name],
        param_grid=PARAM_GRIDS[name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    train_ms = (time.time() - t0) * 1000

    print(f"     Best params  : {grid.best_params_}")
    print(f"     Best CV F1-macro : {grid.best_score_:.4f}")

    metrics = evaluate_model(grid.best_estimator_, X_test, y_test, train_ms)
    metrics["model"] = grid.best_estimator_
    metrics["params"] = grid.best_params_
    metrics["cv_best_score"] = grid.best_score_
    return metrics


def error_analysis(name, y_test, y_pred, X_test_index, top_n=10):
    """Kumpulkan contoh prediksi salah untuk analisis kesalahan."""
    wrong_mask = (y_test.values != y_pred)
    wrong_idx = np.where(wrong_mask)[0]
    rows = []
    for i in wrong_idx[:top_n]:
        rows.append({
            "model": name,
            "row_index": int(X_test_index[i]),
            "y_true": ORDER_TARGET[y_test.values[i]],
            "y_pred": ORDER_TARGET[y_pred[i]],
        })
    return rows


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Preprocessing (sama seperti script training lain) ──────────────────
    X, y, scaler, oe = run_preprocessing(
        DATA_PATH,
        models_dir=MODELS_DIR,
        clean_data_path=os.path.join(_BASE_DIR, "data", "obesity_cleaned_data.csv")
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Simpan scaler yang sudah di-fit
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler disimpan ke: {scaler_path}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    all_results = []
    error_rows = []

    for name in ["KNN", "NaiveBayes", "SVM"]:
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        # Baseline
        base = run_baseline(name, BASELINE_PARAMS[name], X_train_sc, y_train, X_test_sc, y_test)
        print(f"  [Baseline]  Acc={base['accuracy']:.4f}  F1-macro={base['f1_macro']:.4f}  "
              f"BalAcc={base['balanced_accuracy']:.4f}  params={base['params']}")

        # Optimized
        opt = run_optimized(name, X_train_sc, y_train, X_test_sc, y_test, cv)
        print(f"  [Optimized] Acc={opt['accuracy']:.4f}  F1-macro={opt['f1_macro']:.4f}  "
              f"BalAcc={opt['balanced_accuracy']:.4f}  params={opt['params']}")

        delta_f1 = opt["f1_macro"] - base["f1_macro"]
        print(f"  Δ F1-macro (optimized - baseline) = {delta_f1:+.4f}")

        for stage, res in [("Baseline", base), ("Optimized", opt)]:
            all_results.append({
                "Model":              name,
                "Stage":              stage,
                "Accuracy":           round(res["accuracy"], 4),
                "Precision (w)":      round(res["precision_weighted"], 4),
                "Recall (w)":         round(res["recall_weighted"], 4),
                "F1 (weighted)":      round(res["f1_weighted"], 4),
                "F1-macro":           round(res["f1_macro"], 4),
                "Balanced Acc":       round(res["balanced_accuracy"], 4),
                "Train (ms)":         round(res["train_ms"], 1),
                "params":             json.dumps(res["params"], default=str),
            })

        # Simpan model optimized
        opt_path = os.path.join(MODELS_DIR, f"{name.lower()}_optimized.joblib")
        joblib.dump(opt["model"], opt_path)
        print(f"  ✓ Model optimized disimpan: {opt_path}")

        # Error analysis dari model optimized (yang dipakai final)
        error_rows.extend(
            error_analysis(name, y_test, opt["y_pred"], X_test.index.to_numpy())
        )

    # ── Simpan tabel ringkas baseline vs optimized ──────────────────────────
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Tabel hasil eksperimen disimpan: {results_path}")

    # ── Simpan error analysis ───────────────────────────────────────────────
    error_df = pd.DataFrame(error_rows)
    error_path = os.path.join(REPORTS_DIR, "error_analysis.csv")
    error_df.to_csv(error_path, index=False)
    print(f"✓ Error analysis disimpan: {error_path}")

    # ── Ringkasan model terbaik ──────────────────────────────────────────────
    best_row = results_df[results_df["Stage"] == "Optimized"].sort_values(
        "F1-macro", ascending=False
    ).iloc[0]
    print("\n" + "=" * 70)
    print("MODEL TERBAIK (berdasarkan F1-macro optimized):")
    print(f"  {best_row['Model']}  |  F1-macro={best_row['F1-macro']:.4f}  "
          f"| BalAcc={best_row['Balanced Acc']:.4f}")
    print("=" * 70)

    # Pola kelas yang sering keliru (dari seluruh model)
    if not error_df.empty:
        print("\nPola kesalahan klasifikasi yang paling sering muncul:")
        pattern_counts = error_df.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
        print(pattern_counts.head(5))


if __name__ == "__main__":
    main()
