"""
train_decision_tree.py
======================
Training model Decision Tree untuk klasifikasi tingkat obesitas.

Cara penggunaan:
    python src/train_decision_tree.py
"""

import os
import time
import joblib
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocessing import run_preprocessing

# ── Konstanta ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III"
]
MODELS_DIR = "models"
DATA_PATH  = "data/ObesityDataSet_raw_and_data_sinthetic.csv"


def train_decision_tree(X_train_sc, y_train, X_test_sc, y_test,
                         max_depth: int = 10,
                         criterion: str = "gini") -> dict:
    """
    Melatih model Decision Tree dan mengembalikan dict hasil metrik.

    Parameters
    ----------
    X_train_sc : array-like
    y_train    : array-like
    X_test_sc  : array-like
    y_test     : array-like
    max_depth  : int    — kedalaman maksimum pohon
    criterion  : str    — "gini" atau "entropy"

    Returns
    -------
    result : dict berisi model, prediksi, dan metrik evaluasi
    """
    print("=" * 60)
    print("TRAINING: Decision Tree")
    print("=" * 60)
    print(f"  Hyperparameter : max_depth={max_depth}, criterion={criterion}")

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=RANDOM_SEED
    )

    t0 = time.time()
    model.fit(X_train_sc, y_train)
    t_train = (time.time() - t0) * 1000

    y_pred = model.predict(X_test_sc)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1m  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n  Accuracy    : {acc:.4f}")
    print(f"  Precision   : {prec:.4f}  (weighted)")
    print(f"  Recall      : {rec:.4f}  (weighted)")
    print(f"  F1-weighted : {f1w:.4f}")
    print(f"  F1-macro    : {f1m:.4f}")
    print(f"  Waktu train : {t_train:.1f} ms")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=ORDER_TARGET, zero_division=0))

    # ── Feature Importance ────────────────────────────────────────────────────
    feat_names = (X_train_sc.columns.tolist()
                  if hasattr(X_train_sc, "columns") else list(range(X_train_sc.shape[1])))
    importances = pd.Series(model.feature_importances_, index=feat_names)
    importances = importances.sort_values(ascending=False)
    print("  Top-10 Feature Importance:")
    print(importances.head(10).to_string())

    # ── Cross-validation ──────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(model, X_train_sc, y_train,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\n  5-Fold CV F1-macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return {
        "model":       model,
        "y_pred":      y_pred,
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1_weighted": f1w,
        "f1_macro":    f1m,
        "train_ms":    t_train,
        "cv_mean":     cv_scores.mean(),
        "cv_std":      cv_scores.std(),
        "importances": importances,
    }


def main():
    import glob

    # Cari file CSV dataset
    csv_files = glob.glob("data/**/*.csv", recursive=True) + glob.glob("data/*.csv")
    obesity_files = [f for f in csv_files if "obesity" in f.lower()
                     and "cleaned" not in f.lower()]
    data_path = obesity_files[0] if obesity_files else DATA_PATH

    # Preprocessing
    X, y, scaler, oe = run_preprocessing(
        data_path,
        models_dir=MODELS_DIR,
        clean_data_path="data/obesity_cleaned_data.csv"
    )

    # Split data 80:20 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Scale fitur (fit HANYA pada train)
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    # Simpan scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("✓ StandardScaler disimpan.")

    # Training Decision Tree
    result = train_decision_tree(X_train_sc, y_train, X_test_sc, y_test)

    # Simpan model
    model_path = os.path.join(MODELS_DIR, "decision_tree.joblib")
    joblib.dump(result["model"], model_path)
    print(f"\n✓ Model Decision Tree disimpan ke: {model_path}")


if __name__ == "__main__":
    main()
