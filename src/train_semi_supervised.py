"""
train_semi_supervised.py
========================
Training model Semi-Supervised Learning:
  - Label Propagation
  - Label Spreading

untuk klasifikasi tingkat obesitas.
30% data train dilabeli sebagai UNLABELED (-1) sesuai desain eksperimen.

Cara penggunaan:
    python src/train_semi_supervised.py
"""

import os
import time
import joblib
import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocessing import run_preprocessing

# ── Konstanta ──────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
UNLABELED    = -1
UNLABELED_RATIO = 0.30   # 30% data train dibuat tidak berlabel
ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III"
]
MODELS_DIR = "models"
DATA_PATH  = "data/ObesityDataSet_raw_and_data_sinthetic.csv"


def make_semi_labels(y_train: np.ndarray,
                     unlabeled_ratio: float = 0.30,
                     random_seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Membuat array label semi-supervised dengan sebagian label diset ke -1.

    Parameters
    ----------
    y_train        : array label asli
    unlabeled_ratio: proporsi data yang akan dibuat tidak berlabel
    random_seed    : seed numpy

    Returns
    -------
    y_semi : array label dengan nilai -1 untuk data tidak berlabel
    """
    np.random.seed(random_seed)
    y_semi = y_train.copy()
    idx_ul = np.random.choice(len(y_semi),
                               int(len(y_semi) * unlabeled_ratio),
                               replace=False)
    y_semi[idx_ul] = UNLABELED
    n_labeled   = (y_semi != UNLABELED).sum()
    n_unlabeled = (y_semi == UNLABELED).sum()
    print(f"  Semi-supervised split: {n_labeled} berlabel | {n_unlabeled} tidak berlabel "
          f"({unlabeled_ratio*100:.0f}% unlabeled)")
    return y_semi


def train_label_propagation(X_train_sc, y_semi, X_test_sc, y_test,
                             kernel: str = "rbf",
                             gamma: float = 20,
                             max_iter: int = 1000) -> dict:
    """
    Melatih model Label Propagation dan mengembalikan dict hasil metrik.

    Parameters
    ----------
    X_train_sc : array-like — seluruh data train (berlabel + tidak berlabel)
    y_semi     : array-like — label semi-supervised (-1 = tidak berlabel)
    X_test_sc  : array-like
    y_test     : array-like
    kernel     : str   — 'rbf' atau 'knn'
    gamma      : float — parameter kernel RBF
    max_iter   : int   — iterasi maksimum propagasi

    Returns
    -------
    result : dict berisi model, prediksi, dan metrik evaluasi
    """
    print("=" * 60)
    print("TRAINING: Label Propagation (Semi-supervised)")
    print("=" * 60)
    print(f"  Hyperparameter : kernel={kernel}, gamma={gamma}, max_iter={max_iter}")
    print("  ⏳ Label Propagation ~30 detik, harap tunggu ...")

    model = LabelPropagation(
        kernel=kernel,
        gamma=gamma,
        max_iter=max_iter
    )

    t0 = time.time()
    model.fit(X_train_sc, y_semi)
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
    print(f"  Iterasi konvergen: {model.n_iter_}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=ORDER_TARGET, zero_division=0))

    return {
        "model":       model,
        "y_pred":      y_pred,
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1_weighted": f1w,
        "f1_macro":    f1m,
        "train_ms":    t_train,
    }


def train_label_spreading(X_train_sc, y_semi, X_test_sc, y_test,
                           kernel: str = "rbf",
                           alpha: float = 0.2,
                           max_iter: int = 1000) -> dict:
    """
    Melatih model Label Spreading dan mengembalikan dict hasil metrik.

    Parameters
    ----------
    X_train_sc : array-like — seluruh data train (berlabel + tidak berlabel)
    y_semi     : array-like — label semi-supervised (-1 = tidak berlabel)
    X_test_sc  : array-like
    y_test     : array-like
    kernel     : str   — 'rbf' atau 'knn'
    alpha      : float — clamping factor (0=hard clamp, 1=full spread)
    max_iter   : int   — iterasi maksimum propagasi

    Returns
    -------
    result : dict berisi model, prediksi, dan metrik evaluasi
    """
    print("=" * 60)
    print("TRAINING: Label Spreading (Semi-supervised)")
    print("=" * 60)
    print(f"  Hyperparameter : kernel={kernel}, alpha={alpha}, max_iter={max_iter}")
    print("  ⏳ Label Spreading ~30 detik, harap tunggu ...")

    model = LabelSpreading(
        kernel=kernel,
        alpha=alpha,
        max_iter=max_iter
    )

    t0 = time.time()
    model.fit(X_train_sc, y_semi)
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
    print(f"  Iterasi konvergen: {model.n_iter_}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                  target_names=ORDER_TARGET, zero_division=0))

    return {
        "model":       model,
        "y_pred":      y_pred,
        "accuracy":    acc,
        "precision":   prec,
        "recall":      rec,
        "f1_weighted": f1w,
        "f1_macro":    f1m,
        "train_ms":    t_train,
    }


def main():
    import glob

    csv_files = glob.glob("data/**/*.csv", recursive=True) + glob.glob("data/*.csv")
    obesity_files = [f for f in csv_files if "obesity" in f.lower()
                     and "cleaned" not in f.lower()]
    data_path = obesity_files[0] if obesity_files else DATA_PATH

    X, y, scaler, oe = run_preprocessing(
        data_path, models_dir=MODELS_DIR,
        clean_data_path="data/obesity_cleaned_data.csv"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("✓ StandardScaler disimpan.")

    # Buat label semi-supervised (30% unlabeled)
    y_semi = make_semi_labels(y_train.values, unlabeled_ratio=UNLABELED_RATIO)

    # ── Training Label Propagation ────────────────────────────────────────────
    res_lp = train_label_propagation(
        X_train_sc.values, y_semi, X_test_sc.values, y_test
    )
    lp_path = os.path.join(MODELS_DIR, "labelpropagation.joblib")
    joblib.dump(res_lp["model"], lp_path)
    print(f"\n✓ Model Label Propagation disimpan ke: {lp_path}")

    # ── Training Label Spreading ──────────────────────────────────────────────
    res_ls = train_label_spreading(
        X_train_sc.values, y_semi, X_test_sc.values, y_test
    )
    ls_path = os.path.join(MODELS_DIR, "labelspreading.joblib")
    joblib.dump(res_ls["model"], ls_path)
    print(f"\n✓ Model Label Spreading disimpan ke: {ls_path}")

    # ── Ringkasan ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RINGKASAN SEMI-SUPERVISED")
    print("=" * 60)
    summary = pd.DataFrame([
        {
            "Model":        "LabelPropagation",
            "Accuracy":     res_lp["accuracy"],
            "F1_weighted":  res_lp["f1_weighted"],
            "F1_macro":     res_lp["f1_macro"],
            "Train_ms":     res_lp["train_ms"],
        },
        {
            "Model":        "LabelSpreading",
            "Accuracy":     res_ls["accuracy"],
            "F1_weighted":  res_ls["f1_weighted"],
            "F1_macro":     res_ls["f1_macro"],
            "Train_ms":     res_ls["train_ms"],
        },
    ])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
