"""
ml_core.py
==========
Modul inti Machine Learning — mendefinisikan seluruh model, konfigurasi
hyperparameter grid, dan fungsi evaluasi yang dipakai secara konsisten
di seluruh pipeline training project ini.

Cara penggunaan:
    from src.ml_core import get_baseline_models, get_param_grids, evaluate_model
    models = get_baseline_models()
    grids  = get_param_grids()
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, balanced_accuracy_score,
                              confusion_matrix, classification_report)
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

RANDOM_SEED = 42

ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II",  "Obesity_Type_III",
]

# ── 3 MODEL WAJIB (baseline parameter) ────────────────────────────────────────
BASELINE_PARAMS = {
    "KNN":        {"n_neighbors": 7, "metric": "euclidean", "weights": "uniform"},
    "NaiveBayes": {"var_smoothing": 1e-9},
    "SVM":        {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True,
                   "random_state": RANDOM_SEED},
}

# ── HYPERPARAMETER GRID (GridSearchCV) ─────────────────────────────────────────
PARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan", "minkowski"],
    },
    "NaiveBayes": {
        "var_smoothing": np.logspace(-12, -1, 12),
    },
    "SVM": {
        "C":      [0.1, 1, 10, 100],
        "gamma":  ["scale", "auto", 0.01, 0.1],
        "kernel": ["rbf", "linear"],
    },
}


def get_baseline_models() -> dict:
    """Kembalikan dict berisi 3 model wajib dengan parameter baseline."""
    return {
        "KNN":        KNeighborsClassifier(**BASELINE_PARAMS["KNN"]),
        "NaiveBayes": GaussianNB(var_smoothing=BASELINE_PARAMS["NaiveBayes"]["var_smoothing"]),
        "SVM":        SVC(**BASELINE_PARAMS["SVM"]),
    }


def get_all_models() -> dict:
    """Kembalikan dict berisi seluruh 11 model (wajib + pembanding)."""
    models = {
        "KNN":               KNeighborsClassifier(n_neighbors=7),
        "NaiveBayes":        GaussianNB(),
        "SVM":               SVC(probability=True, random_state=RANDOM_SEED),
        "DecisionTree":      DecisionTreeClassifier(random_state=RANDOM_SEED),
        "ExtraTrees":        ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "RandomForest":      RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        "LogisticRegression":LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        "LabelPropagation":  LabelPropagation(max_iter=1000),
        "LabelSpreading":    LabelSpreading(max_iter=1000),
    }
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                              n_jobs=-1, verbose=-1)
    if HAS_XGB:
        models["XGBoost"]  = XGBClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                             n_jobs=-1, verbosity=0, use_label_encoder=False,
                                             eval_metric="mlogloss")
    return models


def get_param_grids() -> dict:
    """Kembalikan parameter grid untuk GridSearchCV."""
    return PARAM_GRIDS


def evaluate_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    """
    Fit model dan hitung 6 metrik evaluasi standar.
    Return dict berisi semua metrik + waktu training.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train_ms = round((time.time() - t0) * 1000, 1)

    y_pred = model.predict(X_test)

    return {
        "Model":          name,
        "Accuracy":       round(accuracy_score(y_test, y_pred), 4),
        "Precision (w)":  round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "Recall (w)":     round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1 (weighted)":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "F1-macro":       round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Balanced Acc":   round(balanced_accuracy_score(y_test, y_pred), 4),
        "Train (ms)":     t_train_ms,
        "_model":         model,
        "_y_pred":        y_pred,
    }


def print_report(name: str, y_test, y_pred) -> None:
    """Cetak classification report lengkap."""
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, y_pred, target_names=ORDER_TARGET, zero_division=0))


if __name__ == "__main__":
    models = get_all_models()
    print(f"Total model tersedia: {len(models)}")
    for name in models:
        print(f"  - {name}")
    print(f"\nParameter grid tersedia untuk: {list(PARAM_GRIDS.keys())}")
