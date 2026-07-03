"""
feature_selection.py  (opsional)
=================================
Seleksi atribut untuk dataset Obesity Level.
Menyediakan beberapa metode: importance dari ExtraTrees,
SelectKBest (chi2 / f_classif), dan RFE.

Cara penggunaan:
    from src.feature_selection import select_features
    X_sel, selected_cols = select_features(X, y, method="importance", k=15)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

RANDOM_SEED = 42


def feature_importance(X: pd.DataFrame, y: pd.Series,
                        k: int = None, threshold: float = 0.01,
                        plot: bool = False) -> tuple:
    """
    Seleksi fitur berdasarkan feature importance ExtraTreesClassifier.

    Parameters
    ----------
    X         : DataFrame fitur
    y         : Series target
    k         : jumlah fitur top-k; jika None, gunakan threshold
    threshold : batas minimal importance (diabaikan jika k diberikan)
    plot      : tampilkan bar chart importance

    Returns
    -------
    X_selected  : DataFrame dengan kolom terpilih
    selected    : list nama fitur terpilih
    importances : Series importance semua fitur
    """
    et = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    et.fit(X, y)
    importances = pd.Series(et.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    if k is not None:
        selected = importances.head(k).index.tolist()
    else:
        selected = importances[importances >= threshold].index.tolist()

    print(f"[FeatureImportance] {len(selected)}/{len(X.columns)} fitur dipilih")
    print(importances.head(len(selected)).to_string())

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2ECC71" if f in selected else "#BDC3C7" for f in importances.index]
        ax.barh(importances.index[::-1], importances.values[::-1],
                color=colors[::-1], alpha=0.85, edgecolor="white")
        ax.set_title("Feature Importance (ExtraTrees)", fontweight="bold")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig("reports/feature_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✓ Plot disimpan ke reports/feature_importance.png")

    return X[selected], selected, importances


def select_kbest(X: pd.DataFrame, y: pd.Series,
                 k: int = 15, score_func=f_classif) -> tuple:
    """
    Seleksi fitur dengan SelectKBest.

    Returns
    -------
    X_selected : DataFrame
    selected   : list nama fitur terpilih
    """
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    mask = selector.get_support()
    selected = X.columns[mask].tolist()
    print(f"[SelectKBest] {len(selected)}/{len(X.columns)} fitur dipilih: {selected}")
    return X[selected], selected


def select_rfe(X: pd.DataFrame, y: pd.Series,
               k: int = 15) -> tuple:
    """
    Seleksi fitur dengan Recursive Feature Elimination (RFE)
    menggunakan LogisticRegression sebagai estimator.

    Returns
    -------
    X_selected : DataFrame
    selected   : list nama fitur terpilih
    """
    estimator = LogisticRegression(max_iter=500, random_state=RANDOM_SEED)
    rfe = RFE(estimator=estimator, n_features_to_select=k)
    rfe.fit(X, y)
    selected = X.columns[rfe.support_].tolist()
    print(f"[RFE] {len(selected)}/{len(X.columns)} fitur dipilih: {selected}")
    return X[selected], selected


def select_features(X: pd.DataFrame, y: pd.Series,
                    method: str = "importance",
                    k: int = None,
                    threshold: float = 0.01,
                    plot: bool = False) -> tuple:
    """
    Entry point seleksi fitur.

    Parameters
    ----------
    method : "importance" | "kbest" | "rfe"
    k      : jumlah fitur (default: None → pakai threshold untuk 'importance')
    """
    method = method.lower()
    if method == "importance":
        X_sel, selected, _ = feature_importance(X, y, k=k,
                                                 threshold=threshold, plot=plot)
    elif method == "kbest":
        k = k or 15
        X_sel, selected = select_kbest(X, y, k=k)
    elif method == "rfe":
        k = k or 15
        X_sel, selected = select_rfe(X, y, k=k)
    else:
        raise ValueError(f"method harus 'importance', 'kbest', atau 'rfe'. Diterima: {method}")

    print(f"\n✓ Fitur terpilih ({len(selected)}): {selected}")
    return X_sel, selected


# ── Jalankan langsung (demo) ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Butuh preprocessing.py dijalankan dulu agar ada data bersih
    df = pd.read_csv("data/obesity_cleaned_data.csv")
    X = df.drop(columns=["target"]).astype(float)
    y = df["target"].astype(int)

    print("=" * 60)
    print("DEMO: Feature Importance (ExtraTrees)")
    print("=" * 60)
    X_sel, selected = select_features(X, y, method="importance",
                                       threshold=0.01, plot=True)
    print(f"\nShape setelah seleksi: {X_sel.shape}")
