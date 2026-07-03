"""
evaluate.py
===========
Modul evaluasi dan visualisasi komparatif semua model.
Mencakup: tabel metrik, confusion matrix, bar chart, heatmap, ranking.

Cara penggunaan:
    from src.evaluate import evaluate_all_models, plot_comparison
    results_df = evaluate_all_models(fitted_models, X_test_sc, y_test)
    plot_comparison(results_df, preds_dict)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Konstanta ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III"
]
SHORT = ["Insuf.", "Normal", "OW-I", "OW-II", "Ob-I", "Ob-II", "Ob-III"]

WARNA = {
    "KNN":                 "#4472C4",
    "Naive Bayes":         "#ED7D31",
    "Decision Tree":       "#A9D18E",
    "Extra Trees":         "#00B050",
    "Random Forest":       "#70AD47",
    "Logistic Regression": "#FF0000",
    "SVM":                 "#7030A0",
    "LightGBM":            "#FFC000",
    "XGBoost":             "#E26B0A",
    "LabelPropagation":    "#00B0F0",
    "LabelSpreading":      "#FF6699",
}


def evaluate_single(name: str, model, X_test, y_test,
                     tipe: str = "Supervised",
                     models_dir: str = "models") -> dict:
    """
    Evaluasi satu model dan kembalikan dict metrik.
    Juga menyimpan model ke models_dir/<name_snake>.joblib.
    """
    t0 = time.time()
    y_pred = model.predict(X_test)
    t_inf = (time.time() - t0) * 1000

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1m  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Simpan model
    os.makedirs(models_dir, exist_ok=True)
    fname = name.lower().replace(" ", "_")
    joblib.dump(model, os.path.join(models_dir, f"{fname}.joblib"))

    return {
        "Model":        name,
        "Tipe":         tipe,
        "Accuracy":     round(acc, 4),
        "Precision":    round(prec, 4),
        "Recall":       round(rec, 4),
        "F1_weighted":  round(f1w, 4),
        "F1_macro":     round(f1m, 4),
        "Inference_ms": round(t_inf, 1),
        "_y_pred":      y_pred,
        "_model":       model,
    }


def evaluate_all_models(model_defs: list, X_test, y_test,
                         models_dir: str = "models",
                         save_best: bool = True) -> pd.DataFrame:
    """
    Evaluasi semua model sekaligus.

    Parameters
    ----------
    model_defs : list of (name, fitted_model, tipe_str)
    X_test     : array-like fitur test
    y_test     : array-like label test
    models_dir : folder penyimpanan model
    save_best  : simpan model terbaik (F1-macro) ke best_model.joblib

    Returns
    -------
    df_hasil : DataFrame ringkasan metrik semua model
    """
    rows = []
    preds = {}
    fitted = {}

    print("=" * 70)
    print("EVALUASI SEMUA MODEL")
    print("=" * 70)

    for name, model, tipe in model_defs:
        result = evaluate_single(name, model, X_test, y_test,
                                  tipe=tipe, models_dir=models_dir)
        preds[name]  = result.pop("_y_pred")
        fitted[name] = result.pop("_model")
        rows.append(result)
        print(f"  ✓ {name:25s} [{tipe:10s}]: "
              f"Acc={result['Accuracy']:.4f}  F1m={result['F1_macro']:.4f}  "
              f"({result['Inference_ms']:.0f}ms)")

    df_hasil = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("TABEL KOMPARATIF LENGKAP")
    print("=" * 70)
    print(df_hasil.drop(columns=[]).to_string(index=False))

    # Simpan model terbaik
    if save_best:
        best_name = df_hasil.set_index("Model")["F1_macro"].idxmax()
        best_row  = df_hasil[df_hasil["Model"] == best_name].iloc[0]
        best_path = os.path.join(models_dir, "best_model.joblib")
        joblib.dump(fitted[best_name], best_path)
        print(f"\n  ★ MODEL TERBAIK  : {best_name}")
        print(f"    F1-macro       : {best_row['F1_macro']}")
        print(f"    Accuracy       : {best_row['Accuracy']}")
        print(f"    Disimpan ke    : {best_path}")

    df_hasil._preds  = preds   # private attr for plotting
    df_hasil._fitted = fitted
    return df_hasil


# ── Visualisasi ─────────────────────────────────────────────────────────────────

def plot_confusion_matrices(df_hasil: pd.DataFrame, preds: dict,
                              y_test, reports_dir: str = "reports"):
    """Confusion matrix 3×4 grid untuk semua model."""
    os.makedirs(reports_dir, exist_ok=True)
    nama_list = df_hasil["Model"].tolist()
    cmaps_list = ["Blues", "Purples", "Oranges", "YlGn", "Greens",
                  "Reds", "PuRd", "YlOrBr", "OrRd", "BuGn", "PuBu"]

    fig, axes = plt.subplots(3, 4, figsize=(24, 17))
    fig.suptitle(
        "Confusion Matrix 11 Algoritma\nDataset: Obesity Level Estimation (7 Kelas)",
        fontsize=14, fontweight="bold"
    )
    for i, (nm, cmap) in enumerate(zip(nama_list, cmaps_list)):
        ax = axes[i // 4][i % 4]
        cm = confusion_matrix(y_test, preds[nm])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=SHORT, yticklabels=SHORT,
                    cbar=False, linewidths=0.3, annot_kws={"size": 7})
        r   = df_hasil[df_hasil["Model"] == nm].iloc[0]
        tag = " [Semi]" if "Semi" in r["Tipe"] else ""
        ax.set_title(
            f"{nm}{tag}\nAcc={r['Accuracy']:.3f}  F1m={r['F1_macro']:.3f}",
            fontsize=8, fontweight="bold"
        )
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Prediksi", fontsize=7)
        ax.set_ylabel("Aktual", fontsize=7)
    axes[2][3].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(reports_dir, "confusion_matrices.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Confusion matrices disimpan ke: {save_path}")


def plot_comparison(df_hasil: pd.DataFrame, reports_dir: str = "reports"):
    """Bar chart + heatmap perbandingan metrik semua model."""
    os.makedirs(reports_dir, exist_ok=True)
    nama_list  = df_hasil["Model"].tolist()
    warna_list = [WARNA.get(n, "#999999") for n in nama_list]
    metrik     = ["Accuracy", "F1_weighted", "F1_macro"]

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    fig2.suptitle("Perbandingan 11 Algoritma — Obesity Dataset",
                   fontsize=13, fontweight="bold")

    x = np.arange(len(metrik))
    w = 0.07
    for i, (nm, c) in enumerate(zip(nama_list, warna_list)):
        vals = [df_hasil[df_hasil["Model"] == nm][m].values[0] for m in metrik]
        bars = axes2[0].bar(x + i * w, vals, w, label=nm, color=c,
                             alpha=0.87, edgecolor="white")
        for bar, v in zip(bars, vals):
            axes2[0].text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.004,
                          f"{v:.2f}", ha="center", va="bottom",
                          fontsize=5, rotation=90)
    axes2[0].set_xticks(x + w * 5)
    axes2[0].set_xticklabels(metrik, fontsize=11)
    axes2[0].set_ylim(0, 1.15)
    axes2[0].legend(fontsize=7, ncol=2)
    axes2[0].set_title("Bar Chart Semua Metrik", fontweight="bold")
    axes2[0].grid(axis="y", alpha=0.3)

    df_hm = df_hasil.set_index("Model")[metrik]
    sns.heatmap(df_hm, annot=True, fmt=".3f", cmap="YlGn", ax=axes2[1],
                linewidths=0.5, vmin=0.3, vmax=1.0, annot_kws={"size": 9})
    axes2[1].set_title("Heatmap Metrik", fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(reports_dir, "komparatif.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Plot komparatif disimpan ke: {save_path}")


def plot_ranking(df_hasil: pd.DataFrame, reports_dir: str = "reports"):
    """Ranking F1-macro + trade-off waktu vs performa."""
    os.makedirs(reports_dir, exist_ok=True)
    nama_list  = df_hasil["Model"].tolist()
    warna_list = [WARNA.get(n, "#999999") for n in nama_list]

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    fig3.suptitle("Ranking & Trade-off: Kecepatan vs Performa",
                   fontsize=12, fontweight="bold")

    df_sorted = df_hasil.sort_values("F1_macro", ascending=True)
    c_sorted  = [WARNA.get(n, "#999999") for n in df_sorted["Model"]]
    bars = axes3[0].barh(df_sorted["Model"], df_sorted["F1_macro"],
                          color=c_sorted, alpha=0.87, edgecolor="white")
    for bar, val in zip(bars, df_sorted["F1_macro"]):
        axes3[0].text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                      f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
    axes3[0].set_xlabel("F1-score (macro)")
    axes3[0].set_title("Ranking F1-macro", fontweight="bold")
    axes3[0].set_xlim(0.3, 1.05)
    axes3[0].axvline(0.95, color="red", ls="--", alpha=0.5)
    axes3[0].grid(axis="x", alpha=0.3)

    for nm, c in zip(nama_list, warna_list):
        r  = df_hasil[df_hasil["Model"] == nm].iloc[0]
        mk = "*" if "Semi" in r["Tipe"] else (
             "D" if nm in ["LightGBM", "XGBoost"] else "o")
        axes3[1].scatter(r["Inference_ms"], r["F1_macro"],
                          color=c, s=180, zorder=5, marker=mk,
                          edgecolors="white", lw=0.5)
        axes3[1].annotate(nm, (r["Inference_ms"], r["F1_macro"]),
                           textcoords="offset points", xytext=(7, 4), fontsize=7.5)
    axes3[1].set_xlabel("Waktu Inferensi (ms, log scale)")
    axes3[1].set_ylabel("F1-macro")
    axes3[1].set_title("Trade-off Kecepatan vs Performa", fontweight="bold")
    axes3[1].set_xscale("log")
    axes3[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(reports_dir, "ranking.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Plot ranking disimpan ke: {save_path}")


def run_full_evaluation(model_defs: list, X_test, y_test,
                         models_dir: str = "models",
                         reports_dir: str = "reports") -> pd.DataFrame:
    """
    Menjalankan evaluasi lengkap + semua visualisasi.

    Parameters
    ----------
    model_defs : list of (name, fitted_model, tipe_str)
    """
    df_hasil = evaluate_all_models(model_defs, X_test, y_test,
                                    models_dir=models_dir)
    preds = df_hasil._preds

    plot_confusion_matrices(df_hasil, preds, y_test, reports_dir=reports_dir)
    plot_comparison(df_hasil, reports_dir=reports_dir)
    plot_ranking(df_hasil, reports_dir=reports_dir)

    return df_hasil


# ── Jalankan langsung (contoh load model dari disk) ────────────────────────────
if __name__ == "__main__":
    import glob

    print("Memuat model dari folder models/ ...")
    model_files = glob.glob("models/*.joblib")
    exclude = {"best_model.joblib", "scaler.joblib", "ordinal_encoder.joblib"}

    NAME_MAP = {
        "knn":                  ("KNN",                 "Supervised"),
        "naive_bayes":          ("Naive Bayes",         "Supervised"),
        "decision_tree":        ("Decision Tree",       "Supervised"),
        "extra_trees":          ("Extra Trees",         "Supervised"),
        "random_forest":        ("Random Forest",       "Supervised"),
        "logistic_regression":  ("Logistic Regression", "Supervised"),
        "svm":                  ("SVM",                 "Supervised"),
        "lightgbm":             ("LightGBM",            "Supervised"),
        "xgboost":              ("XGBoost",             "Supervised"),
        "labelpropagation":     ("LabelPropagation",    "Semi-sup"),
        "labelspreading":       ("LabelSpreading",      "Semi-sup"),
    }

    # Load data bersih
    df_clean = pd.read_csv("data/obesity_cleaned_data.csv")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df_clean.drop(columns=["target"]).astype(float)
    y = df_clean["target"].astype(int)

    scaler = joblib.load("models/scaler.joblib")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model_defs = []
    for fpath in sorted(model_files):
        basename = os.path.basename(fpath).replace(".joblib", "")
        if basename in exclude:
            continue
        if basename in NAME_MAP:
            name, tipe = NAME_MAP[basename]
            model = joblib.load(fpath)
            model_defs.append((name, model, tipe))

    if model_defs:
        run_full_evaluation(model_defs, X_test_sc, y_test)
    else:
        print("Tidak ada model ditemukan. Jalankan train_*.py terlebih dahulu.")
