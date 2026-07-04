"""
data_generator.py
=================
Modul utilitas dataset untuk project UAS Pembelajaran Mesin.
Menyediakan fungsi load, validasi, dan ringkasan audit dataset obesitas.

Cara penggunaan:
    from src.data_generator import load_dataset, audit_dataset, get_data_summary
    df = load_dataset()
    audit = audit_dataset(df)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DEFAULT_DATASET = os.path.join(DATA_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")

ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II",  "Obesity_Type_III",
]


def load_dataset(path: str = DEFAULT_DATASET) -> pd.DataFrame:
    """Load dataset mentah dari file CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    print(f"Dataset dimuat: {df.shape[0]} baris x {df.shape[1]} kolom")
    return df


def audit_dataset(df: pd.DataFrame, save_report: bool = True) -> dict:
    """
    Lakukan audit lengkap dataset:
    - Jumlah baris/kolom
    - Missing value
    - Duplikat
    - Distribusi target
    - Outlier (IQR)
    - Class imbalance
    - Potensi leakage
    """
    target_col = "NObeyesdad"
    num_cols = df.select_dtypes(include="number").columns.tolist()

    # Outlier per kolom numerik
    outlier_summary = {}
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        n_out = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
        outlier_summary[col] = {"Q1": round(Q1, 3), "Q3": round(Q3, 3), "n_outlier": n_out,
                                 "pct_outlier": round(n_out / len(df) * 100, 2)}

    target_dist = df[target_col].value_counts().to_dict() if target_col in df.columns else {}
    vc = list(target_dist.values())
    imbalance_ratio = round(max(vc) / min(vc), 4) if vc else None

    audit = {
        "dataset_path":           DEFAULT_DATASET,
        "n_rows_raw":             int(df.shape[0]),
        "n_cols_raw":             int(df.shape[1]),
        "n_duplicates_removed":   int(df.duplicated().sum()),
        "n_rows_clean":           int(df.shape[0] - df.duplicated().sum()),
        "n_features_final":       int(df.shape[1] - 1),
        "missing_values_total":   int(df.isnull().sum().sum()),
        "target_distribution_raw": target_dist,
        "class_imbalance_ratio_max_min": imbalance_ratio,
        "outlier_summary_iqr":    outlier_summary,
        "potential_leakage":      ["Height", "Weight (keduanya membentuk BMI yang near-deterministik terhadap target)"],
        "preprocessing_steps":    [
            "drop_duplicates", "binary_encoding", "ordinal_encoding_CAEC_CALC",
            "one_hot_encoding_MTRANS", "winsorize_outlier_IQR_1.5x",
            "target_encoding_0to6", "train_test_split_80_20_stratified", "standard_scaler"
        ],
        "random_seed":            42,
        "split_ratio":            "80:20 stratified",
    }

    if save_report:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out = os.path.join(REPORTS_DIR, "audit_dataset.json")
        with open(out, "w") as f:
            json.dump(audit, f, indent=2, default=str)
        print(f"Audit disimpan ke: {out}")

    return audit


def get_data_summary(df: pd.DataFrame) -> None:
    """Cetak ringkasan statistik dataset."""
    print("=" * 55)
    print("RINGKASAN DATASET")
    print("=" * 55)
    print(f"Dimensi        : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Missing value  : {df.isnull().sum().sum()}")
    print(f"Duplikat       : {df.duplicated().sum()}")
    if "NObeyesdad" in df.columns:
        print(f"\nDistribusi Target:")
        for k, v in df["NObeyesdad"].value_counts().items():
            print(f"  {k:<25}: {v}")
    print("=" * 55)


if __name__ == "__main__":
    df = load_dataset()
    get_data_summary(df)
    audit = audit_dataset(df)
    print(f"\nClass imbalance ratio: {audit['class_imbalance_ratio_max_min']}")
    print(f"Total missing values : {audit['missing_values_total']}")
