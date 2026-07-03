"""
preprocessing.py
================
Modul preprocessing dataset Obesity Level Estimation.
Menangani: duplikat, encoding, outlier, scaling, dan penyimpanan scaler.

Cara penggunaan:
    from src.preprocessing import run_preprocessing
    X, y, scaler, oe = run_preprocessing("data/ObesityDataSet_raw_and_data_sinthetic.csv")
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ── Path otomatis (tidak bergantung dari mana file dijalankan) ─────────────────
# src/ → naik satu level → obesity_project/
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR_DEFAULT = os.path.join(BASE_DIR, "models")

# ── Konstanta ──────────────────────────────────────────────────────────────────
ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II", "Obesity_Type_III"
]

NUM_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]


def load_data(data_path: str) -> pd.DataFrame:
    """Memuat dataset dari file CSV."""
    df = pd.read_csv(data_path)
    print(f"[Load] Dataset dimuat dari: {data_path}")
    print(f"       Dimensi awal : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"       Missing value: {df.isnull().sum().sum()}")
    print(f"       Duplikat     : {df.duplicated().sum()}")
    return df


def preprocess(df_raw: pd.DataFrame,
               models_dir: str = "models",
               clean_data_path: str = "data/obesity_cleaned_data.csv"):
    """
    Pipeline preprocessing lengkap.

    Parameters
    ----------
    df_raw        : DataFrame mentah hasil load_data()
    models_dir    : folder untuk menyimpan scaler & ordinal encoder
    clean_data_path : path output CSV bersih

    Returns
    -------
    X      : pd.DataFrame  — fitur (belum di-scale)
    y      : pd.Series     — target integer 0-6
    scaler : StandardScaler (sudah di-fit pada X_train, dikembalikan unfitted)
    oe     : OrdinalEncoder (sudah di-fit)
    """
    os.makedirs(models_dir, exist_ok=True)
    df = df_raw.copy()

    # ── Langkah 1: Hapus duplikat ──────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[1] Hapus {n_dup} duplikat → {len(df)} baris tersisa")

    # ── Langkah 2: Binary encoding ─────────────────────────────────────────────
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    for col in ["family_history_with_overweight", "FAVC", "SMOKE", "SCC"]:
        df[col] = df[col].map({"yes": 1, "no": 0})
    print("[2] Binary encoding: Gender, family_history, FAVC, SMOKE, SCC")

    # ── Langkah 3: Ordinal encoding CAEC & CALC ────────────────────────────────
    ORDER_ORD = [["no", "Sometimes", "Frequently", "Always"]] * 2
    oe = OrdinalEncoder(categories=ORDER_ORD)
    df[["CAEC", "CALC"]] = oe.fit_transform(df[["CAEC", "CALC"]])
    print("[3] Ordinal encoding: CAEC & CALC (no=0, Sometimes=1, Frequently=2, Always=3)")

    # Simpan ordinal encoder
    oe_path = os.path.join(models_dir, "ordinal_encoder.joblib")
    joblib.dump(oe, oe_path)
    print(f"    OrdinalEncoder disimpan ke: {oe_path}")

    # ── Langkah 4: One-hot encoding MTRANS ────────────────────────────────────
    df = pd.get_dummies(df, columns=["MTRANS"], prefix="MTRANS", drop_first=False)
    mtrans_cols = [c for c in df.columns if c.startswith("MTRANS_")]
    df[mtrans_cols] = df[mtrans_cols].astype(int)
    print(f"[4] One-hot MTRANS → {len(mtrans_cols)} kolom baru: {mtrans_cols}")

    # ── Langkah 5: Winsorizing outlier ────────────────────────────────────────
    for col in NUM_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        df[col] = df[col].clip(Q1 - 1.5 * (Q3 - Q1), Q3 + 1.5 * (Q3 - Q1))
    print(f"[5] Winsorizing outlier pada: {NUM_COLS}")

    # ── Langkah 6: Encoding target ────────────────────────────────────────────
    tmap = {v: i for i, v in enumerate(ORDER_TARGET)}
    df["target"] = df["NObeyesdad"].map(tmap)
    df = df.drop(columns=["NObeyesdad"])
    print("[6] Encoding target: Insufficient_Weight=0 … Obesity_Type_III=6")

    # Simpan data bersih
    df.to_csv(clean_data_path, index=False)
    print(f"\n✓ Data bersih disimpan ke: {clean_data_path}")

    # ── X dan y ───────────────────────────────────────────────────────────────
    X = df.drop(columns=["target"]).astype(float)
    y = df["target"].astype(int)

    scaler = StandardScaler()  # akan di-fit di train_*.py

    print(f"\n  Shape akhir : X={X.shape}, y={y.shape}")
    print("\n" + "="*55)
    print("TABEL RINGKASAN BEFORE/AFTER PREPROCESSING")
    print("="*55)
    summary = pd.DataFrame({
        "Aspek":   ["Baris", "Kolom", "Missing value", "Duplikat",
                    "Encoding target", "Skala fitur"],
        "Sebelum": [2111, 17, 0, n_dup, "String (7 label)", "Heterogen"],
        "Sesudah": [len(df), X.shape[1], 0, 0, "Integer 0–6", "StandardScaler"]
    })
    print(summary.to_string(index=False))
    print("\n✓ Preprocessing selesai!")

    return X, y, scaler, oe


def run_preprocessing(data_path: str = None,
                      models_dir: str = None,
                      clean_data_path: str = None):
    """
    Entry point tunggal: load + preprocess.
    Semua path default ke folder obesity_project/ secara otomatis.
    """
    if models_dir is None:
        models_dir = MODELS_DIR_DEFAULT
    if clean_data_path is None:
        clean_data_path = os.path.join(DATA_DIR, "obesity_cleaned_data.csv")
    if data_path is None:
        # Cari file CSV dataset secara otomatis di folder data/
        import glob
        csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
        csv_files += glob.glob(os.path.join(DATA_DIR, "*.csv"))
        obesity_files = [f for f in csv_files
                         if "obesity" in os.path.basename(f).lower()
                         and "cleaned" not in os.path.basename(f).lower()]
        if obesity_files:
            data_path = obesity_files[0]
        else:
            data_path = os.path.join(DATA_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")

    df_raw = load_data(data_path)
    return preprocess(df_raw, models_dir=models_dir, clean_data_path=clean_data_path)


# ── Jalankan langsung ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, y, scaler, oe = run_preprocessing()
    print(f"\nFitur (5 baris pertama):\n{X.head()}")
    print(f"\nTarget (5 baris pertama):\n{y.head()}")
