"""
inference.py
============
Modul inferensi terpusat (single source of truth) untuk aplikasi prediksi
tingkat obesitas. Dipakai bersama oleh app_streamlit.py dan
obesity_prediction_gradio.py agar logika preprocessing & prediksi tidak
terduplikasi dan tetap konsisten dengan pipeline training (preprocessing.py,
train_optimization.py).

Model default: model TERBAIK hasil perbandingan 11 algoritma (models/best_model.joblib),
dengan fallback otomatis ke model lain (SVM/KNN/NaiveBayes optimized) jika tidak ditemukan.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

ORDER_TARGET = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III",
]

CLASS_LABELS = {
    "Insufficient_Weight": "Berat Badan Kurang",
    "Normal_Weight":        "Berat Badan Normal",
    "Overweight_Level_I":   "Overweight Level I",
    "Overweight_Level_II":  "Overweight Level II",
    "Obesity_Type_I":       "Obesitas Tipe I",
    "Obesity_Type_II":      "Obesitas Tipe II",
    "Obesity_Type_III":     "Obesitas Tipe III (Ekstrem)",
}

REKOMENDASI = {
    "Insufficient_Weight": "Tingkatkan asupan kalori bergizi dan konsultasi dengan ahli gizi.",
    "Normal_Weight":        "Pertahankan pola makan seimbang dan aktivitas fisik rutin.",
    "Overweight_Level_I":   "Mulai kurangi makanan tinggi kalori, tingkatkan aktivitas fisik.",
    "Overweight_Level_II":  "Disarankan konsultasi dokter dan olahraga terstruktur.",
    "Obesity_Type_I":       "Segera konsultasi dokter/ahli gizi untuk program penurunan berat badan.",
    "Obesity_Type_II":      "Penanganan medis diperlukan; konsultasikan dengan dokter spesialis.",
    "Obesity_Type_III":     "Diperlukan intervensi medis segera dari dokter spesialis obesitas.",
}

FEATURE_COLS = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Automobile", "MTRANS_Bike", "MTRANS_Motorbike",
    "MTRANS_Public_Transportation", "MTRANS_Walking",
]

DATA_BOUNDS = {
    "Age": (14.0, 61.0), "Height": (1.45, 1.98), "Weight": (39.0, 173.0),
    "FCVC": (1.0, 3.0), "NCP": (1.0, 4.0), "CH2O": (1.0, 3.0),
    "FAF": (0.0, 3.0), "TUE": (0.0, 2.0),
}

CANDIDATE_MODELS = [
    "best_model.joblib",
    "svm_optimized.joblib",
    "knn_optimized.joblib",
    "naivebayes_optimized.joblib",
]


def load_artifacts():
    """Muat model terbaik (atau fallback) + scaler + ordinal encoder."""
    model, model_name = None, None
    for fname in CANDIDATE_MODELS:
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            model = joblib.load(fpath)
            model = model.get("model", model) if isinstance(model, dict) else model
            model_name = fname.replace(".joblib", "")
            break

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    oe_path = os.path.join(MODELS_DIR, "ordinal_encoder.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    oe = joblib.load(oe_path) if os.path.exists(oe_path) else None

    return model, model_name, scaler, oe


def preprocess_input(gender_male, age, height, weight, family_history, favc,
                      fcvc, ncp, caec_level, smoke, ch2o, scc, faf, tue,
                      calc_level, mtrans):
    """
    Susun satu baris fitur sesuai pipeline preprocessing.py.
    Parameter kategori berupa nilai yang SUDAH dipetakan ke kode internal:
      gender_male : bool
      family_history, favc, smoke, scc : bool
      caec_level, calc_level : int 0-3 (no/Sometimes/Frequently/Always)
      mtrans : salah satu dari ["Automobile","Bike","Motorbike","Public_Transportation","Walking"]
    """
    mtrans_options = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
    mtrans_enc = {f"MTRANS_{m}": int(m == mtrans) for m in mtrans_options}

    row = {
        "Gender": int(gender_male),
        "Age": float(age), "Height": float(height), "Weight": float(weight),
        "family_history_with_overweight": int(family_history),
        "FAVC": int(favc), "FCVC": float(fcvc), "NCP": float(ncp),
        "CAEC": float(caec_level), "SMOKE": int(smoke), "CH2O": float(ch2o),
        "SCC": int(scc), "FAF": float(faf), "TUE": float(tue), "CALC": float(calc_level),
        **mtrans_enc,
    }
    return pd.DataFrame([row])[FEATURE_COLS].astype(float)


def predict(df_input: pd.DataFrame, model, scaler):
    """Kembalikan (label_kelas, array_probabilitas[7])."""
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)
    else:
        X = df_input
    pred_idx = int(model.predict(X)[0])
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X)[0])
    else:
        probs = np.zeros(7); probs[pred_idx] = 1.0
    return ORDER_TARGET[pred_idx], probs


def bmi_info(height, weight):
    bmi = float(weight) / (float(height) ** 2)
    if bmi < 18.5:
        cat = "Berat Badan Kurang (< 18.5)"
    elif bmi < 25.0:
        cat = "Normal (18.5–24.9)"
    elif bmi < 30.0:
        cat = "Overweight (25–29.9)"
    elif bmi < 40.0:
        cat = "Obesitas (30–39.9)"
    else:
        cat = "Obesitas Ekstrem (>= 40)"
    return round(bmi, 2), cat
