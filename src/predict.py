"""
predict.py
==========
Modul prediksi CLI dan fungsi inferensi — dapat digunakan sebagai:
1. Script command line untuk prediksi satu individu
2. Modul yang diimpor oleh app Streamlit/Gradio
3. Backend untuk REST API (FastAPI)

Cara penggunaan CLI:
    python src/predict.py --gender L --age 22 --height 1.70 --weight 85

Cara penggunaan sebagai modul:
    from src.predict import predict_single, predict_batch
    result = predict_single(gender_male=True, age=22, height=1.70, weight=85, ...)
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(BASE_DIR, "src")
# Pastikan kedua path ada di sys.path agar import bekerja dari mana saja
for _p in [BASE_DIR, _SRC_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from inference import (
        load_artifacts, preprocess_input, predict,
        bmi_info, ORDER_TARGET, CLASS_LABELS, REKOMENDASI,
    )
except ImportError:
    from src.inference import (
        load_artifacts, preprocess_input, predict,
        bmi_info, ORDER_TARGET, CLASS_LABELS, REKOMENDASI,
    )

# ── Load model saat modul diimport ────────────────────────────────────────────
_model, _model_name, _scaler, _oe = load_artifacts()

ORD_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
MTRANS_OPTIONS = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]


def predict_single(
    gender_male: bool,
    age: float,
    height: float,
    weight: float,
    family_history: bool = False,
    favc: bool = False,
    fcvc: float = 2.0,
    ncp: float = 3.0,
    caec_level: int = 1,
    smoke: bool = False,
    ch2o: float = 2.0,
    scc: bool = False,
    faf: float = 1.0,
    tue: float = 1.0,
    calc_level: int = 0,
    mtrans: str = "Public_Transportation",
    model=None,
    scaler=None,
) -> dict:
    """
    Prediksi tingkat obesitas untuk satu individu.

    Returns:
        dict berisi: kelas, label_indonesia, probabilitas, BMI, kategori_BMI, rekomendasi
    """
    if model is None:
        model = _model
    if scaler is None:
        scaler = _scaler

    df_input = preprocess_input(
        gender_male=gender_male, age=age, height=height, weight=weight,
        family_history=family_history, favc=favc, fcvc=fcvc, ncp=ncp,
        caec_level=caec_level, smoke=smoke, ch2o=ch2o, scc=scc, faf=faf,
        tue=tue, calc_level=calc_level, mtrans=mtrans,
    )

    pred_class, probs = predict(df_input, model, scaler)
    bmi_val, bmi_cat = bmi_info(height, weight)

    return {
        "kelas":           pred_class,
        "label":           CLASS_LABELS[pred_class],
        "probabilitas":    {ORDER_TARGET[i]: round(float(probs[i]), 4) for i in range(7)},
        "prob_max":        round(float(probs.max()), 4),
        "BMI":             bmi_val,
        "kategori_BMI":    bmi_cat,
        "rekomendasi":     REKOMENDASI[pred_class],
        "model_digunakan": _model_name,
    }


def predict_batch(df_features: pd.DataFrame, model=None, scaler=None) -> pd.DataFrame:
    """
    Prediksi untuk banyak baris sekaligus dari DataFrame yang sudah terencoding.

    Parameters:
        df_features: DataFrame dengan kolom fitur yang sudah diproses
                     (output dari preprocess_input atau run_preprocessing)

    Returns:
        DataFrame dengan kolom tambahan: predicted_class, predicted_label, confidence
    """
    if model is None:
        model = _model
    if scaler is None:
        scaler = _scaler

    X_sc = pd.DataFrame(scaler.transform(df_features), columns=df_features.columns)
    preds = model.predict(X_sc)
    probs = model.predict_proba(X_sc) if hasattr(model, "predict_proba") else None

    result = df_features.copy()
    result["predicted_class"] = [ORDER_TARGET[p] for p in preds]
    result["predicted_label"] = [CLASS_LABELS[ORDER_TARGET[p]] for p in preds]
    if probs is not None:
        result["confidence"] = np.max(probs, axis=1).round(4)
    return result


def _cli():
    """Antarmuka command line untuk prediksi satu individu."""
    parser = argparse.ArgumentParser(
        description="Prediksi tingkat obesitas dari command line",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gender",  choices=["L", "P"], default="L",
                        help="Jenis kelamin: L=Laki-laki, P=Perempuan")
    parser.add_argument("--age",     type=float, default=22.0,  help="Usia (tahun)")
    parser.add_argument("--height",  type=float, default=1.70,  help="Tinggi badan (meter)")
    parser.add_argument("--weight",  type=float, default=70.0,  help="Berat badan (kg)")
    parser.add_argument("--family",  action="store_true",       help="Riwayat keluarga obesitas")
    parser.add_argument("--favc",    action="store_true",       help="Sering konsumsi makanan tinggi kalori")
    parser.add_argument("--fcvc",    type=float, default=2.0,   help="Frekuensi konsumsi sayur (1-3)")
    parser.add_argument("--ncp",     type=float, default=3.0,   help="Jumlah makan utama/hari (1-4)")
    parser.add_argument("--caec",    type=int,   default=1,     help="Camilan antar makan (0=no,1=Sometimes,2=Freq,3=Always)")
    parser.add_argument("--smoke",   action="store_true",       help="Merokok")
    parser.add_argument("--ch2o",    type=float, default=2.0,   help="Konsumsi air harian liter (1-3)")
    parser.add_argument("--scc",     action="store_true",       help="Pantau kalori harian")
    parser.add_argument("--faf",     type=float, default=1.0,   help="Frekuensi olahraga/minggu (0-3)")
    parser.add_argument("--tue",     type=float, default=1.0,   help="Waktu layar/hari (0-2)")
    parser.add_argument("--calc",    type=int,   default=0,     help="Frekuensi alkohol (0=no,1=Sometimes,2=Freq,3=Always)")
    parser.add_argument("--mtrans",  default="Public_Transportation",
                        choices=MTRANS_OPTIONS,                 help="Moda transportasi utama")
    args = parser.parse_args()

    result = predict_single(
        gender_male=(args.gender == "L"),
        age=args.age, height=args.height, weight=args.weight,
        family_history=args.family, favc=args.favc, fcvc=args.fcvc, ncp=args.ncp,
        caec_level=args.caec, smoke=args.smoke, ch2o=args.ch2o, scc=args.scc,
        faf=args.faf, tue=args.tue, calc_level=args.calc, mtrans=args.mtrans,
    )

    print("\n" + "=" * 55)
    print("HASIL PREDIKSI TINGKAT OBESITAS")
    print("=" * 55)
    print(f"Tingkat Obesitas : {result['label']}")
    print(f"Keyakinan Model  : {result['prob_max'] * 100:.1f}%")
    print(f"BMI              : {result['BMI']} ({result['kategori_BMI']})")
    print(f"Rekomendasi      : {result['rekomendasi']}")
    print(f"Model digunakan  : {result['model_digunakan']}")
    print("\nProbabilitas per kelas:")
    for kelas, prob in sorted(result["probabilitas"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {kelas:<25}: {prob:.4f} {bar}")
    print("=" * 55)
    print("DISCLAIMER: Output ini bersifat decision support, bukan diagnosis medis.")


if __name__ == "__main__":
    _cli()
