"""
api_fastapi.py
==============
REST API untuk model Prediksi Tingkat Obesitas menggunakan FastAPI.
UAS Pembelajaran Mesin | Universitas Dian Nuswantoro

Cara Menjalankan:
-----------------
1. Pastikan fastapi dan uvicorn sudah terinstall:
   pip install fastapi uvicorn pydantic

2. Jalankan server:
   uvicorn api_fastapi:app --reload

3. Buka dokumentasi interaktif (Swagger UI):
   http://localhost:8000/docs
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Tambahkan src/ ke sys.path agar bisa import modul inference
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

from inference import (
    load_artifacts, preprocess_input, predict, bmi_info,
    CLASS_LABELS, REKOMENDASI
)

# Inisialisasi FastAPI
app = FastAPI(
    title="ObesityAI API",
    description="REST API untuk memprediksi tingkat obesitas berdasarkan data fisik dan gaya hidup.",
    version="1.0.0"
)

# Konfigurasi CORS agar bisa ditembak dari Web Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan request dari semua domain (HTML/JS)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Muat model saat startup
model, model_name, scaler, oe = load_artifacts()
if model is None:
    raise RuntimeError("Model tidak ditemukan! Pastikan file model ada di folder 'models/'.")

# ── Skema Input (Pydantic Model) ──────────────────────────────────────────────
class PredictionInput(BaseModel):
    # Demografi & Fisik
    gender: str = Field(..., description="'Laki-laki' atau 'Perempuan'")
    age: float = Field(..., ge=14.0, le=65.0, description="Usia (14 - 65 tahun)")
    height: float = Field(..., ge=1.40, le=2.10, description="Tinggi badan dalam meter (contoh: 1.70)")
    weight: float = Field(..., ge=30.0, le=300.0, description="Berat badan dalam kg")
    family_history_with_overweight: str = Field(..., description="'Ya' atau 'Tidak'")
    
    # Kebiasaan Makan
    favc: str = Field(..., description="Sering konsumsi kalori tinggi? 'Ya' atau 'Tidak'")
    fcvc: float = Field(..., ge=1.0, le=3.0, description="Frekuensi sayuran (1.0 - 3.0)")
    ncp: float = Field(..., ge=1.0, le=4.0, description="Jumlah makan utama (1.0 - 4.0)")
    caec: str = Field(..., description="Ngemil: 'Tidak pernah', 'Kadang-kadang', 'Sering', 'Selalu'")
    calc: str = Field(..., description="Alkohol: 'Tidak pernah', 'Kadang-kadang', 'Sering', 'Selalu'")
    ch2o: float = Field(..., ge=1.0, le=3.0, description="Konsumsi air (1.0 - 3.0)")
    
    # Gaya Hidup
    smoke: str = Field(..., description="Merokok? 'Ya' atau 'Tidak'")
    scc: str = Field(..., description="Pantau kalori? 'Ya' atau 'Tidak'")
    faf: float = Field(..., ge=0.0, le=3.0, description="Frekuensi olahraga (0.0 - 3.0)")
    tue: float = Field(..., ge=0.0, le=2.0, description="Waktu layar gadget (0.0 - 2.0)")
    mtrans: str = Field(..., description="Transportasi: 'Transportasi Umum', 'Mobil', 'Motor', 'Sepeda', 'Jalan Kaki'")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Laki-laki",
                "age": 22.0,
                "height": 1.70,
                "weight": 70.0,
                "family_history_with_overweight": "Tidak",
                "favc": "Ya",
                "fcvc": 2.0,
                "ncp": 3.0,
                "caec": "Kadang-kadang",
                "calc": "Tidak pernah",
                "ch2o": 2.0,
                "smoke": "Tidak",
                "scc": "Tidak",
                "faf": 1.0,
                "tue": 1.0,
                "mtrans": "Motor"
            }
        }

# Mapping untuk Ordinal Encoding & Transportasi
ORD_MAP = {"Tidak pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
MTRANS_MAP = {
    "Transportasi Umum": "Public_Transportation",
    "Mobil":             "Automobile",
    "Jalan Kaki":        "Walking",
    "Motor":             "Motorbike",
    "Sepeda":            "Bike",
}

# ── Endpoint API ──────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {
        "message": "Selamat datang di ObesityAI API",
        "model_aktif": model_name,
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_obesity(data: PredictionInput):
    try:
        # 1. Validasi pemetaan nilai teks
        if data.caec not in ORD_MAP:
            raise ValueError(f"Nilai caec tidak valid: {data.caec}")
        if data.calc not in ORD_MAP:
            raise ValueError(f"Nilai calc tidak valid: {data.calc}")
        if data.mtrans not in MTRANS_MAP:
            raise ValueError(f"Nilai mtrans tidak valid: {data.mtrans}")
            
        # 2. Preprocessing input
        df_in = preprocess_input(
            gender_male=(data.gender.lower() == "laki-laki"),
            age=data.age,
            height=data.height,
            weight=data.weight,
            family_history=(data.family_history_with_overweight.lower() == "ya"),
            favc=(data.favc.lower() == "ya"),
            fcvc=data.fcvc,
            ncp=data.ncp,
            caec_level=ORD_MAP[data.caec],
            smoke=(data.smoke.lower() == "ya"),
            ch2o=data.ch2o,
            scc=(data.scc.lower() == "ya"),
            faf=data.faf,
            tue=data.tue,
            calc_level=ORD_MAP[data.calc],
            mtrans=MTRANS_MAP[data.mtrans]
        )
        
        # 3. Prediksi menggunakan model yang dimuat
        # (logika BMI override & clipping sudah ditangani di dalam predict() function)
        pred_class, probs = predict(df_in, model, scaler)
        
        # 4. Hitung informasi BMI
        bmi_val, bmi_cat = bmi_info(data.height, data.weight)
        
        # 5. Susun respons (probabilitas per kelas dalam bentuk dictionary)
        from inference import ORDER_TARGET
        prob_dict = {
            CLASS_LABELS.get(ORDER_TARGET[i], ORDER_TARGET[i]): round(float(probs[i]) * 100, 2)
            for i in range(len(ORDER_TARGET))
        }
        
        return {
            "status": "success",
            "model_used": model_name,
            "prediction": {
                "class_id": pred_class,
                "label": CLASS_LABELS.get(pred_class, pred_class),
                "confidence_percent": round(float(probs.max()) * 100, 2),
            },
            "bmi_info": {
                "bmi_value": bmi_val,
                "bmi_category": bmi_cat
            },
            "probabilities": prob_dict,
            "recommendation": REKOMENDASI.get(pred_class, "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
