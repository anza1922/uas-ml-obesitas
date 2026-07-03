import requests
import json

# 1. URL API Anda (Pastikan uvicorn sedang jalan)
API_URL = "http://localhost:8000/predict"

# 2. Data inputan (misal dari chat user di Telegram)
data_input = {
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

# 3. Tembak API-nya
response = requests.post(API_URL, json=data_input)

# 4. Ambil dan cetak hasilnya
if response.status_code == 200:
    hasil = response.json()
    print("=== HASIL PREDIKSI ===")
    print(f"Status Model : {hasil['model_used']}")
    print(f"Prediksi     : {hasil['prediction']['label']} ({hasil['prediction']['confidence_percent']}%)")
    print(f"Nilai BMI    : {hasil['bmi_info']['bmi_value']} - {hasil['bmi_info']['bmi_category']}")
    print(f"Saran Medis  : {hasil['recommendation']}")
else:
    print("Error:", response.text)
