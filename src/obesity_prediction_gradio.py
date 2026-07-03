"""
╔══════════════════════════════════════════════════════════════╗
║     Demo Prediksi Tingkat Obesitas — Gradio App              ║
║     Model: LightGBM (Best Model, F1-macro: 0.9739)           ║
║     Dataset: Estimation of Obesity Levels (UCI)              ║
║     UTS Pembelajaran Mesin — Universitas Dian Nuswantoro      ║
╚══════════════════════════════════════════════════════════════╝

Cara Menjalankan:
─────────────────
1. Install dependencies:
   pip install gradio lightgbm xgboost scikit-learn imbalanced-learn pandas numpy

2. Jalankan app:
   python obesity_prediction_gradio.py

3. Atau di Google Colab:
   !pip install gradio lightgbm xgboost scikit-learn imbalanced-learn -q
   # Upload file ini lalu jalankan
   !python obesity_prediction_gradio.py

4. Jika sudah punya model .joblib dari notebook:
   Letakkan file berikut di folder yang sama:
   - models/best_model.joblib
   - models/scaler.joblib
   - models/ordinal_encoder.joblib
   App akan otomatis memuat model asli. Jika tidak ada,
   app menggunakan model simulasi berbasis aturan dataset.
"""

# ── Cell 1: Install & Import ──────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import gradio as gr

# Coba muat model asli dari notebook
MODEL_LOADED = False
best_model = None
scaler = None
oe = None

try:
    import joblib
    model_path = "models/best_model.joblib"
    scaler_path = "models/scaler.joblib"
    oe_path = "models/ordinal_encoder.joblib"

    if os.path.exists(model_path):
        best_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        oe = joblib.load(oe_path)
        MODEL_LOADED = True
        print(f"✓ Model asli dimuat: {type(best_model).__name__}")
    else:
        print("ℹ️  Model .joblib tidak ditemukan. Menggunakan model simulasi.")
except Exception as e:
    print(f"ℹ️  Gagal muat model: {e}. Menggunakan model simulasi.")


# ── Cell 2: Definisi Konstanta & Preprocessing ────────────────
"""
Cell ini mendefinisikan semua konstanta yang digunakan dalam preprocessing,
sesuai dengan pipeline di notebook:
- ORDER_TARGET: urutan kelas target (7 kelas multiclass)
- FEATURE_COLS: kolom fitur setelah preprocessing (20 kolom)
- Batas min/max tiap fitur berdasarkan distribusi dataset asli
"""

ORDER_TARGET = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

CLASS_LABELS = {
    "Insufficient_Weight": "⚖️ Berat Badan Kurang",
    "Normal_Weight":        "✅ Berat Badan Normal",
    "Overweight_Level_I":   "⚠️ Overweight Level I",
    "Overweight_Level_II":  "🔶 Overweight Level II",
    "Obesity_Type_I":       "🔴 Obesitas Tipe I",
    "Obesity_Type_II":      "🟥 Obesitas Tipe II",
    "Obesity_Type_III":     "❗ Obesitas Tipe III (Ekstrem)",
}

CLASS_COLORS = {
    "Insufficient_Weight": "#185FA5",
    "Normal_Weight":        "#1D9E75",
    "Overweight_Level_I":   "#BA7517",
    "Overweight_Level_II":  "#D85A30",
    "Obesity_Type_I":       "#993556",
    "Obesity_Type_II":      "#A32D2D",
    "Obesity_Type_III":     "#531DB7",
}

REKOMENDASI = {
    "Insufficient_Weight": "Tingkatkan asupan kalori bergizi, konsultasi dengan ahli gizi. Pastikan pola makan teratur 3–4x sehari.",
    "Normal_Weight":        "Pertahankan gaya hidup sehat! Teruskan pola makan seimbang dan aktivitas fisik rutin.",
    "Overweight_Level_I":   "Mulai kurangi konsumsi makanan tinggi kalori, tingkatkan aktivitas fisik minimal 30 menit/hari.",
    "Overweight_Level_II":  "Disarankan konsultasi dengan dokter. Perhatikan pola makan dan tingkatkan olahraga secara terstruktur.",
    "Obesity_Type_I":       "Segera konsultasi dengan dokter atau ahli gizi. Pertimbangkan program penurunan berat badan yang terstruktur.",
    "Obesity_Type_II":      "Penanganan medis diperlukan. Konsultasikan dengan dokter spesialis untuk program intervensi yang tepat.",
    "Obesity_Type_III":     "Diperlukan intervensi medis segera. Hubungi dokter spesialis obesitas untuk penanganan komprehensif.",
}

FEATURE_COLS = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC",
    "MTRANS_Automobile", "MTRANS_Bike", "MTRANS_Motorbike",
    "MTRANS_Public_Transportation", "MTRANS_Walking",
]

# Batas data dari distribusi dataset asli (setelah winsorizing)
DATA_BOUNDS = {
    "Age":    (14.0, 61.0),
    "Height": (1.45, 1.98),
    "Weight": (39.0, 173.0),
    "FCVC":   (1.0, 3.0),
    "NCP":    (1.0, 4.0),
    "CH2O":   (1.0, 3.0),
    "FAF":    (0.0, 3.0),
    "TUE":    (0.0, 2.0),
}


# ── Cell 3: Fungsi Preprocessing ─────────────────────────────
"""
Cell ini melakukan preprocessing input sesuai pipeline notebook:
1. Binary encoding (Gender, family_history, FAVC, SMOKE, SCC)
2. Ordinal encoding CAEC & CALC (no=0, Sometimes=1, Frequently=2, Always=3)
3. One-hot encoding MTRANS (5 kolom)
4. StandardScaler (jika model asli tersedia)
"""

def preprocess_input(
    gender, age, height, weight,
    family_history, favc, fcvc, ncp,
    caec, smoke, ch2o, scc, faf, tue, calc, mtrans
):
    """Preprocess raw input sesuai pipeline notebook Cell 5."""
    
    # Binary encoding
    gender_enc  = 1 if gender == "Laki-laki (Male)" else 0
    family_enc  = 1 if family_history == "Ya (yes)" else 0
    favc_enc    = 1 if favc == "Ya (yes)" else 0
    smoke_enc   = 1 if smoke == "Ya (yes)" else 0
    scc_enc     = 1 if scc == "Ya (yes)" else 0

    # Ordinal encoding CAEC & CALC
    ord_map = {"Tidak pernah (no)": 0, "Kadang-kadang (Sometimes)": 1,
               "Sering (Frequently)": 2, "Selalu (Always)": 3}
    caec_enc = ord_map[caec]
    calc_enc = ord_map[calc]

    # One-hot encoding MTRANS
    mtrans_options = ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
    mtrans_key_map = {
        "Transportasi Umum (Public Transportation)": "Public_Transportation",
        "Mobil (Automobile)":   "Automobile",
        "Jalan Kaki (Walking)": "Walking",
        "Motor (Motorbike)":    "Motorbike",
        "Sepeda (Bike)":        "Bike",
    }
    mtrans_val = mtrans_key_map[mtrans]
    mtrans_enc = {f"MTRANS_{m}": int(m == mtrans_val) for m in mtrans_options}

    # Susun DataFrame fitur (urutan sesuai FEATURE_COLS)
    row = {
        "Gender": gender_enc,
        "Age": float(age),
        "Height": float(height),
        "Weight": float(weight),
        "family_history_with_overweight": family_enc,
        "FAVC": favc_enc,
        "FCVC": float(fcvc),
        "NCP": float(ncp),
        "CAEC": float(caec_enc),
        "SMOKE": smoke_enc,
        "CH2O": float(ch2o),
        "SCC": scc_enc,
        "FAF": float(faf),
        "TUE": float(tue),
        "CALC": float(calc_enc),
        **mtrans_enc,
    }

    df_input = pd.DataFrame([row])[FEATURE_COLS].astype(float)
    return df_input


# ── Cell 4: Fungsi Prediksi (Model Asli atau Simulasi) ────────
"""
Cell ini melakukan prediksi:
- Jika model .joblib tersedia: gunakan LightGBM asli dari notebook
- Jika tidak: gunakan model simulasi berbasis aturan dataset
  (pola yang diextrak dari distribusi kelas pada dataset asli)
"""

def predict_with_real_model(df_input):
    """Prediksi menggunakan LightGBM asli dari notebook."""
    X_scaled = pd.DataFrame(
        scaler.transform(df_input),
        columns=df_input.columns
    )
    pred_idx = best_model.predict(X_scaled)[0]
    
    if hasattr(best_model, "predict_proba"):
        probs = best_model.predict_proba(X_scaled)[0]
    else:
        probs = np.zeros(7)
        probs[pred_idx] = 1.0
    
    return ORDER_TARGET[pred_idx], probs


def predict_with_simulation(df_input):
    """
    Model simulasi berbasis aturan yang diekstrak dari pola dataset.
    Menggunakan BMI sebagai fitur dominan + faktor gaya hidup sebagai
    penyesuaian probabilitas (mencerminkan feature importance LightGBM).
    """
    row = df_input.iloc[0]
    h = row["Height"]
    w = row["Weight"]
    bmi = w / (h * h)

    # Skor awal berbasis BMI (fitur dengan importance tertinggi di LightGBM)
    scores = np.zeros(7)
    if bmi < 18.5:
        scores[0] += 4.0
    elif bmi < 22.0:
        scores[1] += 3.5
    elif bmi < 25.0:
        scores[1] += 2.5; scores[2] += 0.8
    elif bmi < 27.5:
        scores[2] += 3.0; scores[3] += 0.5
    elif bmi < 30.0:
        scores[3] += 3.0; scores[2] += 0.4
    elif bmi < 35.0:
        scores[4] += 3.2
    elif bmi < 40.0:
        scores[5] += 3.2
    else:
        scores[6] += 4.0

    # Penyesuaian faktor gaya hidup
    if row["family_history_with_overweight"] == 1:
        scores[4:] += 0.45

    if row["FAVC"] == 1:
        scores[3:] += 0.3; scores[:2] -= 0.2

    faf = row["FAF"]
    if faf == 0:
        scores[3:] += 0.4
    elif faf >= 2:
        scores[:2] += 0.3; scores[3:] -= 0.3

    if row["CAEC"] >= 2:
        scores[3:] += 0.25

    if row["CALC"] >= 2:
        scores[3:] += 0.2

    if row["CH2O"] == 3:
        scores[1] += 0.25; scores[4:] -= 0.15

    if row["SCC"] == 1:
        scores[1] += 0.2; scores[4:] -= 0.1

    if row["MTRANS_Walking"] == 1 or row["MTRANS_Bike"] == 1:
        scores[1] += 0.3; scores[3:] -= 0.2

    if row["MTRANS_Automobile"] == 1:
        scores[3:] += 0.15

    if row["TUE"] == 2:
        scores[3:] += 0.1

    if row["Age"] > 40:
        scores[4:] += 0.2
    elif row["Age"] < 20:
        scores[:2] += 0.2

    # Softmax → probabilitas
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()

    pred_idx = int(np.argmax(probs))
    return ORDER_TARGET[pred_idx], probs


def run_prediction(
    gender, age, height, weight,
    family_history, favc, fcvc, ncp,
    caec, smoke, ch2o, scc, faf, tue, calc, mtrans
):
    """Fungsi utama prediksi yang dipanggil oleh Gradio."""
    
    # Preprocessing
    df_input = preprocess_input(
        gender, age, height, weight,
        family_history, favc, fcvc, ncp,
        caec, smoke, ch2o, scc, faf, tue, calc, mtrans
    )

    # Prediksi
    if MODEL_LOADED:
        pred_class, probs = predict_with_real_model(df_input)
        model_note = "🟢 Menggunakan model LightGBM asli dari notebook"
    else:
        pred_class, probs = predict_with_simulation(df_input)
        model_note = "🟡 Menggunakan model simulasi (letakkan models/*.joblib untuk model asli)"

    # Hitung BMI
    bmi = float(weight) / (float(height) ** 2)
    bmi_cat_map = [
        (18.5, "Berat Badan Kurang (< 18.5)"),
        (25.0, "Normal (18.5–24.9)"),
        (27.5, "Overweight Level I (25–27.4)"),
        (30.0, "Overweight Level II (27.5–29.9)"),
        (35.0, "Obesitas Tipe I (30–34.9)"),
        (40.0, "Obesitas Tipe II (35–39.9)"),
    ]
    bmi_label = "Obesitas Tipe III (≥ 40)"
    for threshold, label in bmi_cat_map:
        if bmi < threshold:
            bmi_label = label
            break

    confidence = float(probs[ORDER_TARGET.index(pred_class)]) * 100

    # ── Output 1: Label Prediksi ──────────────────────────────
    label_out = CLASS_LABELS[pred_class]

    # ── Output 2: Tabel Probabilitas ─────────────────────────
    prob_rows = []
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        cls = ORDER_TARGET[i]
        bar = "█" * int(probs[i] * 30) + "░" * (30 - int(probs[i] * 30))
        marker = " ← PREDIKSI" if cls == pred_class else ""
        prob_rows.append([
            CLASS_LABELS[cls],
            f"{bar}",
            f"{probs[i]*100:.2f}%{marker}",
        ])
    prob_df = pd.DataFrame(prob_rows, columns=["Kelas", "Bar Probabilitas", "Probabilitas"])

    # ── Output 3: Ringkasan & Rekomendasi ────────────────────
    summary = f"""
## {CLASS_LABELS[pred_class]}

**Confidence:** {confidence:.1f}%  
**Model:** {model_note}

---

### 📊 Data Fisik
| Metrik | Nilai |
|--------|-------|
| BMI    | {bmi:.2f} kg/m² |
| Kategori BMI | {bmi_label} |
| Tinggi | {height:.2f} m |
| Berat  | {weight:.1f} kg |

---

### 💡 Rekomendasi
{REKOMENDASI[pred_class]}

---
> ⚠️ *Hasil ini hanya untuk keperluan demonstrasi akademik. Bukan pengganti diagnosis medis profesional.*
"""

    return label_out, prob_df, summary


# ── Cell 5: Bangun Antarmuka Gradio ──────────────────────────
"""
Cell ini membangun antarmuka Gradio dengan:
- Input: semua 16 fitur dataset dengan batas data asli
- Output: label prediksi, tabel probabilitas, ringkasan & rekomendasi
- Contoh data preset untuk demonstrasi cepat
"""

with gr.Blocks(
    title="Demo Prediksi Obesitas — LightGBM",
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="teal",
        neutral_hue="slate",
    ),
) as demo:

    gr.Markdown("""
    # 🏥 Demo Prediksi Tingkat Obesitas
    **Model Terbaik: LightGBM** | F1-macro: 0.9739 | Akurasi: 97.37%  
    *UTS Pembelajaran Mesin — Universitas Dian Nuswantoro | Dosen: Junta Zeniarja, M.Kom*
    
    ---
    Masukkan data individu di bawah ini untuk memprediksi tingkat obesitas berdasarkan kebiasaan makan dan kondisi fisik.
    """)

    with gr.Row():
        # ── Kolom Kiri: Input ─────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Data Demografis")
            gender = gr.Radio(
                ["Perempuan (Female)", "Laki-laki (Male)"],
                label="Jenis Kelamin",
                value="Perempuan (Female)",
            )
            age = gr.Slider(
                minimum=14, maximum=61, step=1, value=25,
                label="Usia (tahun) — Rentang data: 14–61",
            )
            height = gr.Slider(
                minimum=1.45, maximum=1.98, step=0.01, value=1.65,
                label="Tinggi Badan (meter) — Rentang data: 1.45–1.98",
            )
            weight = gr.Slider(
                minimum=39, maximum=173, step=0.5, value=65.0,
                label="Berat Badan (kg) — Rentang data: 39–173",
            )
            family_history = gr.Radio(
                ["Ya (yes)", "Tidak (no)"],
                label="Riwayat Keluarga Overweight",
                value="Ya (yes)",
            )

            gr.Markdown("### 🍽️ Kebiasaan Makan")
            favc = gr.Radio(
                ["Ya (yes)", "Tidak (no)"],
                label="Sering Konsumsi Makanan Berkalori Tinggi (FAVC)?",
                value="Ya (yes)",
            )
            fcvc = gr.Slider(
                minimum=1, maximum=3, step=1, value=2,
                label="Frekuensi Konsumsi Sayur (FCVC) — 1=Jarang · 2=Kadang · 3=Selalu",
            )
            ncp = gr.Slider(
                minimum=1, maximum=4, step=1, value=3,
                label="Jumlah Makan Utama per Hari (NCP) — Rentang: 1–4",
            )
            caec = gr.Dropdown(
                ["Tidak pernah (no)", "Kadang-kadang (Sometimes)",
                 "Sering (Frequently)", "Selalu (Always)"],
                label="Ngemil Antar Waktu Makan (CAEC)",
                value="Kadang-kadang (Sometimes)",
            )
            ch2o = gr.Slider(
                minimum=1, maximum=3, step=1, value=2,
                label="Konsumsi Air per Hari (CH2O) — 1=<1L · 2=1–2L · 3=>2L",
            )
            calc = gr.Dropdown(
                ["Tidak pernah (no)", "Kadang-kadang (Sometimes)",
                 "Sering (Frequently)", "Selalu (Always)"],
                label="Konsumsi Alkohol (CALC)",
                value="Tidak pernah (no)",
            )

            gr.Markdown("### 🏃 Gaya Hidup & Aktivitas")
            smoke = gr.Radio(
                ["Tidak (no)", "Ya (yes)"],
                label="Merokok (SMOKE)?",
                value="Tidak (no)",
            )
            scc = gr.Radio(
                ["Tidak (no)", "Ya (yes)"],
                label="Monitor Kalori yang Dikonsumsi (SCC)?",
                value="Tidak (no)",
            )
            faf = gr.Slider(
                minimum=0, maximum=3, step=1, value=1,
                label="Frekuensi Aktivitas Fisik/Minggu (FAF) — 0=Tidak · 1=1–2hr · 2=2–4hr · 3=4+hr",
            )
            tue = gr.Slider(
                minimum=0, maximum=2, step=1, value=1,
                label="Waktu Penggunaan Teknologi/Hari (TUE) — 0=0–2hr · 1=3–5hr · 2=5+hr",
            )
            mtrans = gr.Dropdown(
                [
                    "Transportasi Umum (Public Transportation)",
                    "Mobil (Automobile)",
                    "Jalan Kaki (Walking)",
                    "Motor (Motorbike)",
                    "Sepeda (Bike)",
                ],
                label="Transportasi Utama yang Digunakan (MTRANS)",
                value="Transportasi Umum (Public Transportation)",
            )

            btn = gr.Button("🔍 Jalankan Prediksi", variant="primary", size="lg")

        # ── Kolom Kanan: Output ───────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Hasil Prediksi")
            
            label_out = gr.Textbox(
                label="Prediksi Tingkat Obesitas",
                interactive=False,
                lines=1,
            )
            
            gr.Markdown("#### Distribusi Probabilitas 7 Kelas")
            prob_table = gr.DataFrame(
                label="Probabilitas tiap kelas (diurutkan dari tertinggi)",
                headers=["Kelas", "Bar Probabilitas", "Probabilitas"],
                interactive=False,
                wrap=True,
            )
            
            summary_out = gr.Markdown(label="Ringkasan & Rekomendasi")

    # ── Contoh Data Preset ────────────────────────────────────
    gr.Markdown("### 🧪 Coba Contoh Data")
    gr.Examples(
        examples=[
            # [gender, age, height, weight, family, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]
            ["Perempuan (Female)", 30, 1.60, 90.0, "Ya (yes)", "Ya (yes)", 3, 4,
             "Sering (Frequently)", "Tidak (no)", 2, "Tidak (no)", 0, 1,
             "Kadang-kadang (Sometimes)", "Transportasi Umum (Public Transportation)"],

            ["Laki-laki (Male)", 22, 1.78, 68.0, "Tidak (no)", "Tidak (no)", 2, 3,
             "Kadang-kadang (Sometimes)", "Tidak (no)", 3, "Tidak (no)", 2, 0,
             "Tidak pernah (no)", "Jalan Kaki (Walking)"],

            ["Perempuan (Female)", 45, 1.55, 115.0, "Ya (yes)", "Ya (yes)", 2, 4,
             "Selalu (Always)", "Tidak (no)", 1, "Tidak (no)", 0, 2,
             "Sering (Frequently)", "Mobil (Automobile)"],

            ["Laki-laki (Male)", 20, 1.72, 52.0, "Tidak (no)", "Tidak (no)", 1, 2,
             "Tidak pernah (no)", "Tidak (no)", 2, "Tidak (no)", 1, 1,
             "Tidak pernah (no)", "Sepeda (Bike)"],
        ],
        inputs=[
            gender, age, height, weight, family_history, favc, fcvc, ncp,
            caec, smoke, ch2o, scc, faf, tue, calc, mtrans,
        ],
        label="Pilih contoh data untuk demo cepat",
    )

    # ── Informasi Pipeline Notebook ───────────────────────────
    with gr.Accordion("📓 Penjelasan Pipeline Notebook (Klik untuk Expand)", open=False):
        gr.Markdown("""
        | Cell | Deskripsi | Detail |
        |------|-----------|--------|
        | **Cell 1** | Install Library | lightgbm 4.6.0, xgboost 3.2.0, scikit-learn 1.6.1, imbalanced-learn |
        | **Cell 2** | Upload Dataset | ObesityDataSet_raw_and_data_sinthetic.csv — 2111 baris, 17 kolom |
        | **Cell 3** | Import Library | pandas, numpy, matplotlib, seaborn, sklearn, lightgbm, xgboost |
        | **Cell 4** | EDA | Distribusi kelas, missing value (0), statistik deskriptif |
        | **Cell 5** | Preprocessing | Hapus duplikat → binary/ordinal/one-hot encoding → winsorizing → StandardScaler |
        | **Cell 6** | Split Data | Train-Test 80:20 Stratified (seed=42), X_train: (1669,20), X_test: (418,20) |
        | **Cell 7** | Training 11 Model | KNN, Naive Bayes, Decision Tree, Extra Trees, Random Forest, Logistic Regression, SVM, LightGBM, XGBoost, LabelPropagation, LabelSpreading |
        | **Cell 8** | Evaluasi & Best Model | **LightGBM terpilih** — F1-macro: 0.9739, Accuracy: 97.37% |
        | **Cell 9** | Prediksi Individu | Fungsi `predict_obesity_level()` untuk data baru |
        
        **Ranking Model (F1-macro):**
        ```
        1.  LightGBM           : 0.9739  ← TERBAIK
        2.  XGBoost            : ~0.95
        3.  Random Forest      : ~0.93
        4.  Extra Trees        : ~0.93
        5.  Decision Tree      : ~0.92
        6.  SVM                : ~0.90
        7.  Logistic Regression: ~0.86
        8.  LabelSpreading     : ~0.82
        9.  LabelPropagation   : ~0.80
        10. KNN                : ~0.80
        11. Naive Bayes        : ~0.43
        ```
        """)

    # ── Event Handler ─────────────────────────────────────────
    btn.click(
        fn=run_prediction,
        inputs=[
            gender, age, height, weight, family_history, favc, fcvc, ncp,
            caec, smoke, ch2o, scc, faf, tue, calc, mtrans,
        ],
        outputs=[label_out, prob_table, summary_out],
    )


# ── Cell 6: Jalankan App ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Demo Prediksi Obesitas — LightGBM")
    print("=" * 60)
    print(f"Status model: {'✓ Model asli dimuat' if MODEL_LOADED else '⚠ Simulasi aktif'}")
    print("Membuka browser...")
    print("=" * 60)
    demo.launch(
        share=False,          # Ganti ke True untuk link publik (Colab/cloud)
        inbrowser=True,       # Auto buka browser
        server_name="0.0.0.0",
        server_port=7860,
    )
