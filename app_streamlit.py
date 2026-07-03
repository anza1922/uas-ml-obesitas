"""
app_streamlit.py  -  Aplikasi Prediksi Tingkat Obesitas
UAS Pembelajaran Mesin | Universitas Dian Nuswantoro

Lokal  : streamlit run app_streamlit.py
Deploy : push ke GitHub -> connect di share.streamlit.io
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SRC_DIR     = os.path.join(BASE_DIR, "src")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
DATA_PATH   = os.path.join(BASE_DIR, "data", "ObesityDataSet_raw_and_data_sinthetic.csv")
sys.path.insert(0, SRC_DIR)

from inference import (
    load_artifacts, preprocess_input, predict, bmi_info,
    ORDER_TARGET, CLASS_LABELS, REKOMENDASI,
)

st.set_page_config(
    page_title="ObesityAI — Prediksi Obesitas",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: aman, tidak override body/html ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

.main .block-container { padding: 1rem 2rem 3rem; max-width: 1150px; }
#MainMenu, footer, header { visibility: hidden; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0F766E 0%, #0D9488 50%, #14B8A6 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(13,148,136,0.25);
    color: #fff;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: -0.02em;
}
.hero p { margin: 0; font-size: 0.9rem; opacity: 0.88; }
.hero-badges { margin-top: 0.8rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }
.h-badge {
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #fff;
}

/* Section title */
.stitle {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #0F766E;
    border-bottom: 2.5px solid #0D9488;
    padding-bottom: 0.4rem;
    margin: 1.4rem 0 0.9rem;
}

/* Info box */
.infobox {
    background: #F0FDF9;
    border: 1px solid #CCFBF1;
    border-left: 4px solid #0D9488;
    border-radius: 0 10px 10px 0;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #0F766E;
    line-height: 1.6;
    margin-bottom: 1rem;
}

/* Card */
.card {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

/* Metric tile */
.mtile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1rem 0.5rem;
    text-align: center;
}
.mtile .mval {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #0F766E;
}
.mtile .mlbl { font-size: 0.73rem; color: #64748B; margin-top: 0.15rem; font-weight: 500; }
.mtile .mico { font-size: 1.2rem; margin-bottom: 0.2rem; }

/* Result badge */
.rbadge {
    display: inline-block;
    padding: 0.28rem 0.85rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.b-blue   { background: #DBEAFE; color: #1D4ED8; }
.b-green  { background: #DCFCE7; color: #15803D; }
.b-yellow { background: #FEF9C3; color: #92400E; }
.b-orange { background: #FFEDD5; color: #C2410C; }
.b-red    { background: #FEE2E2; color: #B91C1C; }
.b-darkred{ background: #FEE2E2; color: #7F1D1D; }
.b-purple { background: #EDE9FE; color: #6D28D9; }

/* Disclaimer */
.warn {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #F59E0B;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    font-size: 0.8rem;
    color: #92400E;
    margin-top: 0.8rem;
    line-height: 1.6;
}

/* Progress bar */
.pbar-wrap {
    background: #E2E8F0;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.6rem;
    overflow: hidden;
}
.pbar-fill {
    height: 6px;
    border-radius: 4px;
}

/* Form group title */
.fgtitle {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #0F766E;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 0.45rem 0.8rem;
    background: #F0FDF9;
    border-left: 3px solid #0D9488;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.7rem;
}

/* Model card */
.mcard {
    background: #fff;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.55rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

/* Pipeline step */
.pstep {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.855rem;
}
.pstep:last-child { border-bottom: none; }
.pstep-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #0D9488;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* Tech tile */
.ttile {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.9rem 0.5rem;
    text-align: center;
}
.ttile:hover { border-color: #0D9488; background: #F0FDF9; }

/* Kelas tabel row */
.krow {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid #F1F5F9;
    font-size: 0.82rem;
}
.krow:last-child { border-bottom: none; }

/* Button override */
.stButton > button {
    background: linear-gradient(135deg, #0D9488, #0F766E) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.7rem 0 !important;
    box-shadow: 0 6px 20px rgba(13,148,136,0.35) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #14B8A6, #0D9488) !important;
    box-shadow: 0 8px 28px rgba(13,148,136,0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanta ──────────────────────────────────────────────────────────────────
BADGE_CLASS = {
    "Insufficient_Weight": ("b-blue",   "⚖️"),
    "Normal_Weight":        ("b-green",  "✅"),
    "Overweight_Level_I":   ("b-yellow", "⚠️"),
    "Overweight_Level_II":  ("b-orange", "🔶"),
    "Obesity_Type_I":       ("b-red",    "🔴"),
    "Obesity_Type_II":      ("b-darkred","🟥"),
    "Obesity_Type_III":     ("b-purple", "❗"),
}
CHART_COLORS = {
    "Insufficient_Weight": "#3B82F6",
    "Normal_Weight":        "#10B981",
    "Overweight_Level_I":   "#F59E0B",
    "Overweight_Level_II":  "#F97316",
    "Obesity_Type_I":       "#EF4444",
    "Obesity_Type_II":      "#DC2626",
    "Obesity_Type_III":     "#7C3AED",
}
ORD_MAP = {"Tidak pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
MTRANS_MAP = {
    "Transportasi Umum": "Public_Transportation",
    "Mobil":             "Automobile",
    "Jalan Kaki":        "Walking",
    "Motor":             "Motorbike",
    "Sepeda":            "Bike",
}
TEAL  = "#0D9488"
SLATE = "#94A3B8"

LABEL_MAP_ID = {
    "Insufficient_Weight": "Berat Badan Kurang",
    "Normal_Weight":        "Berat Badan Normal",
    "Overweight_Level_I":   "Overweight Tingkat I",
    "Overweight_Level_II":  "Overweight Tingkat II",
    "Obesity_Type_I":       "Obesitas Tipe I",
    "Obesity_Type_II":      "Obesitas Tipe II",
    "Obesity_Type_III":     "Obesitas Tipe III",
}

@st.cache_resource(show_spinner="⏳ Memuat model AI...")
def get_artifacts():
    return load_artifacts()

model, model_name, scaler, oe = get_artifacts()

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="display:flex; align-items:center; gap:1.2rem;">
    <div style="font-size:3rem; line-height:1;">🩺</div>
    <div>
      <h1>ObesityAI — Prediksi Tingkat Obesitas</h1>
      <p>UAS Pembelajaran Mesin &middot; Universitas Dian Nuswantoro &middot; A11.2024.15791</p>
      <div class="hero-badges">
        <span class="h-badge">🤖 KNN</span>
        <span class="h-badge">📊 Naive Bayes</span>
        <span class="h-badge">⚡ SVM</span>
        <span class="h-badge">🏆 LightGBM</span>
        <span class="h-badge">11 Model Diuji</span>
        <span class="h-badge">F1-macro Terbaik: 97.39%</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ Model tidak ditemukan. Jalankan `python src/train_optimization.py` terlebih dahulu.")
    st.stop()

# ── TABS ───────────────────────────────────────────────────────────────────────
tab_pred, tab_dash, tab_about = st.tabs([
    "🔮  Tab 1 — Prediksi Obesitas",
    "📊  Tab 2 — Dashboard & Metrik",
    "ℹ️  Tab 3 — Tentang Model & Pipeline",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: PREDIKSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown('<div class="stitle">📝 Formulir Data Individu</div>', unsafe_allow_html=True)

    with st.form("form_prediksi", clear_on_submit=False):
        c1, c2, c3 = st.columns(3, gap="medium")

        # ── Kolom 1: Profil Fisik ──────────────────────────────────────────
        with c1:
            st.markdown('<div class="fgtitle">👤 Profil Fisik & Demografi</div>', unsafe_allow_html=True)
            gender = st.radio(
                "Jenis Kelamin",
                ["Laki-laki", "Perempuan"],
                horizontal=True,
                help="Pilih jenis kelamin biologis Anda."
            )
            age = st.number_input(
                "Usia (tahun)",
                min_value=14, max_value=65, value=22,
                help="Usia dalam tahun. Rentang data: 14–65 tahun."
            )
            height = st.number_input(
                "Tinggi Badan (meter)",
                min_value=1.40, max_value=2.10, value=1.70, step=0.01, format="%.2f",
                help="Contoh: 1.70 = 170 cm"
            )
            weight = st.number_input(
                "Berat Badan (kg)",
                min_value=30.0, max_value=180.0, value=70.0, step=0.5,
                help="Berat badan dalam kilogram."
            )
            family_history = st.radio(
                "Riwayat Keluarga Obesitas / Overweight",
                ["Ya", "Tidak"],
                horizontal=True,
                help="Apakah ada anggota keluarga yang kelebihan berat badan?"
            )

        # ── Kolom 2: Kebiasaan Makan ────────────────────────────────────────
        with c2:
            st.markdown('<div class="fgtitle">🍽️ Kebiasaan Makan & Asupan</div>', unsafe_allow_html=True)
            favc = st.radio(
                "Sering Konsumsi Makanan Tinggi Kalori?",
                ["Ya", "Tidak"],
                horizontal=True,
                help="Fast food, gorengan, atau makanan berlemak tinggi."
            )
            fcvc = st.selectbox(
                "Frekuensi Konsumsi Sayuran per Hari",
                ["1 — Jarang sekali", "2 — Kadang-kadang", "3 — Setiap hari"],
                index=1,
                help="Seberapa sering Anda makan sayur?"
            )
            ncp = st.selectbox(
                "Jumlah Makan Utama per Hari",
                ["1 — Satu kali", "2 — Dua kali", "3 — Tiga kali (normal)", "4 — Empat kali"],
                index=2,
                help="Berapa kali Anda makan besar dalam sehari?"
            )
            caec = st.selectbox(
                "Kebiasaan Ngemil Antar Waktu Makan",
                ["Tidak pernah", "Kadang-kadang", "Sering", "Selalu"],
                index=1,
                help="Seberapa sering mengonsumsi camilan di luar jam makan?"
            )
            calc = st.selectbox(
                "Frekuensi Konsumsi Minuman Beralkohol",
                ["Tidak pernah", "Kadang-kadang", "Sering", "Selalu"],
                index=0,
                help="Seberapa sering konsumsi alkohol?"
            )
            ch2o = st.selectbox(
                "Konsumsi Air Putih per Hari",
                ["1 — Kurang dari 1 liter", "2 — 1 sampai 2 liter", "3 — Lebih dari 2 liter"],
                index=1,
                help="Perkiraan konsumsi air putih per hari."
            )

        # ── Kolom 3: Gaya Hidup ────────────────────────────────────────────
        with c3:
            st.markdown('<div class="fgtitle">🏃 Gaya Hidup & Aktivitas</div>', unsafe_allow_html=True)
            smoke = st.radio(
                "Apakah Anda Merokok?",
                ["Tidak", "Ya"],
                horizontal=True
            )
            scc = st.radio(
                "Aktif Memantau Kalori Harian?",
                ["Tidak", "Ya"],
                horizontal=True,
                help="Apakah Anda secara rutin menghitung kalori makanan?"
            )
            faf = st.selectbox(
                "Frekuensi Olahraga per Minggu",
                [
                    "0 — Tidak pernah olahraga",
                    "1 — 1 hingga 2 kali seminggu",
                    "2 — 3 hingga 4 kali seminggu",
                    "3 — Setiap hari (intensif)",
                ],
                index=1,
                help="Berapa kali per minggu Anda berolahraga?"
            )
            tue = st.selectbox(
                "Lama Penggunaan Gadget / Layar per Hari",
                [
                    "0 — Kurang dari 2 jam",
                    "1 — 3 sampai 5 jam",
                    "2 — Lebih dari 5 jam",
                ],
                index=1,
                help="Total waktu penggunaan HP, laptop, atau TV."
            )
            mtrans = st.selectbox(
                "Transportasi Utama Sehari-hari",
                ["Transportasi Umum", "Mobil", "Jalan Kaki", "Motor", "Sepeda"],
                index=0,
                help="Moda transportasi yang paling sering digunakan."
            )

        st.write("")
        submitted = st.form_submit_button(
            "🔍  Prediksi Tingkat Obesitas Saya",
            use_container_width=True
        )

    # ── Parsing nilai selectbox ────────────────────────────────────────────────
    def parse_int_prefix(val):
        """Ambil angka di depan string seperti '2 — Kadang-kadang' → 2.0"""
        return float(val.split(" ")[0])

    # ── Hasil Prediksi ────────────────────────────────────────────────────────
    if submitted:
        df_in = preprocess_input(
            gender_male=(gender == "Laki-laki"),
            age=age, height=height, weight=weight,
            family_history=(family_history == "Ya"),
            favc=(favc == "Ya"),
            fcvc=parse_int_prefix(fcvc),
            ncp=parse_int_prefix(ncp),
            caec_level=ORD_MAP[caec],
            smoke=(smoke == "Ya"),
            ch2o=parse_int_prefix(ch2o),
            scc=(scc == "Ya"),
            faf=parse_int_prefix(faf),
            tue=parse_int_prefix(tue),
            calc_level=ORD_MAP[calc],
            mtrans=MTRANS_MAP[mtrans],
        )
        pred_class, probs = predict(df_in, model, scaler)
        bmi_val, bmi_cat  = bmi_info(height, weight)

        st.markdown('<div class="stitle">🎯 Hasil Analisis & Prediksi</div>', unsafe_allow_html=True)

        # ── Baris atas: Prediksi + BMI ─────────────────────────────────────────
        col_pred, col_bmi = st.columns([1, 1], gap="medium")

        with col_pred:
            badge_cls, icon = BADGE_CLASS.get(pred_class, ("b-yellow", "❓"))
            conf        = probs.max() * 100
            label_text  = CLASS_LABELS[pred_class]
            rek_text    = REKOMENDASI[pred_class]

            st.markdown(f"""
            <div class="card">
              <div class="card-title">Tingkat Obesitas Terdeteksi</div>
              <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem;
                          font-weight:800; color:#0F172A; margin:0.25rem 0 0.6rem;">
                {icon} {label_text}
              </div>
              <span class="rbadge {badge_cls}">Keyakinan Model: {conf:.1f}%</span>
              <div style="margin-top:1rem; font-size:0.8rem; color:#64748B; line-height:1.6;">
                <b style="color:#334155;">Data Individu:</b><br>
                👤 {gender}, {age} tahun &nbsp;|&nbsp;
                📏 {height:.2f} m / {weight:.1f} kg
              </div>
            </div>
            <div class="warn">
              <strong>⚠️ Peringatan:</strong> Hasil ini adalah <em>alat bantu keputusan</em>,
              bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan profesional.
            </div>
            """, unsafe_allow_html=True)

        with col_bmi:
            # ── Panel BMI Visual ────────────────────────────────────────────
            bmi_num = float(bmi_val)

            # Rentang BMI untuk skala visual (10–50)
            bmi_segments = [
                (18.5, "#3B82F6", "Kurang"),
                (25.0, "#10B981", "Normal"),
                (30.0, "#F59E0B", "Overweight"),
                (40.0, "#EF4444", "Obesitas"),
                (50.0, "#7C3AED", "Ektrem"),
            ]
            # Tentukan warna & label posisi pengguna
            bmi_color = "#7C3AED"
            bmi_zone  = "Obesitas Ekstrem"
            for threshold, color, zone in bmi_segments:
                if bmi_num < threshold:
                    bmi_color = color
                    bmi_zone  = zone
                    break

            # Posisi pointer pada skala 10–50 → persentase 0–100
            bmi_pointer_pct = min(max((bmi_num - 10) / (50 - 10) * 100, 2), 97)

            bmi_zona_desc = {
                "Kurang":         ("BMI di bawah 18.5", "Berat badan terlalu rendah. Asupan gizi perlu ditingkatkan."),
                "Normal":         ("BMI 18.5 – 24.9",   "Berat badan ideal. Pertahankan gaya hidup sehat Anda!"),
                "Overweight":     ("BMI 25.0 – 29.9",   "Mulai waspada. Perhatikan pola makan dan perbanyak olahraga."),
                "Obesitas":       ("BMI 30.0 – 39.9",   "Risiko kesehatan meningkat. Konsultasi dokter sangat disarankan."),
                "Ektrem":         ("BMI ≥ 40",          "Risiko sangat tinggi. Diperlukan intervensi medis segera."),
            }
            zona_range, zona_desc = bmi_zona_desc.get(bmi_zone, ("", ""))

            st.markdown(f"""
            <div class="card">
              <div class="card-title">📊 Indeks Massa Tubuh (BMI)</div>

              <div style="display:flex; align-items:baseline; gap:0.5rem; margin-bottom:0.3rem;">
                <span style="font-family:'Space Grotesk',sans-serif; font-size:2rem;
                             font-weight:800; color:{bmi_color};">{bmi_num:.1f}</span>
                <span style="font-size:0.85rem; color:#64748B;">kg/m²</span>
                <span class="rbadge" style="background:{bmi_color}22; color:{bmi_color};
                      border:1px solid {bmi_color}55; margin-left:0.3rem;">{bmi_cat}</span>
              </div>

              <!-- Skala BMI visual -->
              <div style="position:relative; margin:0.9rem 0 0.4rem;">
                <!-- Bar segmen warna -->
                <div style="display:flex; border-radius:6px; overflow:hidden; height:14px;">
                  <div style="flex:1.7; background:#3B82F6;" title="Kurang (<18.5)"></div>
                  <div style="flex:3.0; background:#10B981;" title="Normal (18.5-25)"></div>
                  <div style="flex:2.5; background:#F59E0B;" title="Overweight (25-30)"></div>
                  <div style="flex:5.0; background:#EF4444;" title="Obesitas (30-40)"></div>
                  <div style="flex:2.8; background:#7C3AED;" title="Ekstrem (>40)"></div>
                </div>
                <!-- Pointer posisi BMI pengguna -->
                <div style="position:absolute; top:-5px; left:{bmi_pointer_pct:.1f}%;
                            transform:translateX(-50%); font-size:1.1rem; line-height:1;">▼</div>
              </div>

              <!-- Label skala -->
              <div style="display:flex; justify-content:space-between; font-size:0.67rem;
                          color:#94A3B8; margin-top:0.25rem; margin-bottom:0.75rem;">
                <span>10</span><span>18.5</span><span>25</span><span>30</span><span>40</span><span>50</span>
              </div>

              <!-- Legenda -->
              <div style="display:flex; flex-wrap:wrap; gap:0.35rem; margin-bottom:0.8rem;">
                <span style="font-size:0.68rem; padding:0.15rem 0.5rem; border-radius:4px;
                      background:#DBEAFE; color:#1D4ED8;">🔵 Kurang &lt;18.5</span>
                <span style="font-size:0.68rem; padding:0.15rem 0.5rem; border-radius:4px;
                      background:#DCFCE7; color:#15803D;">🟢 Normal 18.5–25</span>
                <span style="font-size:0.68rem; padding:0.15rem 0.5rem; border-radius:4px;
                      background:#FEF9C3; color:#92400E;">🟡 Overweight 25–30</span>
                <span style="font-size:0.68rem; padding:0.15rem 0.5rem; border-radius:4px;
                      background:#FEE2E2; color:#B91C1C;">🔴 Obesitas 30–40</span>
                <span style="font-size:0.68rem; padding:0.15rem 0.5rem; border-radius:4px;
                      background:#EDE9FE; color:#6D28D9;">🟣 Ekstrem &gt;40</span>
              </div>

              <div style="background:#F8FAFC; border-radius:8px; border:1px solid #E2E8F0;
                          padding:0.7rem 0.85rem;">
                <div style="font-size:0.72rem; font-weight:700; color:#94A3B8;
                             text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem;">
                  Zona Anda — {zona_range}
                </div>
                <div style="font-size:0.83rem; color:#334155; line-height:1.55;">
                  {zona_desc}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Baris bawah: Grafik + Saran Detail ────────────────────────────────
        col_chart, col_saran = st.columns([1.5, 1], gap="medium")

        with col_chart:
            bar_colors = [
                CHART_COLORS.get(c, TEAL) if c == pred_class else "#CBD5E1"
                for c in ORDER_TARGET
            ]
            labels_id = [
                f"{BADGE_CLASS.get(c,('','❓'))[1]} {CLASS_LABELS[c]}"
                for c in ORDER_TARGET
            ]
            fig = go.Figure(go.Bar(
                x=probs * 100,
                y=labels_id,
                orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
                text=[f"{v:.1f}%" for v in probs * 100],
                textposition="outside",
                textfont=dict(size=11, color="#334155"),
            ))
            fig.update_layout(
                title=dict(
                    text="<b>Distribusi Probabilitas per Kelas</b>",
                    font=dict(size=13, family="Space Grotesk"),
                    x=0,
                ),
                height=320,
                margin=dict(l=0, r=70, t=40, b=10),
                xaxis=dict(range=[0, 120], showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                plot_bgcolor="#fff",
                paper_bgcolor="#fff",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Gauge confidence
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                number={"suffix": "%", "font": {"size": 26, "color": TEAL, "family": "Space Grotesk"}},
                title={"text": "Tingkat Keyakinan Model", "font": {"size": 12, "color": "#64748B"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
                    "bar": {"color": TEAL, "thickness": 0.22},
                    "bgcolor": "#F8FAFC",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  50], "color": "#FEE2E2"},
                        {"range": [50, 75], "color": "#FEF9C3"},
                        {"range": [75,100], "color": "#DCFCE7"},
                    ],
                    "threshold": {
                        "line": {"color": TEAL, "width": 3},
                        "thickness": 0.75, "value": conf,
                    },
                },
            ))
            fig_g.update_layout(
                height=190,
                margin=dict(l=20, r=20, t=25, b=10),
                paper_bgcolor="#fff",
            )
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

        with col_saran:
            # ── Saran Spesifik Berdasarkan Kondisi Input ──────────────────
            saran_list = []

            # Saran berdasarkan BMI / kelas prediksi
            saran_list.append(("🎯", "Rekomendasi Utama", rek_text))

            # Olahraga
            faf_num = parse_int_prefix(faf)
            if faf_num == 0:
                saran_list.append(("🏃", "Aktivitas Fisik",
                    "Anda tidak berolahraga. Mulai dengan jalan kaki 15–30 menit/hari, "
                    "lalu tingkatkan secara bertahap ke 3–5 kali per minggu."))
            elif faf_num == 1:
                saran_list.append(("🏃", "Aktivitas Fisik",
                    "Olahraga 1–2x/minggu sudah bagus! Tingkatkan ke 3–4x/minggu "
                    "untuk hasil optimal. Pilih olahraga yang Anda sukai."))
            else:
                saran_list.append(("🏃", "Aktivitas Fisik",
                    "Frekuensi olahraga Anda sudah baik. Pastikan juga waktu pemulihan "
                    "yang cukup dan variasikan jenis latihan (kardio + kekuatan)."))

            # Pola makan
            if favc == "Ya":
                saran_list.append(("🍔", "Makanan Tinggi Kalori",
                    "Kurangi konsumsi makanan berkalori tinggi (fast food, gorengan). "
                    "Ganti dengan makanan tinggi protein dan serat agar kenyang lebih lama."))

            # Minum air
            ch2o_num = parse_int_prefix(ch2o)
            if ch2o_num < 2:
                saran_list.append(("💧", "Konsumsi Air",
                    "Asupan air Anda kurang. Targetkan minimal 8 gelas (2 liter) per hari. "
                    "Air putih membantu metabolisme dan mengurangi rasa lapar palsu."))

            # Screen time
            tue_num = parse_int_prefix(tue)
            if tue_num >= 2:
                saran_list.append(("📱", "Waktu Layar",
                    "Penggunaan gadget/layar Anda lebih dari 5 jam/hari. "
                    "Coba gunakan teknik 20-20-20: setiap 20 menit, istirahat 20 detik, "
                    "lihat benda 20 kaki jauhnya. Batasi layar sebelum tidur."))

            # Transportasi
            if mtrans in ["Mobil", "Motor"]:
                saran_list.append(("🚶", "Transportasi & Gerak",
                    "Anda sering menggunakan kendaraan bermotor. Sesekali coba jalan kaki "
                    "atau bersepeda untuk jarak dekat, atau parkir lebih jauh dari tujuan."))

            # Sayur
            fcvc_num = parse_int_prefix(fcvc)
            if fcvc_num < 2:
                saran_list.append(("🥦", "Konsumsi Sayuran",
                    "Perbanyak konsumsi sayur dan buah setiap hari. Isi setengah piring Anda "
                    "dengan sayuran berwarna — kaya serat, vitamin, dan rendah kalori."))

            # Merokok
            if smoke == "Ya":
                saran_list.append(("🚭", "Merokok",
                    "Merokok meningkatkan risiko berbagai penyakit kronis. "
                    "Pertimbangkan program berhenti merokok dengan bantuan tenaga kesehatan."))

            st.markdown('<div class="stitle">💡 Saran Kesehatan Personal</div>', unsafe_allow_html=True)
            for ico, judul, isi in saran_list:
                st.markdown(f"""
                <div style="background:#fff; border:1px solid #E2E8F0; border-radius:12px;
                            padding:0.85rem 1rem; margin-bottom:0.55rem;
                            border-left:4px solid {TEAL};">
                  <div style="font-weight:700; font-size:0.85rem; color:#0F172A;
                               margin-bottom:0.2rem;">{ico} {judul}</div>
                  <div style="font-size:0.8rem; color:#475569; line-height:1.6;">{isi}</div>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: DASHBOARD & METRIK
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("""
    <div class="infobox">
      <strong>📌 Tentang Dashboard:</strong> Halaman ini menampilkan statistik dataset,
      distribusi kelas target, serta perbandingan performa model <em>baseline</em>
      vs <em>optimized</em> untuk semua model yang diuji.
    </div>
    """, unsafe_allow_html=True)

    # Statistik dataset
    st.markdown('<div class="stitle">📦 Statistik Dataset</div>', unsafe_allow_html=True)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        k1, k2, k3, k4 = st.columns(4)
        for col_w, ico, val, lbl in [
            (k1, "🗃️", f"{df.shape[0]:,}", "Total Sampel"),
            (k2, "📐", str(df.shape[1] - 1), "Jumlah Fitur"),
            (k3, "🎯", str(df["NObeyesdad"].nunique()), "Kelas Target"),
            (k4, "✅", "0", "Missing Value"),
        ]:
            col_w.markdown(
                f'<div class="mtile"><div class="mico">{ico}</div>'
                f'<div class="mval">{val}</div><div class="mlbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

        # Distribusi kelas
        st.markdown('<div class="stitle">📊 Distribusi Kelas Target</div>', unsafe_allow_html=True)
        vc = df["NObeyesdad"].value_counts().reset_index()
        vc.columns = ["Kelas", "Jumlah"]
        vc["Kelas_ID"] = vc["Kelas"].map(lambda x: LABEL_MAP_ID.get(x, x))
        vc["Warna"]    = vc["Kelas"].map(CHART_COLORS)

        fig_vc = go.Figure(go.Bar(
            x=vc["Kelas_ID"], y=vc["Jumlah"],
            marker_color=vc["Warna"], marker_line_width=0,
            text=vc["Jumlah"], textposition="outside",
            textfont=dict(size=11, color="#334155"),
        ))
        fig_vc.update_layout(
            height=300, margin=dict(l=0, r=0, t=15, b=0),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(title="", tickangle=-20, tickfont=dict(size=11), showgrid=False),
            yaxis=dict(title="Jumlah Sampel", showgrid=True, gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_vc, use_container_width=True, config={"displayModeBar": False})

        # Distribusi Usia & BMI
        if "Age" in df.columns and "Height" in df.columns and "Weight" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)
            col_h1, col_h2 = st.columns(2)

            with col_h1:
                st.markdown('<div class="stitle">📈 Distribusi Usia</div>', unsafe_allow_html=True)
                fig_age = px.histogram(df, x="Age", nbins=20, color_discrete_sequence=[TEAL])
                fig_age.update_traces(marker_line_width=0, opacity=0.85)
                fig_age.update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    xaxis=dict(title="Usia (tahun)", showgrid=False),
                    yaxis=dict(title="Jumlah", showgrid=True, gridcolor="#F1F5F9"),
                    bargap=0.06,
                )
                st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

            with col_h2:
                st.markdown('<div class="stitle">📈 Distribusi BMI</div>', unsafe_allow_html=True)
                fig_bmi = px.histogram(df, x="BMI", nbins=25, color_discrete_sequence=["#7C3AED"])
                fig_bmi.update_traces(marker_line_width=0, opacity=0.8)
                for bv, bn, bc in [
                    (18.5, "Kurang", "#3B82F6"),
                    (25.0, "Normal", "#10B981"),
                    (30.0, "Overweight", "#F59E0B"),
                    (40.0, "Obesitas", "#EF4444"),
                ]:
                    fig_bmi.add_vline(
                        x=bv, line_dash="dash", line_color=bc,
                        annotation_text=bn,
                        annotation_font=dict(size=9, color=bc),
                        annotation_position="top",
                    )
                fig_bmi.update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    xaxis=dict(title="Nilai BMI (kg/m²)", showgrid=False),
                    yaxis=dict(title="Jumlah", showgrid=True, gridcolor="#F1F5F9"),
                    bargap=0.06,
                )
                st.plotly_chart(fig_bmi, use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("⚠️ Dataset tidak ditemukan. Pastikan file CSV ada di folder `data/`.")

    # Baseline vs Optimized
    exp_path = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    if os.path.exists(exp_path):
        st.markdown('<div class="stitle">⚔️ Perbandingan Baseline vs Optimized (3 Model Wajib)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="infobox">
          Grafik ini membandingkan performa model sebelum (<em>baseline</em>) dan sesudah
          (<em>optimized</em>) tuning hyperparameter menggunakan metrik <strong>F1-macro</strong>.
          Semakin tinggi nilainya, semakin baik performa model.
        </div>
        """, unsafe_allow_html=True)
        res = pd.read_csv(exp_path)
        fig_exp = px.bar(
            res, x="Model", y="F1-macro", color="Stage", barmode="group",
            text="F1-macro",
            color_discrete_map={"Baseline": "#CBD5E1", "Optimized": TEAL},
            labels={"F1-macro": "Skor F1-Macro", "Stage": "Tahap"},
        )
        fig_exp.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig_exp.update_layout(
            height=310, margin=dict(l=0, r=0, t=15, b=0),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(title="", showgrid=False),
            yaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#F1F5F9", title="Skor F1-Macro"),
            legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title=""),
        )
        st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False})

    # 11 Model
    eleven_path = os.path.join(REPORTS_DIR, "eleven_model_comparison.csv")
    if os.path.exists(eleven_path):
        st.markdown('<div class="stitle">🏆 Peringkat Performa 11 Model (F1-Macro)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="infobox">
          Peringkat semua 11 model yang diuji, diurutkan dari tertinggi ke terendah.
          <strong style="color:#0D9488;">Warna hijau-teal</strong> = 3 model wajib
          (KNN, Naive Bayes, SVM optimized).
          <strong style="color:#94A3B8;">Warna abu-abu</strong> = model pembanding tambahan.
        </div>
        """, unsafe_allow_html=True)
        e11 = pd.read_csv(eleven_path).sort_values("F1-macro", ascending=True)
        WAJIB = ["KNN (optimized)", "SVM (optimized)", "NaiveBayes (optimized)"]
        clrs  = [TEAL if m in WAJIB else SLATE for m in e11["Model"]]
        fig11 = go.Figure(go.Bar(
            y=e11["Model"], x=e11["F1-macro"], orientation="h",
            marker_color=clrs, marker_line_width=0,
            text=[f"{v:.3f}" for v in e11["F1-macro"]],
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
        ))
        fig11.update_layout(
            height=400, margin=dict(l=0, r=70, t=15, b=10),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#F1F5F9",
                       showticklabels=False, zeroline=False),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig11, use_container_width=True, config={"displayModeBar": False})
        ca, cb = st.columns(2)
        ca.caption("🟢 Warna hijau-teal = model wajib (KNN, Naive Bayes, SVM optimized)")
        cb.caption("⬜ Warna abu = model pembanding tambahan")
        with st.expander("📋 Lihat Tabel Lengkap Semua Model"):
            df_11 = pd.read_csv(eleven_path).sort_values("F1-macro", ascending=False)
            df_11.index = range(1, len(df_11) + 1)
            st.dataframe(df_11, use_container_width=True, hide_index=False)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TENTANG MODEL & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="infobox">
      <strong>📌 Tentang Halaman Ini:</strong> Penjelasan lengkap mengenai arsitektur model,
      pipeline preprocessing, performa 3 model wajib, dataset yang digunakan,
      penjelasan 7 kelas obesitas, serta teknologi yang mendukung aplikasi ini.
    </div>
    """, unsafe_allow_html=True)

    la, ra = st.columns(2, gap="large")

    with la:
        # Model Aktif
        st.markdown('<div class="stitle">🤖 Model Aktif</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Model yang Sedang Digunakan</div>
          <code style="color:{TEAL}; font-size:0.95rem; font-weight:700;">{model_name}</code>
          <div style="margin-top:0.4rem; font-size:0.8rem; color:#64748B;">
            Model terbaik hasil seleksi dari 11 algoritma machine learning.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Pipeline
        st.markdown('<div class="stitle">⚙️ Pipeline Preprocessing (8 Langkah)</div>',
                    unsafe_allow_html=True)
        steps = [
            ("1", "Hapus Duplikat",     "24 baris duplikat dihapus → 2087 baris bersih"),
            ("2", "Binary Encoding",    "Gender, Family_History, FAVC, SMOKE, SCC → nilai 0 atau 1"),
            ("3", "Ordinal Encoding",   "CAEC & CALC → skala angka 0–3 (Tidak Pernah → Selalu)"),
            ("4", "One-Hot Encoding",   "MTRANS dipecah menjadi 5 kolom biner"),
            ("5", "Winsorizing Outlier","Nilai ekstrem dikap pada batas IQR × 1.5"),
            ("6", "Target Encoding",    "7 kelas target → integer 0–6"),
            ("7", "Train-Test Split",   "80% latih : 20% uji, stratified, random_state=42"),
            ("8", "StandardScaler",     "Normalisasi fitur (mean=0, std=1), fit pada data latih saja"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for num, name, desc in steps:
            st.markdown(f"""
            <div class="pstep">
              <span class="pstep-num">{num}</span>
              <div>
                <div style="font-weight:600; color:#0F172A;">{name}</div>
                <div style="font-size:0.775rem; color:#64748B; margin-top:0.1rem;">{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Dataset
        st.markdown('<div class="stitle">📂 Informasi Dataset</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        for cw, ico, val, lbl in [
            (d1, "📊", "2111", "Total Baris"),
            (d2, "📐", "16",   "Fitur Input"),
            (d3, "🎯", "7",    "Kelas Target"),
            (d4, "🧬", "77%",  "Data Sintetis"),
        ]:
            cw.markdown(
                f'<div class="mtile"><div class="mico">{ico}</div>'
                f'<div class="mval" style="font-size:1.2rem;">{val}</div>'
                f'<div class="mlbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )
        st.markdown("""
        <div class="card" style="margin-top:0.6rem;">
          <div style="font-size:0.82rem; color:#64748B; line-height:1.7;">
            <b style="color:#334155;">Sumber:</b> UCI Machine Learning Repository<br>
            <b style="color:#334155;">Judul:</b> Estimation of Obesity Levels Based on
            Eating Habits and Physical Condition<br>
            <b style="color:#334155;">Asal Data Asli:</b> Kolombia, Peru, Meksiko<br>
            <b style="color:#334155;">Data Sintetis:</b> Dibuat menggunakan SMOTE / Weka
          </div>
        </div>
        """, unsafe_allow_html=True)

    with ra:
        # Performa 3 Model
        st.markdown('<div class="stitle">🏅 Performa 3 Model Wajib (Setelah Optimasi)</div>',
                    unsafe_allow_html=True)
        models_data = [
            ("KNN (Optimized)",         "0.8574", "0.8585",
             "k=5, Jarak Manhattan, bobot jarak",  "#3B82F6", "📍"),
            ("Naive Bayes (Optimized)", "0.5294", "0.5721",
             "var_smoothing = 0.1 (Gaussian NB)", "#7C3AED", "📊"),
            ("SVM (Optimized)",         "0.9615", "0.9611",
             "Kernel Linear, C = 10",              "#10B981", "⚡"),
        ]
        for name, f1, ba, params, color, ico in models_data:
            bar_w = float(f1) * 100
            st.markdown(f"""
            <div class="mcard">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <div style="font-weight:700; font-size:0.9rem; color:#0F172A;">{ico} {name}</div>
                  <div style="font-size:0.75rem; color:#94A3B8; margin-top:0.1rem;">{params}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-family:'Space Grotesk',sans-serif; font-size:1.1rem;
                               font-weight:800; color:{color};">F1 = {f1}</div>
                  <div style="font-size:0.72rem; color:#64748B;">Balanced Acc: {ba}</div>
                </div>
              </div>
              <div style="margin-top:0.6rem;">
                <div style="display:flex; justify-content:space-between;
                             font-size:0.72rem; color:#94A3B8; margin-bottom:0.25rem;">
                  <span>Skor F1-Macro</span><span>{bar_w:.1f}%</span>
                </div>
                <div class="pbar-wrap">
                  <div class="pbar-fill" style="width:{bar_w:.1f}%; background:{color};"></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warn">
          <strong>⚠️ Etika AI:</strong> Aplikasi ini adalah <em>alat bantu keputusan akademik</em>.
          Output tidak boleh menjadi satu-satunya dasar keputusan medis.
          Data pengguna bersifat anonim dan tidak disimpan.
        </div>
        """, unsafe_allow_html=True)

        # 7 Kelas Obesitas
        st.markdown('<div class="stitle">🏷️ Penjelasan 7 Kelas Obesitas</div>',
                    unsafe_allow_html=True)
        kelas_list = [
            ("b-blue",   "⚖️", "Berat Badan Kurang",    "BMI < 18.5",   "Tingkatkan asupan kalori bergizi"),
            ("b-green",  "✅", "Berat Badan Normal",     "18.5 – 24.9",  "Pertahankan gaya hidup sehat"),
            ("b-yellow", "⚠️", "Overweight Tingkat I",  "25.0 – 27.4",  "Mulai perhatikan pola makan"),
            ("b-orange", "🔶", "Overweight Tingkat II", "27.5 – 29.9",  "Konsultasi dokter disarankan"),
            ("b-red",    "🔴", "Obesitas Tipe I",        "30.0 – 34.9",  "Penanganan medis diperlukan"),
            ("b-darkred","🟥", "Obesitas Tipe II",       "35.0 – 39.9",  "Intervensi medis segera"),
            ("b-purple", "❗", "Obesitas Tipe III",      "BMI ≥ 40",     "Penanganan medis darurat"),
        ]
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for badge, ico, nama, bmi_r, ket in kelas_list:
            st.markdown(f"""
            <div class="krow">
              <span class="rbadge {badge}" style="min-width:88px; text-align:center;
                    white-space:nowrap; font-size:0.7rem;">{bmi_r}</span>
              <div>
                <div style="font-weight:600; color:#0F172A; font-size:0.83rem;">{ico} {nama}</div>
                <div style="color:#64748B; font-size:0.74rem;">{ket}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tech Stack
    st.markdown('<div class="stitle">🛠️ Teknologi yang Digunakan</div>', unsafe_allow_html=True)
    tc = st.columns(6)
    techs = [
        ("🐍", "Python 3.12",   "Bahasa Pemrograman"),
        ("📊", "scikit-learn",  "Framework ML"),
        ("⚡", "LightGBM",     "Gradient Boosting"),
        ("🎛️", "Streamlit",    "Framework Web App"),
        ("📦", "joblib",        "Penyimpanan Model"),
        ("📉", "Plotly",        "Visualisasi Interaktif"),
    ]
    for col, (ico, name, desc) in zip(tc, techs):
        col.markdown(f"""
        <div class="ttile">
          <div style="font-size:1.6rem; margin-bottom:0.35rem;">{ico}</div>
          <div style="font-family:'Space Grotesk',sans-serif; font-size:0.8rem;
                      font-weight:700; color:#0F172A;">{name}</div>
          <div style="font-size:0.72rem; color:#64748B;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align:center; padding:1.2rem; border-top:1px solid #E2E8F0;
                font-size:0.78rem; color:#94A3B8; margin-top:0.5rem;">
      🩺 <strong style="color:{TEAL};">ObesityAI</strong> &mdash;
      UAS Pembelajaran Mesin &middot; Universitas Dian Nuswantoro<br>
      Dibuat untuk keperluan akademik &middot; Bukan pengganti diagnosis medis profesional.
    </div>
    """, unsafe_allow_html=True)
