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
    page_title="ObesityAI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main .block-container{padding:1.5rem 2rem 3rem;max-width:1120px}
#MainMenu,footer,header{visibility:hidden}
.app-header{background:linear-gradient(135deg,#134E4A 0%,#0D9488 100%);border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;display:flex;align-items:center;gap:1.5rem}
.app-header h1{font-family:'Space Grotesk',sans-serif;font-size:1.8rem;font-weight:700;color:#fff;margin:0;line-height:1.2}
.app-header p{color:#99F6E4;font-size:.875rem;margin:.3rem 0 0}
.header-icon{font-size:3rem;line-height:1}
.stTabs [data-baseweb="tab-list"]{gap:0;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:4px;margin-bottom:1.25rem}
.stTabs [data-baseweb="tab"]{border-radius:7px;padding:.45rem 1.1rem;font-size:.875rem;font-weight:500;color:#64748B}
.stTabs [aria-selected="true"]{background:#fff !important;color:#0F766E !important;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.card{background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.card-title{font-size:.75rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:#64748B;margin-bottom:.5rem}
.result-badge{display:inline-block;padding:.3rem .9rem;border-radius:999px;font-size:.78rem;font-weight:600;letter-spacing:.04em}
.badge-safe{background:#DCFCE7;color:#15803D}
.badge-warn{background:#FEF3C7;color:#92400E}
.badge-danger{background:#FEE2E2;color:#991B1B}
.metric-tile{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:1rem 1.25rem;text-align:center}
.metric-tile .val{font-family:'Space Grotesk',sans-serif;font-size:1.6rem;font-weight:700;color:#0F766E}
.metric-tile .lbl{font-size:.78rem;color:#64748B;margin-top:.2rem}
.sec-title{font-family:'Space Grotesk',sans-serif;font-size:1rem;font-weight:600;color:#0F172A;margin:1.25rem 0 .75rem;padding-bottom:.4rem;border-bottom:2px solid #CCFBF1}
.disclaimer{background:#CCFBF1;border-left:3px solid #0D9488;border-radius:0 8px 8px 0;padding:.75rem 1rem;font-size:.8rem;color:#0F766E;margin-top:.75rem}
.stButton>button{background:#0F766E !important;color:#fff !important;border:none !important;border-radius:10px !important;font-weight:600 !important;font-size:.95rem !important;padding:.65rem 0 !important;width:100%}
.stButton>button:hover{background:#0D9488 !important}
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {
    "Insufficient_Weight": "badge-warn",
    "Normal_Weight":       "badge-safe",
    "Overweight_Level_I":  "badge-warn",
    "Overweight_Level_II": "badge-warn",
    "Obesity_Type_I":      "badge-danger",
    "Obesity_Type_II":     "badge-danger",
    "Obesity_Type_III":    "badge-danger",
}
ORD_MAP = {"Tidak pernah": 0, "Kadang-kadang": 1, "Sering": 2, "Selalu": 3}
TEAL = "#0D9488"; SLATE = "#94A3B8"

@st.cache_resource(show_spinner="Memuat model...")
def get_artifacts():
    return load_artifacts()

model, model_name, scaler, oe = get_artifacts()

st.markdown("""
<div class="app-header">
  <div class="header-icon">&#x1FA7A;</div>
  <div>
    <h1>ObesityAI &mdash; Prediksi Tingkat Obesitas</h1>
    <p>UAS Pembelajaran Mesin &middot; Universitas Dian Nuswantoro
       &middot; KNN &middot; Naive Bayes &middot; SVM &middot; LightGBM (11 model)</p>
  </div>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model tidak ditemukan. Jalankan `python src/train_optimization.py` terlebih dahulu.")
    st.stop()

tab_pred, tab_dash, tab_about = st.tabs(["🔮  Prediksi", "📊  Dashboard & Metrik", "ℹ️  Tentang Model"])

# ── TAB 1: PREDIKSI ─────────────────────────────────────────────────────────
with tab_pred:
    st.markdown('<div class="sec-title">Data Individu</div>', unsafe_allow_html=True)
    with st.form("form_prediksi"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Profil Fisik**")
            gender  = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], horizontal=True)
            age     = st.number_input("Usia (tahun)", 14, 65, 22)
            height  = st.number_input("Tinggi badan (m)", 1.40, 2.10, 1.70, 0.01, format="%.2f")
            weight  = st.number_input("Berat badan (kg)", 30.0, 250.0, 70.0, 0.5)
        with c2:
            st.markdown("**Kebiasaan Makan**")
            family_history = st.checkbox("Riwayat keluarga obesitas")
            favc    = st.checkbox("Sering konsumsi makanan tinggi kalori (FAVC)")
            fcvc    = st.slider("Frekuensi konsumsi sayur (FCVC)", 1.0, 3.0, 2.0, 0.5)
            ncp     = st.slider("Jumlah makan utama/hari (NCP)", 1.0, 4.0, 3.0, 0.5)
            caec    = st.selectbox("Camilan antar makan (CAEC)",
                                   ["Tidak pernah","Kadang-kadang","Sering","Selalu"])
            calc    = st.selectbox("Konsumsi alkohol (CALC)",
                                   ["Tidak pernah","Kadang-kadang","Sering","Selalu"])
        with c3:
            st.markdown("**Gaya Hidup**")
            smoke   = st.checkbox("Merokok")
            scc     = st.checkbox("Pantau kalori harian (SCC)")
            ch2o    = st.slider("Konsumsi air harian, liter (CH2O)", 1.0, 3.0, 2.0, 0.5)
            faf     = st.slider("Frekuensi olahraga/minggu (FAF)", 0.0, 3.0, 1.0, 0.5)
            tue     = st.slider("Waktu layar/hari (TUE)", 0.0, 2.0, 1.0, 0.5)
            mtrans  = st.selectbox("Transportasi utama",
                                   ["Public_Transportation","Automobile","Motorbike","Bike","Walking"])
        submitted = st.form_submit_button("🔍  Prediksi Tingkat Obesitas", use_container_width=True)

    if submitted:
        df_in = preprocess_input(
            gender_male=(gender == "Laki-laki"), age=age, height=height, weight=weight,
            family_history=family_history, favc=favc, fcvc=fcvc, ncp=ncp,
            caec_level=ORD_MAP[caec], smoke=smoke, ch2o=ch2o, scc=scc,
            faf=faf, tue=tue, calc_level=ORD_MAP[calc], mtrans=mtrans,
        )
        pred_class, probs = predict(df_in, model, scaler)
        bmi_val, bmi_cat  = bmi_info(height, weight)

        st.markdown('<div class="sec-title">Hasil Analisis</div>', unsafe_allow_html=True)
        col_res, col_chart = st.columns([1, 1.5], gap="large")

        with col_res:
            badge = BADGE_CLASS.get(pred_class, "badge-warn")
            conf  = probs.max() * 100
            st.markdown(f"""
            <div class="card">
              <div class="card-title">Tingkat Obesitas Terdeteksi</div>
              <div style="font-family:'Space Grotesk',sans-serif;font-size:1.4rem;
                          font-weight:700;color:#0F172A;margin:.25rem 0 .5rem">
                {CLASS_LABELS[pred_class]}</div>
              <span class="result-badge {badge}">Keyakinan model {conf:.1f}%</span>
              <div style="display:flex;gap:.75rem;margin-top:1.1rem">
                <div class="metric-tile" style="flex:1">
                  <div class="val">{bmi_val}</div><div class="lbl">BMI</div>
                </div>
                <div class="metric-tile" style="flex:1.6">
                  <div class="val" style="font-size:.9rem;padding:.2rem 0">{bmi_cat}</div>
                  <div class="lbl">Kategori BMI</div>
                </div>
              </div>
              <div style="margin-top:1rem;padding:.85rem 1rem;background:#F8FAFC;
                          border-radius:8px;border:1px solid #E2E8F0">
                <div style="font-size:.72rem;font-weight:600;text-transform:uppercase;
                             letter-spacing:.05em;color:#64748B;margin-bottom:.3rem">Rekomendasi</div>
                <div style="font-size:.875rem;color:#0F172A;line-height:1.5">
                  {REKOMENDASI[pred_class]}</div>
              </div>
            </div>
            <div class="disclaimer">
              <strong>Disclaimer:</strong> Hasil ini bersifat <em>decision support</em>,
              bukan diagnosis medis. Konsultasikan dengan tenaga kesehatan.
            </div>
            """, unsafe_allow_html=True)

        with col_chart:
            bar_clrs = [TEAL if ORDER_TARGET[i] == pred_class else "#CBD5E1" for i in range(7)]
            fig = go.Figure(go.Bar(
                x=probs * 100,
                y=[CLASS_LABELS[c] for c in ORDER_TARGET],
                orientation="h",
                marker_color=bar_clrs, marker_line_width=0,
                text=[f"{v:.1f}%" for v in probs * 100],
                textposition="outside",
                textfont=dict(size=11, color="#334155"),
            ))
            fig.update_layout(
                title=dict(text="Probabilitas per Kelas", font=dict(size=13), x=0),
                height=310, margin=dict(l=0, r=55, t=35, b=10),
                xaxis=dict(range=[0, 115], showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── TAB 2: DASHBOARD ─────────────────────────────────────────────────────────
with tab_dash:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        k1,k2,k3,k4 = st.columns(4)
        for col,val,lbl in [(k1,f"{df.shape[0]:,}","Total Sampel"),(k2,str(df.shape[1]-1),"Jumlah Fitur"),
                             (k3,str(df["NObeyesdad"].nunique()),"Kelas Target"),(k4,"0","Missing Value")]:
            col.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                         unsafe_allow_html=True)

        st.markdown('<div class="sec-title">Distribusi Kelas Target</div>', unsafe_allow_html=True)
        vc = df["NObeyesdad"].value_counts().reset_index()
        vc.columns = ["Kelas","Jumlah"]
        fig_vc = px.bar(vc, x="Kelas", y="Jumlah", text="Jumlah", color_discrete_sequence=[TEAL])
        fig_vc.update_traces(textposition="outside", marker_line_width=0)
        fig_vc.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="#fff",
                              paper_bgcolor="#fff", xaxis=dict(title="",tickangle=-20),
                              yaxis=dict(title="",showgrid=True,gridcolor="#F1F5F9"))
        st.plotly_chart(fig_vc, use_container_width=True, config={"displayModeBar": False})

    exp_path = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    if os.path.exists(exp_path):
        st.markdown('<div class="sec-title">Baseline vs Optimized - 3 Model Wajib</div>',
                    unsafe_allow_html=True)
        res = pd.read_csv(exp_path)
        fig_exp = px.bar(res, x="Model", y="F1-macro", color="Stage", barmode="group",
                         text="F1-macro",
                         color_discrete_map={"Baseline":"#CBD5E1","Optimized":TEAL})
        fig_exp.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
        fig_exp.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), plot_bgcolor="#fff",
                              paper_bgcolor="#fff", xaxis_title="", legend=dict(
                              orientation="h",y=1.12,x=1,xanchor="right",title=""),
                              yaxis=dict(range=[0,1.1],showgrid=True,gridcolor="#F1F5F9",title="F1-macro"))
        st.plotly_chart(fig_exp, use_container_width=True, config={"displayModeBar": False})

    eleven_path = os.path.join(REPORTS_DIR, "eleven_model_comparison.csv")
    if os.path.exists(eleven_path):
        st.markdown('<div class="sec-title">Perbandingan 11 Model (F1-macro)</div>', unsafe_allow_html=True)
        e11 = pd.read_csv(eleven_path).sort_values("F1-macro", ascending=True)
        WAJIB = ["KNN (optimized)","SVM (optimized)","NaiveBayes (optimized)"]
        clrs  = [TEAL if m in WAJIB else SLATE for m in e11["Model"]]
        fig11 = go.Figure(go.Bar(
            y=e11["Model"], x=e11["F1-macro"], orientation="h",
            marker_color=clrs, marker_line_width=0,
            text=[f"{v:.3f}" for v in e11["F1-macro"]], textposition="outside",
            textfont=dict(size=11,color="#334155"),
        ))
        fig11.update_layout(height=370, margin=dict(l=0,r=60,t=10,b=10), plot_bgcolor="#fff",
                            paper_bgcolor="#fff",
                            xaxis=dict(range=[0,1.1],showgrid=True,gridcolor="#F1F5F9",
                                       showticklabels=False,zeroline=False,title=""),
                            yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig11, use_container_width=True, config={"displayModeBar": False})
        ca,cb = st.columns(2)
        ca.caption("Hijau = model wajib (KNN, NaiveBayes, SVM optimized)")
        cb.caption("Abu = model pembanding tambahan")
        with st.expander("Lihat tabel lengkap"):
            st.dataframe(pd.read_csv(eleven_path).sort_values("F1-macro",ascending=False),
                         use_container_width=True, hide_index=True)

# ── TAB 3: TENTANG ───────────────────────────────────────────────────────────
with tab_about:
    la, ra = st.columns(2, gap="large")
    with la:
        st.markdown('<div class="sec-title">Informasi Project</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Model Aktif</div>
          <code style="color:{TEAL};font-size:.9rem">{model_name}</code>
          <div style="margin-top:1.1rem">
            <div class="card-title">Pipeline Preprocessing (8 Langkah)</div>
            <ol style="font-size:.875rem;color:#334155;margin:.4rem 0 0;padding-left:1.2rem;line-height:2">
              <li>Drop duplikat (24 baris dihapus)</li>
              <li>Binary encoding (Gender, Yes/No vars)</li>
              <li>Ordinal encoding (CAEC, CALC)</li>
              <li>One-hot encoding (MTRANS &rarr; 5 kolom)</li>
              <li>Winsorizing outlier (IQR 1.5&times;)</li>
              <li>Target encoding (7 kelas &rarr; int 0-6)</li>
              <li>Train-test split 80:20 stratified (seed=42)</li>
              <li>StandardScaler (fit pada train only)</li>
            </ol>
          </div>
          <div style="margin-top:1.1rem">
            <div class="card-title">Dataset</div>
            <div style="font-size:.875rem;color:#334155;line-height:1.7">
              UCI Obesity Levels &middot; 2111 baris &middot; 16 fitur &middot; 7 kelas<br>
              77% data sintetis (SMOTE dari Kolombia/Peru/Meksiko)
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with ra:
        st.markdown('<div class="sec-title">Performa 3 Model Wajib</div>', unsafe_allow_html=True)
        for name,f1,ba,params in [
            ("KNN (Optimized)",        "0.8574","0.8585","k=5, manhattan, distance"),
            ("Naive Bayes (Optimized)","0.5294","0.5721","var_smoothing=0.1"),
            ("SVM (Optimized)",        "0.9615","0.9611","linear kernel, C=10"),
        ]:
            bar_w = float(f1) * 100
            st.markdown(f"""
            <div class="card" style="padding:.9rem 1.1rem;margin-bottom:.6rem">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                  <div style="font-weight:600;font-size:.9rem;color:#0F172A">{name}</div>
                  <div style="font-size:.75rem;color:#94A3B8;margin-top:.1rem">{params}</div>
                </div>
                <div style="text-align:right">
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;
                               font-weight:700;color:{TEAL}">F1-macro {f1}</div>
                  <div style="font-size:.75rem;color:#64748B">Bal.Acc {ba}</div>
                </div>
              </div>
              <div style="margin-top:.7rem;background:#E2E8F0;border-radius:4px;height:5px">
                <div style="width:{bar_w:.1f}%;background:{TEAL};border-radius:4px;height:5px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
          <strong>Etika:</strong> Aplikasi ini adalah <em>decision support tool</em>.
          Output tidak boleh menjadi satu-satunya dasar keputusan medis.
          Data bersifat anonim.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Tech Stack</div>', unsafe_allow_html=True)
    tc = st.columns(5)
    for col,(icon,lbl) in zip(tc,[("🐍","Python 3.12"),("📊","scikit-learn"),
                                   ("⚡","LightGBM"),("🎛️","Streamlit"),("📦","joblib")]):
        col.markdown(f'<div class="metric-tile"><div style="font-size:1.5rem">{icon}</div>'
                     f'<div class="lbl" style="margin-top:.3rem">{lbl}</div></div>',
                     unsafe_allow_html=True)
