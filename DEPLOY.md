# Panduan Deploy ke Streamlit Community Cloud (Gratis)

## Prasyarat
- Akun GitHub (gratis)
- Akun Streamlit Cloud: https://share.streamlit.io (login pakai GitHub)

---

## Langkah 1 — Upload project ke GitHub

1. Buka https://github.com/new
2. Buat repo baru, misal: `uas-ml-obesitas`
3. Upload seluruh isi folder project ini ke repo tersebut
   (bisa drag & drop di GitHub web, atau pakai Git):

```bash
git init
git add .
git commit -m "UAS Pembelajaran Mesin - ObesityAI"
git remote add origin https://github.com/USERNAME/uas-ml-obesitas.git
git push -u origin main
```

---

## Langkah 2 — Deploy di Streamlit Cloud

1. Buka https://share.streamlit.io
2. Klik **"New app"**
3. Isi form:
   - **Repository** : `USERNAME/uas-ml-obesitas`
   - **Branch**     : `main`
   - **Main file**  : `app_streamlit.py`
4. Klik **"Deploy!"**
5. Tunggu 2-5 menit -> aplikasi live dengan URL:
   `https://USERNAME-uas-ml-obesitas-app-streamlit-XXXXX.streamlit.app`

---

## Catatan Penting

- **Model `.joblib` harus ikut di-push** ke GitHub (sudah ada di folder `models/`)
- **File besar** (>100MB) perlu Git LFS. Cek ukuran:
  ```bash
  du -sh models/*.joblib | sort -rh | head -5
  ```
  Jika ada yang >100MB, install Git LFS:
  ```bash
  git lfs install
  git lfs track "models/*.joblib"
  git add .gitattributes
  ```
- **packages.txt** tidak diperlukan (semua Python murni)
- Streamlit Cloud otomatis baca `requirements.txt`

---

## Alternatif Deploy Lain

| Platform | Gratis | Cara |
|---|---|---|
| Streamlit Cloud | Ya (terbaik untuk Streamlit) | Cara di atas |
| Railway.app | Trial gratis | Butuh Dockerfile |
| Hugging Face Spaces | Ya | Pilih "Streamlit" saat buat Space |
| Render.com | Tier gratis ada | `streamlit run app_streamlit.py --server.port $PORT` |

**Rekomendasi untuk presentasi**: Streamlit Cloud (paling mudah, URL bersih, tidak perlu konfigurasi server).
