# ☁️ Panduan Deploy ObesityAI ke AWS EC2 (Free Tier 1 Tahun)

AWS Free Tier memberikan Anda server VPS (Virtual Private Server) **menyala 24/7 secara gratis selama 1 tahun pertama**. Panduan ini akan menunjukkan cara meng-online-kan **API FastAPI** dan **Website PHP** Anda di server tersebut.

---

## 🛠️ Tahap 1: Membuat Server EC2 di AWS

1. **Daftar Akun AWS:**
   - Kunjungi [aws.amazon.com/free](https://aws.amazon.com/free/).
   - Buat akun (Anda akan diminta memasukkan kartu kredit/debit berlogo Visa/Mastercard/Jago/Jenius untuk verifikasi. AWS akan memotong sekitar Rp 15.000 lalu mengembalikannya lagi).

2. **Buat Instance EC2 (Server Baru):**
   - Setelah masuk ke dashboard AWS, cari layanan **EC2** di kolom pencarian atas.
   - Klik tombol oranye **"Launch instance"**.
   - **Name:** Beri nama `ObesityAI-Server`
   - **OS Image:** Pilih **Ubuntu 24.04 LTS** atau **Ubuntu 22.04 LTS** (pastikan ada tulisan *Free tier eligible*).
   - **Instance Type:** Pilih **t2.micro** atau **t3.micro** (Ini yang gratis, memiliki 1GB RAM).
   - **Key Pair (Login):** Klik *Create new key pair*. Beri nama `aws-key`, pilih format `.pem`. File ini akan otomatis terdownload ke laptop Anda. **SIMPAN FILE INI BAIK-BAIK**, ini adalah "kunci rumah" server Anda.

3. **Pengaturan Jaringan (Network Settings):**
   - Centang **Allow SSH traffic from Anywhere**
   - Centang **Allow HTTP traffic from the internet** (Penting agar web bisa diakses)
   - Centang **Allow HTTPS traffic from the internet**
   - Terakhir, klik **Launch instance**.

---

## 🔌 Tahap 2: Mengakses Server Anda

Setelah server menyala (status *Running*), klik server tersebut dan salin **Public IPv4 address**-nya (misal: `13.250.xx.xx`).

Buka Terminal/Command Prompt di laptop Anda tempat file `.pem` tadi berada, lalu ketik:

```bash
# Ubah permission kunci (khusus Linux/Mac, Windows skip saja)
chmod 400 aws-key.pem

# Masuk ke server AWS Anda
ssh -i "aws-key.pem" ubuntu@IP_SERVER_ANDA
```
Jika ada pertanyaan `Are you sure you want to continue connecting?`, ketik `yes` lalu Enter. Anda sekarang sudah masuk ke dalam komputer server milik AWS!

---

## 📦 Tahap 3: Instalasi Python & Download Project

Sekarang Anda berada di dalam server AWS. Jalankan perintah berikut satu per satu:

```bash
# 1. Update sistem operasi
sudo apt update && sudo apt upgrade -y

# 2. Install Python dan pip
sudo apt install python3-pip python3-venv git -y

# 3. Clone (Download) project Anda dari Github
git clone https://github.com/USERNAME/uas-ml-obesitas.git
cd uas-ml-obesitas

# 4. Buat virtual environment agar rapi
python3 -m venv venv
source venv/bin/activate

# 5. Install semua library yang dibutuhkan
pip install -r requirements.txt
pip install fastapi uvicorn pydantic
```

---

## 🚀 Tahap 4: Menjalankan FastAPI (Backend) Non-Stop

Agar API Anda (`api_fastapi.py`) tetap menyala walaupun Anda menutup Terminal laptop, kita akan menggunakan alat bernama `tmux` atau `nohup`. Kita gunakan `nohup` karena lebih mudah:

```bash
# Masih di dalam folder uas-ml-obesitas
nohup uvicorn api_fastapi:app --host 0.0.0.0 --port 8000 &
```
*Tanda `&` di belakang akan membuat proses berjalan di background.*

> **⚠️ PENTING - Buka Port 8000 di AWS:**
> Secara default, AWS memblokir port 8000. 
> 1. Pergi ke dashboard AWS EC2 -> Klik Instance Anda -> Tab **Security** -> Klik tulisan *sg-xxxx (launch-wizard)*.
> 2. Klik **Edit inbound rules** -> **Add rule**.
> 3. Type: `Custom TCP`, Port Range: `8000`, Source: `Anywhere-IPv4` (`0.0.0.0/0`).
> 4. Save rules.

Cek di browser: `http://IP_SERVER_ANDA:8000/docs`. Jika muncul tampilan Swagger UI, berarti API Anda sudah sukses online!

---

## 🌐 Tahap 5: Menjalankan Web PHP (Frontend)

Sekarang kita jalankan website antarmuka (`index.php`).

1. Edit file `index.php` untuk mengganti alamat API. Gunakan teks editor `nano`:
   ```bash
   nano index.php
   ```
   - Cari baris: `$api_url = "http://localhost:8000/predict";`
   - Ganti tulisan `localhost` menjadi IP Server Anda. Contoh: `$api_url = "http://13.250.xx.xx:8000/predict";`
   - Simpan: Tekan `Ctrl + X`, lalu tekan `Y`, lalu `Enter`.

2. Jalankan server PHP menggunakan Nginx atau Apache. Karena ini project cepat, kita bisa menjalankan PHP bawaan di background via port 80 (port web standar):
   ```bash
   # Install PHP
   sudo apt install php-cli -y

   # Jalankan web server PHP di port 80 (harus pakai sudo)
   sudo nohup php -S 0.0.0.0:80 &
   ```

## 🎉 SELESAI!

Sekarang Anda bisa memberikan **IP Server AWS** Anda ke dosen penguji! 
Misalnya dosen membuka: **`http://13.250.xx.xx`** di HP atau laptopnya, mereka akan melihat Website UI Premium Anda. Saat mengisi form dan klik tombol, web PHP akan berkomunikasi dengan API Python di *background* server yang sama, dan langsung menampilkan hasilnya! 

Sistem ini akan **ON Terus 24 Jam Non-stop** selama setahun penuh.
