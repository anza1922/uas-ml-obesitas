# Data Dictionary — Obesity Level Estimation Dataset

Sumber: *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*
(UCI Machine Learning Repository / Kaggle), 2111 baris, 17 kolom asli.

| Fitur | Tipe Data Asli | Deskripsi | Satuan/Kategori | Hasil Encoding |
|---|---|---|---|---|
| Gender | string | Jenis kelamin | Male / Female | Binary 0/1 |
| Age | float | Usia | tahun | Winsorized + StandardScaler |
| Height | float | Tinggi badan | meter | Winsorized + StandardScaler |
| Weight | float | Berat badan | kg | Winsorized + StandardScaler |
| family_history_with_overweight | string | Riwayat keluarga dengan kelebihan berat badan | yes/no | Binary 0/1 |
| FAVC | string | Sering konsumsi makanan berkalori tinggi | yes/no | Binary 0/1 |
| FCVC | float | Frekuensi konsumsi sayur | skala 1-3 | Winsorized + StandardScaler |
| NCP | float | Jumlah makan utama per hari | skala 1-4 | Winsorized + StandardScaler |
| CAEC | string | Konsumsi makanan di antara waktu makan utama | no/Sometimes/Frequently/Always | Ordinal 0-3 |
| SMOKE | string | Status merokok | yes/no | Binary 0/1 |
| CH2O | float | Konsumsi air harian | liter | Winsorized + StandardScaler |
| SCC | string | Memantau konsumsi kalori harian | yes/no | Binary 0/1 |
| FAF | float | Frekuensi aktivitas fisik | skala 0-3 | Winsorized + StandardScaler |
| TUE | float | Waktu penggunaan perangkat teknologi | skala 0-2 | Winsorized + StandardScaler |
| CALC | string | Frekuensi konsumsi alkohol | no/Sometimes/Frequently/Always | Ordinal 0-3 |
| MTRANS | string | Moda transportasi utama | Automobile/Bike/Motorbike/Public_Transportation/Walking | One-Hot (5 kolom) |
| NObeyesdad | string | **Target**: tingkat obesitas | 7 kelas (lihat di bawah) | Integer 0-6 |

## Urutan Label Target (severity, dipakai sebagai integer encoding)
0. Insufficient_Weight
1. Normal_Weight
2. Overweight_Level_I
3. Overweight_Level_II
4. Obesity_Type_I
5. Obesity_Type_II
6. Obesity_Type_III

## Catatan Etika, Privasi, dan Batasan Penggunaan
- Dataset bersifat publik/anonim, tidak memuat identitas pribadi (nama, NIM, kontak).
- Sebagian data merupakan data sintetik hasil SMOTE (bukan murni hasil survei), sehingga generalisasi
  ke populasi nyata perlu dilakukan dengan hati-hati.
- **Potensi data leakage**: label target diturunkan langsung dari BMI (Weight/Height^2), sehingga
  model yang menyertakan Height & Weight akan memiliki akurasi sangat tinggi yang merefleksikan
  rumus BMI, bukan murni pola perilaku. Hal ini didokumentasikan secara eksplisit untuk transparansi
  akademik (lihat bagian 1.6 pada notebook audit).
- Output model dalam aplikasi akhir (Soal 05) wajib diposisikan sebagai *decision support* (skrining
  awal), bukan diagnosis medis definitif, dan tidak digunakan untuk keputusan tunggal apa pun
  terhadap individu.
