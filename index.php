<?php
$result = null;
$error = null;

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Kumpulkan data dari form
    $data = [
        "gender" => $_POST['gender'],
        "age" => floatval($_POST['age']),
        "height" => floatval($_POST['height']),
        "weight" => floatval($_POST['weight']),
        "family_history_with_overweight" => $_POST['family_history_with_overweight'],
        "favc" => $_POST['favc'],
        "fcvc" => floatval($_POST['fcvc']),
        "ncp" => floatval($_POST['ncp']),
        "caec" => $_POST['caec'],
        "calc" => $_POST['calc'],
        "ch2o" => floatval($_POST['ch2o']),
        "smoke" => $_POST['smoke'],
        "scc" => $_POST['scc'],
        "faf" => floatval($_POST['faf']),
        "tue" => floatval($_POST['tue']),
        "mtrans" => $_POST['mtrans']
    ];

    // Endpoint API FastAPI
    $api_url = "http://localhost:8000/predict";

    // Inisialisasi cURL
    $ch = curl_init($api_url);
    $payload = json_encode($data);
    
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLINFO_HEADER_OUT, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
    
    // Set HTTP Header untuk JSON
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($payload)
    ]);
    
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    
    if(curl_errno($ch)){
        $error = "Gagal menghubungi API: " . curl_error($ch);
    } elseif ($http_code != 200) {
        $error = "Error API (Kode: $http_code): " . $response;
    } else {
        $result = json_decode($response, true);
    }
    
    curl_close($ch);
}
?>
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ObesityAI - PHP Integration</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #0F766E;      /* Teal / Medis */
            --primary-hover: #0D9488;
            --bg-color: #F8FAFC;
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-main: #0F172A;
            --text-muted: #64748B;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: linear-gradient(135deg, #F0FDF4 0%, #E0F2FE 100%); /* Subtle Green-Blue */
            min-height: 100vh;
            margin: 0;
            padding: 2rem;
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 20px 40px rgba(0,0,0,0.08);
            padding: 3rem;
            overflow: hidden;
            position: relative;
        }

        .header {
            text-align: center;
            margin-bottom: 2.5rem;
        }
        
        .header h1 {
            font-weight: 800;
            font-size: 2.5rem;
            margin: 0;
            background: linear-gradient(to right, #0F766E, #0284C7); /* Teal to Sky Blue */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p {
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        label {
            font-weight: 600;
            font-size: 0.9rem;
            color: #334155;
        }

        input, select {
            padding: 0.8rem 1rem;
            border: 2px solid #E2E8F0;
            border-radius: 12px;
            background: #F8FAFC;
            font-family: 'Outfit', sans-serif;
            font-size: 1rem;
            color: var(--text-main);
            transition: all 0.3s ease;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(15, 118, 110, 0.1); /* Teal glow */
            background: #FFFFFF;
        }

        .btn-submit {
            background: linear-gradient(135deg, #0F766E, #0284C7); /* Teal to Blue */
            color: white;
            font-weight: 800;
            font-size: 1.1rem;
            border: none;
            border-radius: 12px;
            padding: 1.2rem;
            width: 100%;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 2rem;
            font-family: 'Outfit', sans-serif;
        }

        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(15, 118, 110, 0.25);
        }

        .result-card {
            background: white;
            border-radius: 16px;
            padding: 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border-top: 6px solid var(--primary);
            animation: slideIn 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .result-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 1.5rem;
        }
        @media (max-width: 768px) {
            .result-grid { grid-template-columns: 1fr; }
        }

        .r-title { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 700; margin-bottom: 0.5rem; }
        .r-class { font-size: 2.5rem; font-weight: 800; color: var(--text-main); margin: 0; line-height: 1.2; }
        
        .badge-info {
            display: inline-flex; align-items: center; gap: 0.5rem;
            padding: 0.5rem 1.2rem; background: #F0FDF4; color: #166534;
            border-radius: 30px; font-weight: 700; font-size: 0.95rem;
            margin: 1rem 0; border: 1px solid #BBF7D0;
        }

        .section-box {
            background: #F8FAFC; border: 1px solid #E2E8F0;
            border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
        }
        
        .box-title { font-weight: 700; color: #1E293B; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }

        .prob-bar-container { margin-bottom: 0.8rem; }
        .prob-label { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.3rem; font-weight: 600; color: #475569; }
        .prob-track { background: #E2E8F0; height: 10px; border-radius: 5px; overflow: hidden; }
        .prob-fill { background: linear-gradient(90deg, #0F766E, #0284C7); height: 100%; border-radius: 5px; transition: width 1s ease-in-out; }
        
        .data-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .data-table tr { border-bottom: 1px solid #E2E8F0; }
        .data-table tr:last-child { border-bottom: none; }
        .data-table td { padding: 0.6rem 0; color: #475569; }
        .data-table td:last-child { text-align: right; font-weight: 700; color: #0F172A; }

        .helper-text { font-size: 0.75rem; color: #94A3B8; margin-top: 0.2rem; font-weight: 400; }

        .error-card {
            background: #FEF2F2;
            color: #991B1B;
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border: 1px solid #F87171;
            font-weight: 600;
        }

    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>🩺 ObesityAI</h1>
        <p>Prediksi Tingkat Obesitas (Integrasi PHP & FastAPI)</p>
    </div>

    <?php if ($error): ?>
        <div class="error-card">🚨 <?php echo $error; ?></div>
    <?php endif; ?>

    <?php if ($result && isset($result['status']) && $result['status'] == 'success'): ?>
        <div class="result-card">
            <div style="text-align: center; margin-bottom: 2rem;">
                <div class="r-title">Tingkat Obesitas Terdeteksi</div>
                <div class="r-class">🎯 <?php echo $result['prediction']['label']; ?></div>
                <div class="badge-info">
                    <span>⚡ Model: <?php echo $result['model_used']; ?></span> | 
                    <span>Akurasi: <?php echo $result['prediction']['confidence_percent']; ?>%</span>
                </div>
            </div>

            <div class="result-grid">
                <!-- Kolom Kiri: Detail Pengguna & BMI -->
                <div>
                    <div class="section-box">
                        <div class="box-title">📏 Indeks Massa Tubuh (BMI)</div>
                        <div style="display: flex; align-items: baseline; gap: 0.5rem;">
                            <span style="font-size: 3rem; font-weight: 800; color: var(--primary);"><?php echo $result['bmi_info']['bmi_value']; ?></span>
                            <span style="font-weight: 600; color: var(--text-muted);">kg/m²</span>
                        </div>
                        <div style="color: #475569; font-weight: 600; margin-top: 0.2rem;">
                            Kategori: <span style="color: #0F172A;"><?php echo $result['bmi_info']['bmi_category']; ?></span>
                        </div>
                    </div>

                    <div class="section-box" style="background: #FFFBEB; border-color: #FDE68A;">
                        <div class="box-title" style="color: #92400E;">💡 Saran & Rekomendasi Medis</div>
                        <div style="color: #92400E; font-size: 0.95rem; line-height: 1.6;">
                            <?php echo $result['recommendation']; ?>
                        </div>
                    </div>

                    <div class="section-box">
                        <div class="box-title">👤 Ringkasan Data Input</div>
                        <table class="data-table">
                            <tr><td>Jenis Kelamin</td><td><?php echo htmlspecialchars($_POST['gender']); ?></td></tr>
                            <tr><td>Usia</td><td><?php echo htmlspecialchars($_POST['age']); ?> Tahun</td></tr>
                            <tr><td>Tinggi / Berat</td><td><?php echo htmlspecialchars($_POST['height']); ?> m / <?php echo htmlspecialchars($_POST['weight']); ?> kg</td></tr>
                            <tr><td>Makan Kalori Tinggi</td><td><?php echo htmlspecialchars($_POST['favc']); ?></td></tr>
                            <tr><td>Olahraga / Minggu</td><td>Level <?php echo htmlspecialchars($_POST['faf']); ?></td></tr>
                        </table>
                    </div>
                </div>

                <!-- Kolom Kanan: Distribusi Probabilitas Model -->
                <div>
                    <div class="section-box">
                        <div class="box-title">📊 Probabilitas per Kelas Obesitas</div>
                        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1.5rem;">
                            Berikut adalah seberapa yakin model AI memprediksi ke dalam 7 kategori kelas obesitas yang berbeda:
                        </p>

                        <?php 
                        // Urutkan probabilitas dari yang terbesar
                        $probs = $result['probabilities'];
                        arsort($probs);
                        
                        foreach ($probs as $kelas => $persen): 
                            // Beri warna khusus jika persennya di atas 0
                            $barColor = ($persen == $result['prediction']['confidence_percent']) ? "linear-gradient(90deg, #0F766E, #0284C7)" : "#94A3B8";
                            $textWeight = ($persen == $result['prediction']['confidence_percent']) ? "800" : "600";
                            $textColor = ($persen == $result['prediction']['confidence_percent']) ? "#0F172A" : "#64748B";
                        ?>
                        <div class="prob-bar-container">
                            <div class="prob-label">
                                <span style="color: <?php echo $textColor; ?>; font-weight: <?php echo $textWeight; ?>;"><?php echo $kelas; ?></span>
                                <span style="color: <?php echo $textColor; ?>; font-weight: <?php echo $textWeight; ?>;"><?php echo $persen; ?>%</span>
                            </div>
                            <div class="prob-track">
                                <div class="prob-fill" style="width: <?php echo $persen; ?>%; background: <?php echo $barColor; ?>;"></div>
                            </div>
                        </div>
                        <?php endforeach; ?>
                    </div>
                </div>
            </div>

            <button class="btn-submit" style="background: #1E293B; margin-top: 1rem;" onclick="window.location.href='index.php'">
                🔄 Analisis Prediksi Data Baru
            </button>
        </div>
    <?php else: ?>

    <form method="POST" action="">
        <div class="form-grid">
            <div class="form-group">
                <label>Jenis Kelamin</label>
                <select name="gender" required>
                    <option value="Laki-laki">Laki-laki</option>
                    <option value="Perempuan">Perempuan</option>
                </select>
                <div class="helper-text">Jenis kelamin biologis</div>
            </div>
            <div class="form-group">
                <label>Usia (Tahun)</label>
                <input type="number" step="1" name="age" value="22" min="14" max="65" required>
                <div class="helper-text">Batas: 14 - 65 tahun</div>
            </div>
            <div class="form-group">
                <label>Tinggi Badan (m)</label>
                <input type="number" step="0.01" name="height" value="1.70" min="1.40" max="2.10" required>
                <div class="helper-text">Batas: 1.40 - 2.10 m</div>
            </div>
            <div class="form-group">
                <label>Berat Badan (kg)</label>
                <input type="number" step="0.1" name="weight" value="70.0" min="30.0" max="300.0" required>
                <div class="helper-text">Batas: 30.0 - 300.0 kg</div>
            </div>

            <div class="form-group">
                <label>Riwayat Overweight Keluarga</label>
                <select name="family_history_with_overweight" required>
                    <option value="Ya">Ya</option>
                    <option value="Tidak" selected>Tidak</option>
                </select>
                <div class="helper-text">Anggota keluarga gemuk?</div>
            </div>
            <div class="form-group">
                <label>Sering Makan Tinggi Kalori?</label>
                <select name="favc" required>
                    <option value="Ya" selected>Ya</option>
                    <option value="Tidak">Tidak</option>
                </select>
                <div class="helper-text">Gorengan, fast food, dll</div>
            </div>
            <div class="form-group">
                <label>Frekuensi Sayuran (1-3)</label>
                <input type="number" step="0.5" name="fcvc" value="2.0" min="1.0" max="3.0" required>
                <div class="helper-text">1=Jarang, 2=Kadang, 3=Sering</div>
            </div>
            <div class="form-group">
                <label>Jumlah Makan Utama (1-4)</label>
                <input type="number" step="1" name="ncp" value="3.0" min="1.0" max="4.0" required>
                <div class="helper-text">Berapa kali makan sehari?</div>
            </div>

            <div class="form-group">
                <label>Kebiasaan Ngemil</label>
                <select name="caec" required>
                    <option value="Tidak pernah">Tidak pernah</option>
                    <option value="Kadang-kadang" selected>Kadang-kadang</option>
                    <option value="Sering">Sering</option>
                    <option value="Selalu">Selalu</option>
                </select>
                <div class="helper-text">Camilan di luar jam makan</div>
            </div>
            <div class="form-group">
                <label>Konsumsi Alkohol</label>
                <select name="calc" required>
                    <option value="Tidak pernah" selected>Tidak pernah</option>
                    <option value="Kadang-kadang">Kadang-kadang</option>
                    <option value="Sering">Sering</option>
                    <option value="Selalu">Selalu</option>
                </select>
                <div class="helper-text">Seberapa sering minum alkohol?</div>
            </div>
            <div class="form-group">
                <label>Konsumsi Air (1-3)</label>
                <input type="number" step="0.5" name="ch2o" value="2.0" min="1.0" max="3.0" required>
                <div class="helper-text">1= <1L, 2= 1-2L, 3= >2L</div>
            </div>
            <div class="form-group">
                <label>Merokok?</label>
                <select name="smoke" required>
                    <option value="Ya">Ya</option>
                    <option value="Tidak" selected>Tidak</option>
                </select>
                <div class="helper-text">Perokok aktif?</div>
            </div>

            <div class="form-group">
                <label>Pantau Kalori Harian?</label>
                <select name="scc" required>
                    <option value="Ya">Ya</option>
                    <option value="Tidak" selected>Tidak</option>
                </select>
                <div class="helper-text">Sering menghitung kalori?</div>
            </div>
            <div class="form-group">
                <label>Frekuensi Olahraga (0-3)</label>
                <input type="number" step="0.5" name="faf" value="1.0" min="0.0" max="3.0" required>
                <div class="helper-text">0=Tidak pernah, 3=Sering</div>
            </div>
            <div class="form-group">
                <label>Waktu Layar/Gadget (0-2)</label>
                <input type="number" step="0.5" name="tue" value="1.0" min="0.0" max="2.0" required>
                <div class="helper-text">0= <3 jam, 2= >5 jam</div>
            </div>
            <div class="form-group">
                <label>Transportasi</label>
                <select name="mtrans" required>
                    <option value="Transportasi Umum">Transportasi Umum</option>
                    <option value="Mobil">Mobil</option>
                    <option value="Motor" selected>Motor</option>
                    <option value="Sepeda">Sepeda</option>
                    <option value="Jalan Kaki">Jalan Kaki</option>
                </select>
                <div class="helper-text">Transportasi yang sering dipakai</div>
            </div>
        </div>

        <button type="submit" class="btn-submit">Mulai Analisis Prediksi ✨</button>
    </form>

    <?php endif; ?>
</div>

</body>
</html>
