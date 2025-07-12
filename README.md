# Sistem Presensi Otomatis

Sistem presensi menggunakan face recognition yang dapat mendeteksi dan menyimpan wajah baru secara otomatis.

## Tech Stack

- OpenCV: Face detection menggunakan Haar Cascade
- LBPH Face Recognizer: Untuk pengenalan wajah
- Template Matching: Untuk perbandingan similarity
- Python: Bahasa pemrograman utama

## Cara Menggunakan

### 1. Setup
```bash
# Aktifkan virtual environment
.venv\Scripts\activate

### Jalankan
python fixed_attendance_system.py
```

### 2. Penggunaan
- Posisikan wajah di depan kamera
- Tunggu deteksi stabil (8 frame)
- Wajah baru akan disimpan otomatis sebagai person_X
- Presensi tercatat di file CSV
- Tekan 'q' untuk keluar

## Alur Sistem

1. **Inisialisasi**: Load database wajah yang sudah ada
2. **Deteksi Wajah**: Menggunakan Haar Cascade Classifier
3. **Normalisasi**: Histogram equalization dan Gaussian blur
4. **Pengecekan Duplikasi**: Bandingkan dengan db menggunakan multiple rotasi
5. **Recognition**: Jika wajah dikenal, tampilkan nama
6. **Auto-Save**: Jika wajah baru dan stabil, simpan ke database
7. **Attendance**: Catat presensi dengan cooldown 10 detik

## File Output

- `db_wajah_fixed/person_X.jpg`: Foto wajah
- `db_wajah_fixed/person_X_face.npy`: Data encoding wajah
- `db_presensi_fixed.csv`: Log presensi
- `db_wajah_fixed/metadata.pkl`: Metadata database

## Konfigurasi

Parameter utama di `fixed_attendance_system.py`:
- min_detection_count = 8 (frame minimum untuk save)
- attendance_cooldown = 10 (cooldown presensi dalam detik)
- similarity_threshold = 0.65 (threshold similarity 0-1)

---
