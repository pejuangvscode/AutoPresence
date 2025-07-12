import cv2
import os
import numpy as np
import csv
from datetime import datetime
import pickle

# folder db wajah
DATABASE_DIR = "db_wajah_fixed"
DATABASE_CSV = "db_presensi_fixed.csv"

# buat db
os.makedirs(DATABASE_DIR, exist_ok=True)

class FixedAttendanceSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # db variables
        self.known_encodings = []
        self.known_names = []
        self.face_id_counter = 0
        self.trained = False
        
        # ngeload data yang sudah ada
        self.load_known_faces()
        
        print(f"Sistem dimuat dengan {len(self.known_names)} wajah terdaftar")
    
    def load_known_faces(self):
        try:
            # Load metadata
            metadata_file = os.path.join(DATABASE_DIR, "metadata.pkl")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_names = data.get('names', [])
                    self.face_id_counter = data.get('counter', 0)
            
            # Load face data untuk training
            faces = []
            labels = []
            
            for i, name in enumerate(self.known_names):
                face_file = os.path.join(DATABASE_DIR, f"{name}_face.npy")
                if os.path.exists(face_file):
                    face_data = np.load(face_file)
                    faces.append(face_data)
                    labels.append(i)
            
            # Train recognizer jika ada data
            if faces and labels:
                self.recognizer.train(faces, np.array(labels))
                self.trained = True
                print(f"Recognizer trained dengan {len(faces)} wajah")
            
        except Exception as e:
            print(f"Error loading database: {e}")
            self.known_names = []
            self.face_id_counter = 0
    
    def save_metadata(self):
        """Simpan metadata database"""
        try:
            metadata_file = os.path.join(DATABASE_DIR, "metadata.pkl")
            data = {
                'names': self.known_names,
                'counter': self.face_id_counter
            }
            with open(metadata_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def add_new_face(self, face_gray, face_color, frame):
        """Menambahkan wajah baru ke database dengan normalisasi"""
        try:
            # Generate nama baru
            self.face_id_counter += 1
            name = f"person_{self.face_id_counter}"
            
            # Normalisasi wajah sebelum disimpan
            face_normalized = self.normalize_face(face_gray)
            
            # Simpan face encoding yang sudah dinormalisasi
            face_file = os.path.join(DATABASE_DIR, f"{name}_face.npy")
            np.save(face_file, face_normalized)
            
            # Simpan gambar wajah asli (tidak dinormalisasi)
            img_file = os.path.join(DATABASE_DIR, f"{name}.jpg")
            cv2.imwrite(img_file, face_color)
            
            # Update database
            self.known_names.append(name)
            self.save_metadata()
            
            # Retrain recognizer
            self.retrain_recognizer()
            
            print(f"Wajah baru disimpan: {name}")
            return name
            
        except Exception as e:
            print(f"Error menambah wajah baru: {e}")
            return None
    
    def retrain_recognizer(self):
        """Retrain recognizer dengan data terbaru"""
        try:
            faces = []
            labels = []
            
            for i, name in enumerate(self.known_names):
                face_file = os.path.join(DATABASE_DIR, f"{name}_face.npy")
                if os.path.exists(face_file):
                    face_data = np.load(face_file)
                    faces.append(face_data)
                    labels.append(i)
            
            if faces and labels:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.train(faces, np.array(labels))
                self.trained = True
                print(f"Recognizer diupdate dengan {len(faces)} wajah")
            
        except Exception as e:
            print(f"Error retraining: {e}")
    
    def recognize_face(self, face_gray):
        """Kenali wajah"""
        if not self.trained or not self.known_names:
            return "Unknown", 0
        
        try:
            face_resized = cv2.resize(face_gray, (100, 100))
            label, confidence = self.recognizer.predict(face_resized)
            
            # Confidence threshold (lower is better for LBPH)
            if confidence < 80:  # Lebih ketat untuk menghindari false positive
                if 0 <= label < len(self.known_names):
                    return self.known_names[label], confidence
            
            return "Unknown", confidence
            
        except Exception as e:
            print(f"Error recognizing: {e}")
            return "Unknown", 0
    
    def normalize_face(self, face_gray):
        """Normalisasi wajah untuk mengurangi pengaruh rotasi dan pencahayaan"""
        try:
            # Resize ke ukuran standar
            face_resized = cv2.resize(face_gray, (100, 100))
            
            # Histogram equalization untuk normalisasi pencahayaan
            face_normalized = cv2.equalizeHist(face_resized)
            
            # Gaussian blur untuk mengurangi noise
            face_smoothed = cv2.GaussianBlur(face_normalized, (3, 3), 0)
            
            return face_smoothed
            
        except Exception as e:
            print(f"Error normalizing face: {e}")
            return cv2.resize(face_gray, (100, 100))
    
    def get_face_rotations(self, face_gray):
        """Generate multiple rotasi wajah untuk perbandingan yang lebih robust"""
        rotations = [face_gray]  # Original
        
        try:
            h, w = face_gray.shape
            center = (w // 2, h // 2)
            
            # Generate rotasi -15, -10, -5, 5, 10, 15 derajat
            angles = [-15, -10, -5, 5, 10, 15]
            
            for angle in angles:
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(face_gray, rotation_matrix, (w, h))
                rotations.append(rotated)
            
            return rotations
            
        except Exception as e:
            print(f"Error generating rotations: {e}")
            return [face_gray]
    
    def is_similar_face(self, new_face_gray):
        """Cek apakah wajah baru mirip dengan yang sudah ada di database (rotation-invariant)"""
        if not self.known_names:
            return False, None
        
        try:
            # Normalisasi wajah baru
            new_face_normalized = self.normalize_face(new_face_gray)
            
            # Generate rotasi wajah baru
            new_face_rotations = self.get_face_rotations(new_face_normalized)
            
            # Bandingkan dengan semua wajah yang ada
            for name in self.known_names:
                face_file = os.path.join(DATABASE_DIR, f"{name}_face.npy")
                if os.path.exists(face_file):
                    existing_face = np.load(face_file)
                    
                    max_similarity = 0
                    
                    # Test semua rotasi wajah baru dengan wajah yang ada
                    for rotated_face in new_face_rotations:
                        try:
                            # Template matching
                            correlation = cv2.matchTemplate(rotated_face, existing_face, cv2.TM_CCOEFF_NORMED)
                            similarity = np.max(correlation)
                            max_similarity = max(max_similarity, similarity)
                            
                            # Structural Similarity Index (SSIM) sebagai backup
                            # Convert to float untuk SSIM calculation
                            face1_float = rotated_face.astype(np.float64)
                            face2_float = existing_face.astype(np.float64)
                            
                            # Simple SSIM calculation
                            mean1 = np.mean(face1_float)
                            mean2 = np.mean(face2_float)
                            std1 = np.std(face1_float)
                            std2 = np.std(face2_float)
                            
                            if std1 > 0 and std2 > 0:
                                covariance = np.mean((face1_float - mean1) * (face2_float - mean2))
                                ssim = (2 * mean1 * mean2 + 1) * (2 * covariance + 1) / ((mean1**2 + mean2**2 + 1) * (std1**2 + std2**2 + 1))
                                max_similarity = max(max_similarity, ssim)
                            
                        except Exception as e:
                            continue
                    
                    # Threshold yang lebih ketat untuk similarity
                    if max_similarity > 0.65:  # Turunkan dari 0.7 ke 0.65
                        print(f"Wajah mirip dengan {name} (similarity: {max_similarity:.3f})")
                        return True, name
            
            return False, None
            
        except Exception as e:
            print(f"Error checking similarity: {e}")
            return False, None
    
    def add_attendance(self, name):
        """Menambahkan presensi ke CSV (hanya sekali per hari)"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M:%S")
        
        try:
            # Cek apakah sudah absen hari ini
            if os.path.exists(DATABASE_CSV):
                with open(DATABASE_CSV, mode='r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2 and row[0] == name and row[1] == date_str:
                            return False  # Sudah absen
            
            # Tambah presensi baru
            with open(DATABASE_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([name, date_str, time_str])
            
            return True
            
        except Exception as e:
            print(f"Error adding attendance: {e}")
            return False

def main():
    print("Memulai Presensi Otomatis")
    system = FixedAttendanceSystem()
    
    # Setup kamera dengan multiple backends
    video_capture = None
    camera_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    print("Mencari kamera...")
    for backend in camera_backends:
        try:
            video_capture = cv2.VideoCapture(0, backend)
            if video_capture.isOpened():
                ret, test_frame = video_capture.read()
                if ret:
                    print(f"Kamera ditemukan dengan backend {backend}")
                    break
                else:
                    video_capture.release()
                    video_capture = None
        except:
            if video_capture:
                video_capture.release()
            video_capture = None
    
    if not video_capture or not video_capture.isOpened():
        print("Tidak dapat mengakses kamera!")
        print("Troubleshooting:")
        print("   - Pastikan kamera terhubung dan tidak digunakan aplikasi lain")
        print("   - Cek Windows Camera privacy settings")
        print("   - Restart aplikasi atau komputer")
        print("   - Coba tutup Zoom, Teams, atau aplikasi video lain")
        return
    
    print("Kamera aktif! Tekan 'q' untuk keluar")
    print("Sistem akan otomatis mendeteksi dan menyimpan wajah baru")
    print("Wajah akan disimpan setelah terdeteksi stabil 8 frame")
    print("Sistem mencegah duplikasi wajah yang sama")
    
    # Untuk tracking deteksi stabil
    detection_counter = {}
    min_detection_count = 8  # Increase untuk deteksi lebih stabil
    last_attendance = {}
    attendance_cooldown = 10  # seconds
    
    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error membaca frame kamera")
                break
            
            # Flip frame untuk efek mirror
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah dengan parameter yang lebih stabil
            faces = system.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,      # Lebih kecil untuk deteksi lebih halus
                minNeighbors=8,        # Lebih tinggi untuk mengurangi false positive
                minSize=(60, 60),      # Ukuran minimum lebih besar
                maxSize=(300, 300),    # Batasi ukuran maksimum
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Proses setiap wajah yang terdeteksi
            for (x, y, w, h) in faces:
                # Extract face regions dengan padding
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_gray = gray[y1:y2, x1:x2]
                face_color = frame[y1:y2, x1:x2]
                
                # Skip jika wajah terlalu kecil
                if face_gray.shape[0] < 50 or face_gray.shape[1] < 50:
                    continue
                
                # Coba kenali wajah
                name, confidence = system.recognize_face(face_gray)
                
                if name == "Unknown":
                    # Cek apakah wajah ini mirip dengan yang sudah ada
                    is_similar, similar_name = system.is_similar_face(face_gray)
                    
                    if is_similar:
                        # Wajah mirip dengan yang sudah ada
                        name = f"{similar_name} (Similar)"
                        color = (0, 255, 255)  # Kuning untuk wajah yang mirip
                    else:
                        # Wajah benar-benar baru - hitung deteksi
                        # Gunakan center position untuk tracking yang lebih stabil
                        center_x = x + w // 2
                        center_y = y + h // 2
                        face_key = f"face_{center_x//50}_{center_y//50}"  # Grid-based tracking
                        
                        if face_key not in detection_counter:
                            detection_counter[face_key] = 0
                        detection_counter[face_key] += 1
                        
                        if detection_counter[face_key] >= min_detection_count:
                            # Deteksi stabil - simpan wajah baru
                            print(f"Menyimpan wajah baru...")
                            new_name = system.add_new_face(face_gray, face_color, frame)
                            if new_name:
                                name = new_name
                                # Reset counter
                                detection_counter = {}
                                print(f"Wajah baru berhasil disimpan: {name}")
                                
                                # Auto attendance untuk wajah baru
                                if system.add_attendance(name):
                                    print(f"Auto-presensi: {name}")
                            else:
                                name = "Error Saving"
                        else:
                            name = f"New Face {detection_counter[face_key]}/{min_detection_count}"
                        
                        color = (0, 0, 255)
                else:
                    # Wajah dikenal
                    color = (0, 255, 0)
                    
                    # Handle attendance
                    current_time = datetime.now()
                    if name in last_attendance:
                        time_diff = (current_time - last_attendance[name]).seconds
                        if time_diff >= attendance_cooldown:
                            if system.add_attendance(name):
                                print(f"Presensi: {name} - {current_time.strftime('%H:%M:%S')}")
                                last_attendance[name] = current_time
                        else:
                            remaining = attendance_cooldown - time_diff
                            name += f" (Wait {remaining}s)"
                    else:
                        if system.add_attendance(name):
                            print(f"Presensi: {name} - {current_time.strftime('%H:%M:%S')}")
                            last_attendance[name] = current_time
                
                # Gambar kotak di wajah dengan koordinat yang sudah dipadding
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Sesuaikan ukuran font dengan ukuran wajah
                font_scale = max(0.4, min(0.8, w/150))
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
                
                if name != "Unknown" and "New Face" not in name and "Error" not in name and "Wait" not in name and "Similar" not in name:
                    cv2.putText(frame, f"Conf: {confidence:.0f}", (x1, y2+25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Tambahkan info deteksi
                if "New Face" in name:
                    cv2.putText(frame, "Hold steady...", (x1, y2+45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Clean detection counter dengan cara yang lebih pintar
            if len(faces) == 0:
                # Kurangi semua counter jika tidak ada wajah terdeteksi
                for key in list(detection_counter.keys()):
                    detection_counter[key] -= 1
                    if detection_counter[key] <= 0:
                        del detection_counter[key]
            else:
                # Hapus counter yang tidak aktif (tidak ada wajah di area tersebut)
                active_keys = []
                for (x, y, w, h) in faces:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    face_key = f"face_{center_x//50}_{center_y//50}"
                    active_keys.append(face_key)
                
                # Hapus key yang tidak aktif
                for key in list(detection_counter.keys()):
                    if key not in active_keys:
                        detection_counter[key] -= 1
                        if detection_counter[key] <= 0:
                            del detection_counter[key]
            
            # Tambahkan info di frame
            cv2.putText(frame, "PRESENSI OTOMATIS", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Database: {len(system.known_names)} orang", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Tekan 'q' untuk keluar", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Presensi Otomatis', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset detection counter jika user tekan 'r'
                detection_counter = {}
                print("Detection counter direset")
                
    except KeyboardInterrupt:
        print("\nSistem dihentikan oleh user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if video_capture:
            video_capture.release()
        cv2.destroyAllWindows()
        print("Sistem presensi ditutup")
        print(f"Total wajah di database: {len(system.known_names)}")

if __name__ == "__main__":
    main()
