"""
12_pose_sudut.py — Kerangka tubuh 33 titik + sudut sendi + penghitung repetisi (webcam live)
IF25-40305 Sistem Teknologi Multimedia · Modul 12 · ITERA

Cara menjalankan (di laptop sendiri, BUKAN di Colab, karena butuh webcam & jendela):
    python 12_pose_sudut.py                                  # pakai webcam
    python 12_pose_sudut.py video/latihan_squat.webm         # atau pakai file video
    python 12_pose_sudut.py video.mp4 --simpan hasil.mp4     # tanpa jendela, hasil disimpan ke file

Tombol:  q = keluar   m = ganti mode (siku / lutut)   r = nolkan penghitung

Cara kerjanya (setiap frame):
  1. MediaPipe Pose Landmarker mencari 33 titik tubuh.
  2. Sudut sendi dihitung dari tiga titik:
       mode SIKU  : bahu - siku - pergelangan tangan   (angkat beban / bicep curl)
       mode LUTUT : pinggul - lutut - pergelangan kaki (squat)
     Dipakai sisi tubuh yang paling jelas terlihat (visibility terbesar).
  3. Penghitung repetisi memakai DUA ambang:
       sudut turun di bawah AMBANG_BAWAH  -> posisi "ditekuk"
       sudut naik lagi di atas AMBANG_ATAS -> 1 repetisi selesai

Butuh: pip install opencv-contrib-python mediapipe numpy
"""
import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

FOLDER = os.path.dirname(os.path.abspath(__file__))

# --- 1. Sumber video: angka 0 = webcam bawaan laptop -------------------------
argumen = [a for a in sys.argv[1:] if not a.startswith("--")]
SIMPAN = sys.argv[sys.argv.index("--simpan") + 1] if "--simpan" in sys.argv else None
if SIMPAN in argumen:
    argumen.remove(SIMPAN)
sumber = argumen[0] if argumen else 0
WEBCAM = sumber == 0
cap = cv2.VideoCapture(sumber)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
if FPS < 1 or FPS > 120:
    FPS = 30

# --- 2. Model pose (±9 MB, diunduh otomatis sekali) --------------------------
MODEL = os.path.join(FOLDER, "model", "pose_landmarker_full.task")
URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
       "float16/1/pose_landmarker_full.task")
if not os.path.exists(MODEL):
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print("Mengunduh model pose (sekali saja)...")
    urllib.request.urlretrieve(URL, MODEL)
opsi = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                    running_mode=vision.RunningMode.VIDEO, num_poses=1)
landmarker = vision.PoseLandmarker.create_from_options(opsi)

SAMBUNG = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
           (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (0, 2), (0, 5)]

# Tiga titik per sendi: (kiri, kanan). Kiri/kanan = milik ORANGNYA.
MODE = {
    "SIKU": {"titik": [(11, 13, 15), (12, 14, 16)], "bawah": 60, "atas": 150, "nama": "Angkat beban"},
    "LUTUT": {"titik": [(23, 25, 27), (24, 26, 28)], "bawah": 110, "atas": 150, "nama": "Squat"},
}


def sudut(a, b, c):
    """Sudut di titik b (derajat): seberapa terbuka 'engsel' b."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def tulis(img, teks, xy, skala=0.8, warna=(255, 255, 255)):
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, 2, cv2.LINE_AA)


def busur(img, a, b, c, nilai, warna):
    """Gambar busur kecil di sendi b, dari arah b->a ke arah b->c (seperti busur derajat)."""
    r = 35
    sa = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0]))
    sc = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
    if (sc - sa) % 360 > 180:
        sa, sc = sc, sa
    awal, akhir = sa, sa + (sc - sa) % 360
    cv2.ellipse(img, tuple(np.int32(b)), (r, r), 0, awal, akhir, warna, 3, cv2.LINE_AA)
    tulis(img, f"{nilai:.0f}", (int(b[0]) + 12, int(b[1]) - 12), 0.8, warna)


# --- 3. Lingkaran utama ---------------------------------------------------------
nama_mode = "SIKU"
hitung, posisi = 0, "atas"
penulis = None
nomor, t0 = 0, time.time()
print("Tombol: q keluar | m ganti mode siku/lutut | r nolkan penghitung")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if WEBCAM:
        frame = cv2.flip(frame, 1)                         # seperti cermin
    h, w = frame.shape[:2]
    waktu_ms = int((time.time() - t0) * 1000) if WEBCAM else int(nomor * 1000 / FPS)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), waktu_ms)
    m = MODE[nama_mode]

    if hasil.pose_landmarks:
        T = np.array([[p.x * w, p.y * h] for p in hasil.pose_landmarks[0]], np.float32)
        V = np.array([p.visibility for p in hasil.pose_landmarks[0]])
        for a, b in SAMBUNG:
            if V[a] > 0.5 and V[b] > 0.5:
                cv2.line(frame, tuple(np.int32(T[a])), tuple(np.int32(T[b])), (74, 195, 232), 3, cv2.LINE_AA)
        for i in range(11, 29):
            if V[i] > 0.5:
                cv2.circle(frame, tuple(np.int32(T[i])), 5, (46, 16, 200), -1, cv2.LINE_AA)

        # pilih sisi tubuh yang paling terlihat
        tiga = max(m["titik"], key=lambda t: min(V[i] for i in t))
        if min(V[i] for i in tiga) > 0.5:
            a, b, c = (T[i] for i in tiga)
            nilai = sudut(a, b, c)
            busur(frame, a, b, c, nilai, (140, 140, 30))
            if posisi == "atas" and nilai < m["bawah"]:
                posisi = "bawah"
            elif posisi == "bawah" and nilai > m["atas"]:
                posisi = "atas"
                hitung += 1
            # bilah kecil: seberapa dekat ke ambang bawah
            isi = np.clip((180 - nilai) / (180 - m["bawah"]), 0, 1)
            cv2.rectangle(frame, (w - 50, 80), (w - 25, 280), (255, 255, 255), 2)
            cv2.rectangle(frame, (w - 50, int(280 - 200 * isi)), (w - 25, 280),
                          (107, 158, 46) if posisi == "bawah" else (74, 195, 232), -1)
        else:
            tulis(frame, "Sendi kurang terlihat", (15, 110), 0.7, (46, 16, 200))
    else:
        tulis(frame, "Tidak ada orang", (15, 110), 0.8, (46, 16, 200))

    tulis(frame, f"{m['nama']}: {hitung}", (15, 40), 1.1, (74, 195, 232))
    tulis(frame, f"mode {nama_mode}  ({m['bawah']}-{m['atas']} derajat)", (15, 75), 0.6)

    if SIMPAN:
        if penulis is None:
            penulis = cv2.VideoWriter(SIMPAN, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        penulis.write(frame)
    else:
        cv2.imshow("Pose & Sudut STM - tekan q untuk keluar", frame)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        if tombol == ord("m"):
            nama_mode = "LUTUT" if nama_mode == "SIKU" else "SIKU"
            hitung, posisi = 0, "atas"
        if tombol == ord("r"):
            hitung, posisi = 0, "atas"
    nomor += 1

cap.release()
if penulis is not None:
    penulis.release()
    print(f"Hasil disimpan: {SIMPAN} ({nomor} frame, {hitung} repetisi)")
if not SIMPAN:
    cv2.destroyAllWindows()
