"""
11_filter_wajah.py — Filter wajah buatan sendiri: kacamata hitam + penghitung kedip + QR jadi panggung AR
IF25-40305 Sistem Teknologi Multimedia · Modul 11 · ITERA

Cara menjalankan (di laptop sendiri, BUKAN di Colab, karena butuh webcam & jendela):
    python 11_filter_wajah.py                      # pakai webcam
    python 11_filter_wajah.py video.mp4            # atau pakai file video
    python 11_filter_wajah.py video.mp4 --simpan hasil.mp4   # tanpa jendela, hasil disimpan ke file

Tombol:  q = keluar   k = kacamata on/off   m = titik wajah on/off   r = nolkan penghitung kedip

Cara kerjanya (setiap frame):
  1. MediaPipe Face Landmarker mencari 478 titik wajah.
  2. Dari 6 titik di tiap mata dihitung EAR (tinggi mata / lebar mata).
     EAR < 0,20 selama >= 2 frame lalu naik lagi  -> dihitung 1 kedip.
     Mata tertutup > 1 detik                       -> muncul peringatan "MENGANTUK?".
  3. Kacamata (gambar PNG tembus pandang) diputar & diperbesar mengikuti dua mata, lalu ditempel.
  4. Kalau ada QR code di depan kamera: temukan 4 sudutnya, tempel gambar kucing di atasnya (AR sederhana).

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

# --- 2. Model titik wajah (±3,7 MB, diunduh otomatis sekali) -----------------
MODEL = os.path.join(FOLDER, "model", "face_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
if not os.path.exists(MODEL):
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print("Mengunduh model titik wajah (sekali saja)...")
    urllib.request.urlretrieve(URL, MODEL)
opsi = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                    running_mode=vision.RunningMode.VIDEO, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(opsi)

# --- 3. Gambar yang akan ditempel ---------------------------------------------
KACAMATA = cv2.imread(os.path.join(FOLDER, "image", "kacamata.png"), cv2.IMREAD_UNCHANGED)
JANGKAR = np.float32([[150, 125], [450, 125]])          # pusat lensa kiri & kanan di kacamata.png
KUCING = cv2.imread(os.path.join(FOLDER, "image", "chelsea.png"))
if KUCING is None:                                       # cadangan kalau gambar tidak ada
    KUCING = np.full((300, 300, 3), (60, 160, 230), np.uint8)
s = min(KUCING.shape[:2])
KUCING = cv2.cvtColor(KUCING[:s, (KUCING.shape[1] - s) // 2:(KUCING.shape[1] + s) // 2], cv2.COLOR_BGR2BGRA)
qr_detektor = cv2.QRCodeDetector()

MATA_KANAN = [33, 160, 158, 133, 153, 144]               # mata kanan pemilik wajah (kiri di layar)
MATA_KIRI = [362, 385, 387, 263, 373, 380]               # mata kiri pemilik wajah (kanan di layar)
AMBANG_EAR = 0.20                                        # di bawah ini = mata dianggap tertutup


def ear(titik, idx):
    """Eye Aspect Ratio: rata-rata tinggi mata dibagi lebar mata."""
    p1, p2, p3, p4, p5, p6 = [titik[i] for i in idx]
    tinggi = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / 2
    return tinggi / np.linalg.norm(p1 - p4)


def tempel_rgba(latar, gambar_rgba, M):
    """Tempel gambar RGBA ke latar memakai matriks M (2x3 affine atau 3x3 perspektif)."""
    h, w = latar.shape[:2]
    if M.shape == (2, 3):
        warp = cv2.warpAffine(gambar_rgba, M, (w, h), borderValue=(0, 0, 0, 0))
    else:
        warp = cv2.warpPerspective(gambar_rgba, M, (w, h), borderValue=(0, 0, 0, 0))
    alpha = warp[..., 3:4].astype(np.float32) / 255.0
    return (warp[..., :3] * alpha + latar * (1 - alpha)).astype(np.uint8)


def pasang_kacamata(frame, titik):
    """Geser, putar, dan perbesar kacamata supaya dua lensanya jatuh tepat di dua mata."""
    mata_a = titik[MATA_KANAN].mean(axis=0)
    mata_b = titik[MATA_KIRI].mean(axis=0)
    if KACAMATA is None:                                  # cadangan: dua lingkaran hitam
        r = int(np.linalg.norm(mata_b - mata_a) * 0.38)
        for p in (mata_a, mata_b):
            cv2.circle(frame, tuple(np.int32(p)), r, (20, 20, 20), -1, cv2.LINE_AA)
        return frame
    M, _ = cv2.estimateAffinePartial2D(JANGKAR, np.float32([mata_a, mata_b]))
    return tempel_rgba(frame, KACAMATA, M)


def tulis(img, teks, xy, skala=0.8, warna=(255, 255, 255)):
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, 2, cv2.LINE_AA)


# --- 4. Lingkaran utama ---------------------------------------------------------
pakai_kacamata, tampil_titik = True, False
kedip, berturut = 0, 0
penulis = None
nomor, t0 = 0, time.time()
print("Tombol: q keluar | k kacamata | m titik wajah | r nolkan penghitung")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if WEBCAM:
        frame = cv2.flip(frame, 1)                         # seperti cermin
    h, w = frame.shape[:2]
    # MediaPipe butuh waktu (milidetik) yang terus naik
    waktu_ms = int((time.time() - t0) * 1000) if WEBCAM else int(nomor * 1000 / FPS)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), waktu_ms)

    if hasil.face_landmarks:
        titik = np.array([[p.x * w, p.y * h] for p in hasil.face_landmarks[0]], np.float32)
        nilai = (ear(titik, MATA_KANAN) + ear(titik, MATA_KIRI)) / 2
        if nilai < AMBANG_EAR:
            berturut += 1
        else:
            if berturut >= 2:                              # tertutup minimal 2 frame = kedip sungguhan
                kedip += 1
            berturut = 0
        if tampil_titik:
            for p in titik:
                cv2.circle(frame, tuple(np.int32(p)), 1, (74, 195, 232), -1)
        if pakai_kacamata:
            frame = pasang_kacamata(frame, titik)
        tulis(frame, f"Kedip: {kedip}", (15, 38), 1.0, (74, 195, 232))
        tulis(frame, f"EAR {nilai:.2f}", (15, 72), 0.7)
        if berturut > FPS * 1.0:                           # mata tertutup lebih dari 1 detik
            tulis(frame, "MENGANTUK?", (w // 2 - 120, h - 40), 1.4, (46, 16, 200))
    else:
        berturut = 0
        tulis(frame, "Wajah tidak terlihat", (15, 38), 0.8, (46, 16, 200))

    # QR code: kalau ketemu, tempel kucing di atasnya
    try:
        teks_qr, sudut, _ = qr_detektor.detectAndDecode(frame)
    except cv2.error:                                      # kadang "QR palsu" dengan 4 sudut berimpit
        teks_qr, sudut = "", None
    if sudut is not None and len(sudut.reshape(-1, 2)) == 4:
        sudut = sudut.reshape(4, 2).astype(np.float32)
        sisi = KUCING.shape[0]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [sisi, 0], [sisi, sisi], [0, sisi]]), sudut)
        frame = tempel_rgba(frame, KUCING, M)
        cv2.polylines(frame, [sudut.astype(np.int32)], True, (107, 158, 46), 3, cv2.LINE_AA)
        if teks_qr:
            tulis(frame, teks_qr[:40], (int(sudut[:, 0].min()), int(sudut[:, 1].min()) - 12), 0.6, (107, 158, 46))

    if SIMPAN:
        if penulis is None:
            penulis = cv2.VideoWriter(SIMPAN, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        penulis.write(frame)
    else:
        cv2.imshow("Filter Wajah STM - tekan q untuk keluar", frame)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        if tombol == ord("k"):
            pakai_kacamata = not pakai_kacamata
        if tombol == ord("m"):
            tampil_titik = not tampil_titik
        if tombol == ord("r"):
            kedip = 0
    nomor += 1

cap.release()
if penulis is not None:
    penulis.release()
    print(f"Hasil disimpan: {SIMPAN} ({nomor} frame, {kedip} kedip)")
if not SIMPAN:
    cv2.destroyAllWindows()
