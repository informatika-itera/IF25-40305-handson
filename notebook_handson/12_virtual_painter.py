"""
12_virtual_painter.py — Melukis di udara: telunjuk jadi kuas (webcam live)
IF25-40305 Sistem Teknologi Multimedia · Modul 12 · ITERA

Cara menjalankan (di laptop sendiri, BUKAN di Colab, karena butuh webcam & jendela):
    python 12_virtual_painter.py                              # pakai webcam
    python 12_virtual_painter.py video/hitung_jari.webm       # atau pakai file video
    python 12_virtual_painter.py video.mp4 --simpan hasil.mp4 # tanpa jendela, hasil disimpan ke file

Tombol:  q = keluar   c = hapus kanvas   s = simpan lukisan (PNG)   t = titik tangan on/off

Gestur:
    hanya TELUNJUK terangkat        -> MENGGAMBAR (ujung telunjuk = kuas)
    TELUNJUK + TENGAH terangkat      -> MODE PILIH (kuas diangkat; sentuh kotak warna di atas layar)
    KEPAL (0 jari) selama 1 detik    -> HAPUS semua coretan

Cara kerjanya (setiap frame):
  1. MediaPipe Hand Landmarker mencari 21 titik tangan.
  2. Tiap jari dicek: terangkat kalau ujungnya lebih jauh dari pergelangan (titik 0) daripada sendi tengahnya.
     Jempol: terangkat kalau ujungnya (4) lebih jauh dari pangkal kelingking (17) daripada sendinya (3).
  3. Posisi ujung telunjuk (titik 8) dihaluskan (rata-rata dengan posisi sebelumnya) supaya garis tidak bergetar.
  4. Garis digambar di KANVAS terpisah, lalu kanvas ditempel ke frame kamera.

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

# --- 2. Model tangan (±7,5 MB, diunduh otomatis sekali) ----------------------
MODEL = os.path.join(FOLDER, "model", "hand_landmarker.task")
URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
if not os.path.exists(MODEL):
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print("Mengunduh model tangan (sekali saja)...")
    urllib.request.urlretrieve(URL, MODEL)
opsi = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=MODEL),
                                    running_mode=vision.RunningMode.VIDEO, num_hands=1,
                                    min_hand_detection_confidence=0.6, min_tracking_confidence=0.5)
landmarker = vision.HandLandmarker.create_from_options(opsi)

SAMBUNG = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
           (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
UJUNG, SENDI = [8, 12, 16, 20], [6, 10, 14, 18]

# Palet di bagian atas layar: (nama, warna BGR). Warna hitam = penghapus.
PALET = [("Merah", (46, 16, 200)), ("Emas", (23, 160, 212)), ("Hijau", (107, 158, 46)),
         ("Biru", (200, 110, 30)), ("Putih", (255, 255, 255)), ("Hapus", (0, 0, 0))]
TINGGI_PALET = 70
HALUS = 0.5               # 0..1: makin kecil makin halus (tapi makin "terlambat")
TEBAL_KUAS, TEBAL_HAPUS = 8, 50


def jarak(t, i, j):
    return float(np.linalg.norm(t[i] - t[j]))


def jari_terangkat(t):
    """[jempol, telunjuk, tengah, manis, kelingking] -> True kalau terangkat."""
    jempol = jarak(t, 4, 17) > jarak(t, 3, 17)
    return [jempol] + [jarak(t, u, 0) > jarak(t, s, 0) for u, s in zip(UJUNG, SENDI)]


def tulis(img, teks, xy, skala=0.8, warna=(255, 255, 255)):
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, 2, cv2.LINE_AA)


def gambar_palet(img, terpilih):
    w = img.shape[1]
    lebar = w // len(PALET)
    for k, (nama, warna) in enumerate(PALET):
        x0 = k * lebar
        cv2.rectangle(img, (x0 + 4, 4), (x0 + lebar - 4, TINGGI_PALET - 4), warna, -1)
        if k == terpilih:
            cv2.rectangle(img, (x0 + 2, 2), (x0 + lebar - 2, TINGGI_PALET - 2), (255, 255, 255), 4)
        tulis(img, nama, (x0 + 12, TINGGI_PALET - 22), 0.6)


# --- 3. Lingkaran utama ---------------------------------------------------------
kanvas = None
terpilih = 0
sebelum = None            # posisi kuas frame sebelumnya (None = kuas sedang diangkat)
kepal_sejak = None
tampil_titik = True
penulis = None
nomor, t0 = 0, time.time()
print("Tombol: q keluar | c hapus kanvas | s simpan lukisan | t titik tangan")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if WEBCAM:
        frame = cv2.flip(frame, 1)                         # seperti cermin: tangan kanan tampak di kanan
    h, w = frame.shape[:2]
    if kanvas is None:
        kanvas = np.zeros_like(frame)
    waktu_ms = int((time.time() - t0) * 1000) if WEBCAM else int(nomor * 1000 / FPS)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hasil = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), waktu_ms)

    status = "tidak ada tangan"
    if hasil.hand_landmarks:
        t = np.array([[p.x * w, p.y * h] for p in hasil.hand_landmarks[0]], np.float32)
        jari = jari_terangkat(t)
        ujung = t[8] if sebelum is None else HALUS * t[8] + (1 - HALUS) * sebelum
        if tampil_titik:
            for a, b in SAMBUNG:
                cv2.line(frame, tuple(np.int32(t[a])), tuple(np.int32(t[b])), (220, 220, 220), 2, cv2.LINE_AA)

        nama, warna = PALET[terpilih]
        if jari[1] and not (jari[2] or jari[3] or jari[4]):            # hanya telunjuk -> gambar
            status = "MENGGAMBAR" if nama != "Hapus" else "MENGHAPUS"
            if sebelum is not None:
                tebal = TEBAL_HAPUS if nama == "Hapus" else TEBAL_KUAS
                cv2.line(kanvas, tuple(np.int32(sebelum)), tuple(np.int32(ujung)), warna, tebal, cv2.LINE_AA)
            sebelum = ujung
            kepal_sejak = None
        else:
            sebelum = None                                              # kuas diangkat
            if jari[1] and jari[2] and not (jari[3] or jari[4]):        # telunjuk + tengah -> pilih warna
                status = "MODE PILIH"
                if t[8, 1] < TINGGI_PALET:
                    terpilih = int(np.clip(t[8, 0] // (w // len(PALET)), 0, len(PALET) - 1))
                kepal_sejak = None
            elif not any(jari):                                         # kepal -> tahan 1 detik untuk hapus
                kepal_sejak = kepal_sejak or time.time()
                sisa = 1.0 - (time.time() - kepal_sejak)
                status = f"KEPAL: hapus dalam {max(sisa, 0):.1f} s"
                if sisa <= 0:
                    kanvas[:] = 0
                    kepal_sejak = None
            else:
                status = f"{sum(jari)} jari"
                kepal_sejak = None
        lingkar = (255, 255, 255) if nama == "Hapus" else warna
        cv2.circle(frame, tuple(np.int32(ujung)), 12 if status != "MENGGAMBAR" else 8, lingkar, -1, cv2.LINE_AA)
    else:
        sebelum, kepal_sejak = None, None

    # tempel kanvas: piksel kanvas yang tidak hitam menimpa frame kamera
    topeng = kanvas.any(axis=2)
    frame[topeng] = kanvas[topeng]
    gambar_palet(frame, terpilih)
    tulis(frame, status, (15, h - 20), 0.9, (74, 195, 232))

    if SIMPAN:
        if penulis is None:
            penulis = cv2.VideoWriter(SIMPAN, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        penulis.write(frame)
    else:
        cv2.imshow("Virtual Painter STM - tekan q untuk keluar", frame)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        if tombol == ord("c"):
            kanvas[:] = 0
        if tombol == ord("t"):
            tampil_titik = not tampil_titik
        if tombol == ord("s"):
            nama_file = os.path.join(FOLDER, "output", f"lukisan_{time.strftime('%H%M%S')}.png")
            os.makedirs(os.path.dirname(nama_file), exist_ok=True)
            cv2.imwrite(nama_file, kanvas)
            print("Lukisan disimpan:", nama_file)
    nomor += 1

cap.release()
if penulis is not None:
    penulis.release()
    print(f"Hasil disimpan: {SIMPAN} ({nomor} frame)")
if not SIMPAN:
    cv2.destroyAllWindows()
