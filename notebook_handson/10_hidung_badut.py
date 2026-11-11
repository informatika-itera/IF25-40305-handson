"""
10_hidung_badut.py — Hidung badut yang ikut bergerak bersama hidung Anda (webcam live)
IF25-40305 Sistem Teknologi Multimedia · Modul 10 · ITERA

Cara menjalankan (di laptop sendiri, BUKAN di Colab, karena butuh webcam & jendela):
    python 10_hidung_badut.py              # pakai webcam
    python 10_hidung_badut.py video.mp4    # atau pakai file video

Tombol:  q = keluar   r = cari wajah lagi   s = pilih hidung sendiri dengan mouse

Cara kerjanya (4 langkah):
  1. Ambil gambar dari webcam, satu frame demi satu frame.
  2. Cari wajah SEKALI saja (detektor YuNet langsung memberi titik ujung hidung).
  3. Ikuti titik-titik di sekitar hidung dengan optical flow Lucas-Kanade.
  4. Gambar bulatan merah (hidung badut) di tengah titik-titik itu.
  Kalau titiknya hilang (misalnya wajah ditutup tangan), kembali ke langkah 2.
"""
import os
import sys
import urllib.request
import cv2
import numpy as np

# --- 1. Sumber video: angka 0 = webcam bawaan laptop -------------------------
sumber = sys.argv[1] if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(sumber)

# --- Detektor wajah YuNet (cara kerjanya dibahas di Pertemuan 11) ------------
# File model (±230 KB) diunduh otomatis sekali ke folder model/ di sebelah script ini.
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "face_detection_yunet_2023mar.onnx")
URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
if not os.path.exists(MODEL):
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print("Mengunduh model detektor wajah (sekali saja)...")
    urllib.request.urlretrieve(URL, MODEL)
detektor = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), 0.8)

# Pengaturan optical flow Lucas-Kanade: jendela 21x21 piksel, 3 tingkat piramida
LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


def titik_di_kotak(abu, x, y, w, h):
    """Ambil titik-titik yang mudah diikuti (sudut/tekstur) di dalam sebuah kotak."""
    mask = np.zeros_like(abu)
    mask[int(y):int(y + h), int(x):int(x + w)] = 255
    titik = cv2.goodFeaturesToTrack(abu, maxCorners=30, qualityLevel=0.01, minDistance=3, mask=mask)
    if titik is None or len(titik) < 5:
        # kulit terlalu mulus? pakai saja kisi 5x5 titik di dalam kotak
        gx, gy = np.meshgrid(np.linspace(x, x + w, 5), np.linspace(y, y + h, 5))
        titik = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
    return titik.astype(np.float32)


def mulai_dari_kotak(abu, x, y, w, h):
    """Kotak hidung -> (titik yang diikuti, geser dari median titik ke pusat hidung, jari-jari)."""
    titik = titik_di_kotak(abu, x, y, w, h)
    pusat = np.array([x + w / 2, y + h / 2])
    geser = pusat - np.median(titik.reshape(-1, 2), axis=0)
    return titik, geser, max(10, int(w * 0.6))


def cari_hidung(frame, abu):
    """Langkah 2: cari wajah terbesar, lalu kotak kecil berpusat di ujung hidung."""
    detektor.setInputSize((frame.shape[1], frame.shape[0]))
    _, wajah = detektor.detect(frame)
    if wajah is None:
        return None, None, None
    f = max(wajah, key=lambda k: k[2] * k[3])    # wajah terbesar (paling dekat kamera)
    lebar = f[2]
    hx, hy = f[8], f[9]                          # YuNet memberi koordinat ujung hidung
    s = 0.24 * lebar                             # kotak hidung = ±1/4 lebar wajah
    return mulai_dari_kotak(abu, hx - s / 2, hy - s / 2, s, s)


def gambar_hidung(frame, pusat, r):
    """Langkah 4: hidung badut = bulatan merah + garis tepi + kilau putih."""
    x, y = int(pusat[0]), int(pusat[1])
    cv2.circle(frame, (x, y), r, (40, 40, 230), -1, cv2.LINE_AA)                          # merah
    cv2.circle(frame, (x, y), r, (20, 20, 140), 2, cv2.LINE_AA)                           # tepi
    cv2.circle(frame, (x - r // 3, y - r // 3), max(2, r // 4), (255, 255, 255), -1, cv2.LINE_AA)  # kilau


titik, geser, jari2 = None, None, 20
abu_lama = None

while True:
    ok, frame = cap.read()                       # langkah 1: ambil satu frame
    if not ok:
        break                                    # video habis / webcam tidak terbaca
    if sumber == 0:
        frame = cv2.flip(frame, 1)               # efek cermin agar terasa natural
    abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if titik is None or len(titik) < 5:
        titik, geser, jari2 = cari_hidung(frame, abu)   # langkah 2: (ulang) cari wajah
    else:
        # langkah 3: ke mana titik-titik itu pindah dari frame lama ke frame baru?
        baru, status, _ = cv2.calcOpticalFlowPyrLK(abu_lama, abu, titik, None, **LK)
        baru = baru[status.ravel() == 1].reshape(-1, 1, 2)
        if len(baru) > 0:
            # buang titik "nyasar" yang terlalu jauh dari kawanannya
            tengah = np.median(baru.reshape(-1, 2), axis=0)
            jarak = np.linalg.norm(baru.reshape(-1, 2) - tengah, axis=1)
            baru = baru[jarak < 3 * jari2]
        titik = baru

    if titik is not None and len(titik) >= 5:
        pusat = np.median(titik.reshape(-1, 2), axis=0) + geser
        gambar_hidung(frame, pusat, jari2)
    else:
        cv2.putText(frame, "Mencari wajah... hadap ke kamera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "q: keluar  r: ulang  s: pilih hidung", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Hidung Badut - Modul 10", frame)
    abu_lama = abu

    tombol = cv2.waitKey(1) & 0xFF
    if tombol == ord("q"):
        break
    if tombol == ord("r"):
        titik = None                             # paksa cari wajah lagi
    if tombol == ord("s"):
        # pilih kotak hidung sendiri: seret mouse, lalu tekan ENTER/SPASI
        x, y, w, h = cv2.selectROI("Hidung Badut - Modul 10", frame, showCrosshair=True)
        if w > 0 and h > 0:
            titik, geser, jari2 = mulai_dari_kotak(abu, x, y, w, h)

cap.release()
cv2.destroyAllWindows()
