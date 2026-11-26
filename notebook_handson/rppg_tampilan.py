"""
rppg_tampilan.py — urusan kamera & tampilan untuk live coding rPPG (Modul 13)
IF25-40305 Sistem Teknologi Multimedia · ITERA

File ini BUKAN materi inti. Isinya hanya hal-hal yang membosankan untuk diketik di kelas
(buka kamera, tulis teks, gambar grafik), supaya live coding fokus ke:
deteksi wajah -> ROI -> tracking -> filter -> FFT -> POS.

Pakai:
    from rppg_tampilan import Kamera, tulis, gambar_gelombang, tampilkan_bpm

    kamera = Kamera()                      # webcam; atau: python 13_live.py video/wajah.mp4
    while True:
        ok, frame, t = kamera.baca()       # t = waktu frame dalam detik
        if not ok:
            break
        ...
        if kamera.tampil(frame) == "q":    # tampilkan + baca tombol
            break
"""
import sys
import time

import cv2
import numpy as np

KUNING = (74, 195, 232)    # BGR
HIJAU = (107, 200, 46)
MERAH = (46, 16, 200)
PUTIH = (255, 255, 255)


class Kamera:
    """Webcam (default) atau file video (argumen pertama di terminal).

    baca()        -> (ok, frame, waktu_detik). Webcam dicerminkan; waktu = jam asli.
                     File video: waktu = nomor_frame / fps video (jadi tetap benar walau diproses cepat).
    tampil(frame) -> huruf tombol yang ditekan ("" kalau tidak ada). FPS ditulis di pojok kanan atas.
    """

    def __init__(self, sumber=None):
        if sumber is None:
            sumber = sys.argv[1] if len(sys.argv) > 1 else 0
        self.webcam = sumber == 0
        self.cap = cv2.VideoCapture(sumber)
        if self.webcam:                              # 720p 30 FPS; 1080p hanya memperberat
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_video = fps if 1 <= fps <= 120 else 30.0
        self.nomor = 0                               # jumlah frame yang sudah dibaca
        self.fps = 0.0
        self._t0 = self._t_frame = time.time()

    def baca(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None, 0.0
        if self.webcam:
            frame = cv2.flip(frame, 1)               # seperti cermin
        t = (time.time() - self._t0) if self.webcam else self.nomor / self.fps_video
        self.nomor += 1
        return True, frame, t

    def tampil(self, frame, judul="rPPG live coding - tekan q untuk keluar"):
        sekarang = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(sekarang - self._t_frame, 1e-3)
        self._t_frame = sekarang
        tulis(frame, f"{self.fps:.0f} FPS", (frame.shape[1] - 110, 32), 0.7)
        cv2.imshow(judul, frame)
        k = cv2.waitKey(1) & 0xFF
        huruf = chr(k) if k < 255 else ""
        if huruf in ("q", "\x1b"):                   # q atau Esc
            self.cap.release()
            cv2.destroyAllWindows()
            return "q"
        return huruf


def tulis(img, teks, xy, skala=0.7, warna=PUTIH, tebal=2):
    """Teks dengan garis tepi hitam, supaya terbaca di atas wajah/latar apa pun."""
    xy = (int(xy[0]), int(xy[1]))
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), tebal + 3, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, tebal, cv2.LINE_AA)


def gambar_gelombang(img, s, judul="sinyal denyut (6 detik terakhir)", warna=KUNING, titik=180):
    """Panel gelap di bawah layar berisi `titik` sampel terakhir dari sinyal s (otomatis diskalakan)."""
    h, w = img.shape[:2]
    tinggi = 90
    img[h - tinggi:h] = (img[h - tinggi:h] * 0.35).astype(np.uint8)
    s = np.asarray(s, dtype=float)[-titik:]
    if len(s) > 2:
        s = s - s.mean()
        s = s / (np.abs(s).max() + 1e-9)
        xs = np.linspace(10, w - 10, len(s))
        ys = h - tinggi / 2 - s * (tinggi / 2 - 12)
        cv2.polylines(img, [np.int32(np.stack([xs, ys], axis=1))], False, warna, 2, cv2.LINE_AA)
    tulis(img, judul, (12, h - tinggi + 18), 0.45, (200, 200, 200), 1)


def tampilkan_bpm(img, bpm, metode="GREEN", sinyal=None):
    """Angka BPM besar + lingkaran merah yang berdenyut mengikuti sinyal. bpm=None -> 'mengumpulkan data'."""
    warna = KUNING if metode == "POS" else HIJAU
    if bpm is None:
        tulis(img, "mengumpulkan data... duduk diam", (15, 50), 0.8, warna)
        return
    nilai = 0.0
    if sinyal is not None and len(sinyal) > 1:
        akhir = np.asarray(sinyal[-60:], dtype=float)
        nilai = max(akhir[-1] / (np.abs(akhir).max() + 1e-9), 0.0)
    cv2.circle(img, (40, 52), int(14 + 8 * nilai), MERAH, -1, cv2.LINE_AA)
    tulis(img, f"{bpm:.0f} BPM", (70, 68), 1.4, warna, 3)
    tulis(img, f"metode {metode}  (m = ganti)", (15, 100), 0.55)
