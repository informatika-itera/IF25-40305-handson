"""
13_rppg_pos.py — Detak jantung dari wajah lewat webcam (rPPG metode POS)
IF25-40305 Sistem Teknologi Multimedia · Modul 13 · ITERA

Cara menjalankan (di laptop sendiri, BUKAN di Colab, karena butuh webcam & jendela):
    python 13_rppg_pos.py                                # pakai webcam
    python 13_rppg_pos.py video/wajah_saya.mp4           # atau pakai file video wajah
    python 13_rppg_pos.py video.mp4 --simpan hasil.mp4   # tanpa jendela, hasil disimpan ke file

Tombol:  q = keluar   m = ganti metode (POS / GREEN)   d = paksa deteksi wajah ulang
         r = mulai hitung ulang   s = simpan sinyal R,G,B ke CSV (untuk notebook)   k = kotak kulit on/off

Cara kerjanya (setiap frame):
  1. CARI WAJAH SEKALI, LALU IKUTI. Detektor wajah YuNet (Modul 11) hanya dijalankan di awal.
     Frame berikutnya kotak wajah cukup diikuti tracker KCF (Modul 10) yang jauh lebih ringan,
     jadi FPS tidak turun. Keduanya bekerja di salinan frame yang DIPERKECIL (lebar 320 px):
     KCF di frame 720p butuh ~70 ms, di 320 px hanya ~5 ms. Deteksi ulang HANYA kalau:
       - tracker kehilangan wajah, atau
       - kotak wajah sudah bergeser jauh dari posisi saat terakhir dideteksi, atau
       - Anda menekan tombol d.
     Karena KCF bekerja di frame 320 px, posisinya meloncat per 4 px di frame asli (1 px kecil = 4 px asli).
     Supaya kotak tidak bergetar, posisinya DIHALUSKAN dengan dua aturan:
       - zona mati: geser < 3% lebar wajah diabaikan (kotak diam total saat wajah diam), dan
       - kalau lebih, kotak menyusul pelan: halus = 0,7 x halus lama + 0,3 x posisi baru (EMA).
     (EMA saja membuat kotak tidak pernah diam = getaran mikro; zona mati saja membuat kotak meloncat.)
  2. AMBIL WARNA KULIT dari SELURUH WAJAH, seperti POS (Wang dkk., 2017). Kotak wajah dipersempit ke
     bagian tengahnya (60% lebar, 80% tinggi) supaya latar, rambut, dan telinga tidak ikut terhitung.
     Rata-rata R, G, B di dalamnya (diambil dari frame ASLI, bukan yang diperkecil) = 3 angka per frame.
  3. HITUNG BPM (setiap 0,5 detik) dari 10 detik terakhir:
       samakan jarak waktu (30 titik per detik) -> POS (atau GREEN) -> buang tren
       -> bandpass 0,7-3 Hz (42-180 BPM) -> FFT -> frekuensi paling kuat x 60 = BPM.

Tips: duduk diam, wajah menghadap cahaya yang rata (jendela/lampu depan), jangan ada lampu berkedip.
Bandingkan hasilnya dengan nadi Anda sendiri: raba nadi di pergelangan, hitung 15 detik, kalikan 4.

Butuh: pip install opencv-contrib-python numpy scipy
"""
import os
import sys
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
from scipy import signal

FOLDER = os.path.dirname(os.path.abspath(__file__))

# --- 1. Sumber video: angka 0 = webcam bawaan laptop -------------------------
argumen = [a for a in sys.argv[1:] if not a.startswith("--")]
SIMPAN = sys.argv[sys.argv.index("--simpan") + 1] if "--simpan" in sys.argv else None
if SIMPAN in argumen:
    argumen.remove(SIMPAN)
sumber = argumen[0] if argumen else 0
WEBCAM = sumber == 0
cap = cv2.VideoCapture(sumber)
if WEBCAM:                                   # minta 720p 30 FPS; 1080p hanya memperberat tanpa manfaat
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
FPS_FILE = cap.get(cv2.CAP_PROP_FPS) or 30
if FPS_FILE < 1 or FPS_FILE > 120:
    FPS_FILE = 30

# --- 2. Detektor wajah YuNet (±230 KB; sudah ada di folder model/, kalau belum diunduh) ---
MODEL = os.path.join(FOLDER, "model", "face_detection_yunet_2023mar.onnx")
URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
       "face_detection_yunet_2023mar.onnx")
if not os.path.exists(MODEL):
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    print("Mengunduh detektor wajah YuNet (sekali saja)...")
    urllib.request.urlretrieve(URL, MODEL)
detektor = cv2.FaceDetectorYN.create(MODEL, "", (320, 320), 0.7)

# --- 3. Pengaturan (boleh diubah) -------------------------------------------------
FS = 30.0                 # sinyal disamakan menjadi 30 titik per detik
JENDELA = 10.0            # BPM dihitung dari 10 detik terakhir
MINIMAL = 6.0             # BPM pertama muncul setelah 6 detik data
BPM_MIN, BPM_MAX = 42, 180
BATAS_GESER = 0.35        # deteksi ulang kalau kotak bergeser > 35% lebar wajah sejak deteksi terakhir
SETIAP = 0.5              # BPM diperbarui setiap 0,5 detik
LEBAR_PROSES = 320        # deteksi & tracking di frame selebar 320 px (jauh lebih cepat)
KULIT_LEBAR, KULIT_TINGGI = 0.6, 0.8   # bagian tengah kotak wajah yang dirata-rata (buang latar & rambut)
HALUS = 0.3               # bobot posisi baru saat menghaluskan kotak (kecil = lebih tenang tapi lebih telat)
ZONA_MATI = 0.03          # geser kotak < 3% lebar wajah diabaikan supaya kotak diam saat wajah diam


def buat_tracker():
    """Tracker KCF (opencv-contrib). Kalau tidak ada, pakai MIL (ada di opencv biasa)."""
    for modul in (cv2, getattr(cv2, "legacy", None)):
        if modul is not None and hasattr(modul, "TrackerKCF_create"):
            return modul.TrackerKCF_create()
    return cv2.TrackerMIL_create()


def cari_wajah(frame):
    """YuNet -> kotak wajah (x, y, lebar, tinggi) atau None."""
    h, w = frame.shape[:2]
    detektor.setInputSize((w, h))
    _, wajah = detektor.detect(frame)
    if wajah is None or len(wajah) == 0:
        return None
    f = max(wajah, key=lambda r: r[2] * r[3])               # wajah terbesar = paling dekat
    x, y = max(0.0, float(f[0])), max(0.0, float(f[1]))   # kotak dijaga tetap di dalam frame
    lw, lh = min(float(f[2]), w - x - 1), min(float(f[3]), h - y - 1)
    return x, y, lw, lh


def kotak_kulit(kotak, w, h):
    """Bagian TENGAH kotak wajah (x0, y0, x1, y1), dipotong supaya tetap di dalam frame. None kalau terlalu kecil."""
    x, y, lw, lh = kotak
    cx, cy = x + lw / 2, y + lh / 2
    ww, hh = KULIT_LEBAR * lw, KULIT_TINGGI * lh
    x0, y0 = int(np.clip(cx - ww / 2, 0, w - 1)), int(np.clip(cy - hh / 2, 0, h - 1))
    x1, y1 = int(np.clip(cx + ww / 2, 1, w)), int(np.clip(cy + hh / 2, 1, h))
    return (x0, y0, x1, y1) if x1 - x0 > 2 and y1 - y0 > 2 else None


def pos(rgb, fs):
    """POS (Wang dkk., 2017). rgb: (N, 3) -> sinyal denyut (N,).
    Jendela pendek 1,6 detik digeser per titik, lalu hasilnya ditumpuk."""
    n, L = len(rgb), int(1.6 * fs)
    h = np.zeros(n)
    for awal in range(0, n - L + 1):
        c = rgb[awal:awal + L]
        cn = c / (c.mean(axis=0) + 1e-9)                    # 1) bagi rata-rata: warna kulit jadi (1, 1, 1)
        s1 = cn[:, 1] - cn[:, 2]                            # 2) dua "arah pandang": G - B
        s2 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]            #    dan -2R + G + B
        p = s1 + (s1.std() / (s2.std() + 1e-9)) * s2        # 3) campur: noise di dua arah saling menghapus
        h[awal:awal + L] += p - p.mean()                    # 4) tumpuk
    return h


def hitung_bpm(waktu, rgb, metode):
    """Dari sampel (waktu, R, G, B) -> (bpm, keyakinan 0-1, sinyal bersih)."""
    t, x = np.asarray(waktu), np.asarray(rgb, dtype=float)
    tt = np.arange(t[0], t[-1], 1 / FS)                     # jarak waktu webcam tidak rata -> samakan
    seragam = np.stack([np.interp(tt, t, x[:, k]) for k in range(3)], axis=1)
    if metode == "POS":
        s = pos(seragam, FS)
    else:                                                    # GREEN: hanya kanal hijau
        s = -(seragam[:, 1] / seragam[:, 1].mean() - 1)     # dibalik: darah banyak = hijau lebih gelap
    s = signal.detrend(s)                                   # buang tren pelan (cahaya berubah)
    b, a = signal.butter(3, [BPM_MIN / 60, BPM_MAX / 60], btype="band", fs=FS)
    s = signal.filtfilt(b, a, s)                            # sisakan 0,7-3 Hz saja
    spek = np.abs(np.fft.rfft(s * np.hanning(len(s)), n=4096)) ** 2
    frek = np.fft.rfftfreq(4096, 1 / FS)
    pita = (frek >= BPM_MIN / 60) & (frek <= BPM_MAX / 60)
    i = np.argmax(spek * pita)
    sekitar = pita & (np.abs(frek - frek[i]) < 0.1)         # daya di sekitar puncak (±6 BPM)
    keyakinan = spek[sekitar].sum() / (spek[pita].sum() + 1e-12)
    return frek[i] * 60, float(keyakinan), s


def tulis(img, teks, xy, skala=0.7, warna=(255, 255, 255), tebal=2):
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), tebal + 3, cv2.LINE_AA)
    cv2.putText(img, teks, xy, cv2.FONT_HERSHEY_SIMPLEX, skala, warna, tebal, cv2.LINE_AA)


def gambar_gelombang(img, s, warna):
    """Panel bawah: 6 detik terakhir sinyal denyut."""
    h, w = img.shape[:2]
    tinggi = 90
    panel = img[h - tinggi:h].copy()
    panel[:] = (panel * 0.35).astype(np.uint8)
    img[h - tinggi:h] = panel
    s = s[-int(6 * FS):]
    if len(s) > 2:
        s = (s - s.mean()) / (np.abs(s).max() + 1e-9)
        xs = np.linspace(10, w - 10, len(s))
        ys = h - tinggi / 2 - s * (tinggi / 2 - 12)
        cv2.polylines(img, [np.int32(np.stack([xs, ys], axis=1))], False, warna, 2, cv2.LINE_AA)
    tulis(img, "sinyal denyut (6 detik terakhir)", (12, h - tinggi + 18), 0.45, (200, 200, 200), 1)


# --- 4. Lingkaran utama ---------------------------------------------------------
METODE = "POS"
tracker, kotak = None, None
halus = None                         # kotak wajah (ukuran asli) yang sudah dihaluskan -> ROI tidak bergetar
kotak_awal = None                    # kotak saat terakhir dideteksi (untuk cek "bergeser jauh")
data = deque()                       # (waktu, R, G, B) untuk jendela 10 detik
rekaman = []                         # semua sampel sejak mulai (untuk disimpan ke CSV)
bpm_list = deque(maxlen=5)           # 5 tebakan terakhir -> median supaya angka tidak lompat-lompat
bpm, keyakinan, bersih = None, 0.0, np.zeros(0)
hitung_terakhir = -1.0
n_deteksi, nomor, t0 = 0, 0, time.time()
ms_deteksi, ms_lacak = 0.0, 0.0
paksa_deteksi, tampil_roi = False, True
penulis = None
fps_tampil, t_frame = 0.0, time.time()
pesan, pesan_sampai = "", 0.0        # pesan status ditahan 2 detik supaya sempat terbaca


def simpan_csv():
    if not rekaman:
        return None
    nama = os.path.join(FOLDER, "output", f"rppg_sinyal_{time.strftime('%H%M%S')}.csv")
    os.makedirs(os.path.dirname(nama), exist_ok=True)
    np.savetxt(nama, np.array(rekaman), delimiter=",", header="waktu_s,R,G,B", comments="", fmt="%.4f")
    return nama


print("Tombol: q keluar | m POS/GREEN | d deteksi ulang | r mulai ulang | s simpan CSV | k kotak kulit")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if WEBCAM:
        frame = cv2.flip(frame, 1)                              # seperti cermin
    h, w = frame.shape[:2]
    sekarang = (time.time() - t0) if WEBCAM else nomor / FPS_FILE
    status = ""
    # salinan kecil untuk detektor & tracker; kotak wajah disimpan dalam koordinat frame kecil ini
    skala = min(1.0, LEBAR_PROSES / w)
    kecil = cv2.resize(frame, None, fx=skala, fy=skala, interpolation=cv2.INTER_AREA)

    # (a) ikuti wajah dengan tracker (ringan)
    if tracker is not None and not paksa_deteksi:
        mulai = time.perf_counter()
        ok_lacak, kotak_baru = tracker.update(kecil)
        ms_lacak = 0.9 * ms_lacak + 0.1 * (time.perf_counter() - mulai) * 1000
        if not ok_lacak:
            tracker, status = None, "tracker kehilangan wajah -> deteksi ulang"
        else:
            kotak = tuple(float(v) for v in kotak_baru)
            geser = np.hypot(kotak[0] - kotak_awal[0], kotak[1] - kotak_awal[1])
            if geser > BATAS_GESER * kotak_awal[2]:
                tracker, status = None, "wajah bergeser jauh -> deteksi ulang"

    # (b) cari wajah dengan detektor (berat) hanya kalau perlu
    if tracker is None or paksa_deteksi:
        mulai = time.perf_counter()
        hasil = cari_wajah(kecil)
        ms_deteksi = (time.perf_counter() - mulai) * 1000
        paksa_deteksi = False
        if hasil is not None:
            kotak = kotak_awal = hasil
            tracker = buat_tracker()
            tracker.init(kecil, tuple(int(v) for v in kotak))
            n_deteksi += 1
        else:
            tracker, kotak = None, None

    # (c) rata-rata warna kulit di bagian tengah wajah
    if kotak is not None and tracker is not None:
        kotak_asli = np.array(kotak, dtype=float) / skala       # kembalikan ke ukuran frame asli
        if halus is None:
            halus = kotak_asli
        elif np.abs(kotak_asli - halus).max() > ZONA_MATI * kotak_asli[2]:   # geser kecil diabaikan
            halus = (1 - HALUS) * halus + HALUS * kotak_asli                 # geser sungguhan: susul pelan
        kulit = kotak_kulit(halus, w, h)
        if kulit is not None:
            x0, y0, x1, y1 = kulit
            b_, g_, r_ = frame[y0:y1, x0:x1].mean(axis=(0, 1))    # OpenCV = BGR
            data.append((sekarang, r_, g_, b_))
            rekaman.append((sekarang, r_, g_, b_))
            if tampil_roi:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (107, 158, 46), 2)
        x, y, lw, lh = (int(v) for v in halus)
        cv2.rectangle(frame, (x, y), (x + lw, y + lh), (255, 255, 255), 1)
    else:
        status = status or "tidak ada wajah - hadap ke kamera"
    while data and sekarang - data[0][0] > JENDELA:                # buang data lebih tua dari 10 detik
        data.popleft()

    # (d) hitung BPM setiap 0,5 detik
    lama = data[-1][0] - data[0][0] if len(data) > 1 else 0.0
    if lama >= MINIMAL and sekarang - hitung_terakhir >= SETIAP:
        tebakan, keyakinan, bersih = hitung_bpm([d[0] for d in data], [d[1:] for d in data], METODE)
        bpm_list.append(tebakan)
        bpm = float(np.median(bpm_list))
        hitung_terakhir = sekarang

    # (e) tampilan
    warna = (74, 195, 232) if METODE == "POS" else (107, 200, 46)
    ada_panel = bpm is not None and lama >= MINIMAL
    if ada_panel:
        nilai = bersih[-1] / (np.abs(bersih[-int(2 * FS):]).max() + 1e-9) if len(bersih) else 0
        cv2.circle(frame, (40, 52), int(14 + 8 * max(nilai, 0)), (46, 16, 200), -1, cv2.LINE_AA)
        tulis(frame, f"{bpm:.0f} BPM", (70, 68), 1.4, warna, 3)
        tulis(frame, f"metode {METODE} | keyakinan {keyakinan * 100:.0f}%", (15, 100), 0.55)
        if keyakinan < 0.25:
            tulis(frame, "sinyal lemah: diam sebentar & cari cahaya rata", (15, 125), 0.55, (46, 16, 200))
        gambar_gelombang(frame, bersih, warna)
    else:
        tulis(frame, f"mengumpulkan data... {lama:.0f}/{MINIMAL:.0f} detik", (15, 50), 0.8, warna)
        tulis(frame, "duduk diam, wajah menghadap cahaya", (15, 80), 0.55)
    nomor += 1
    jeda, t_frame = time.time() - t_frame, time.time()
    fps_tampil = 0.9 * fps_tampil + 0.1 / max(jeda, 1e-3)
    tulis(frame, f"deteksi wajah: {n_deteksi}x dalam {nomor} frame | YuNet {ms_deteksi:.0f} ms vs KCF "
                 f"{ms_lacak:.0f} ms | {fps_tampil:.0f} FPS", (15, h - 100 if ada_panel else h - 15), 0.45, (220, 220, 220), 1)
    if status:
        pesan, pesan_sampai = status, time.time() + 2
    if time.time() < pesan_sampai:
        tulis(frame, pesan, (15, 150), 0.55, (0, 200, 255))

    if SIMPAN:
        if penulis is None:
            penulis = cv2.VideoWriter(SIMPAN, cv2.VideoWriter_fourcc(*"mp4v"), FPS_FILE, (w, h))
        penulis.write(frame)
    else:
        cv2.imshow("rPPG POS STM - tekan q untuk keluar", frame)
        tombol = cv2.waitKey(1) & 0xFF
        if tombol == ord("q"):
            break
        if tombol == ord("m"):
            METODE = "GREEN" if METODE == "POS" else "POS"
            bpm_list.clear()
            hitung_terakhir = -1.0
        if tombol == ord("d"):
            paksa_deteksi = True
        if tombol == ord("r"):
            for wadah in (data, rekaman, bpm_list):
                wadah.clear()
            bpm, hitung_terakhir = None, -1.0
        if tombol == ord("k"):
            tampil_roi = not tampil_roi
        if tombol == ord("s"):
            print("Sinyal disimpan:", simpan_csv())

cap.release()
if penulis is not None:
    penulis.release()
    print(f"Hasil disimpan: {SIMPAN} ({nomor} frame)")
if not WEBCAM:
    print("Sinyal disimpan:", simpan_csv())                    # file video: CSV otomatis untuk notebook
print(f"Selesai: {nomor} frame, detektor wajah dijalankan {n_deteksi}x"
      + (f", BPM terakhir {bpm:.0f}" if bpm else ""))
if not SIMPAN:
    cv2.destroyAllWindows()
