<h1 align="center"><b>IF25-40305: Sistem Teknologi Multimedia</b></h1>
<h3 align="center">Multimedia: System & Technology (STM)</h3>

<p align="center">
  <b>Program Studi Teknik Informatika — Institut Teknologi Sumatera (ITERA)</b><br>
  Tahun Ajaran 2026/2027 · Semester Ganjil · 3 SKS
</p>

<p align="center">
  <a href="https://mctm.web.id/course/if25-40305"><img src="https://img.shields.io/badge/Portal-mctm.web.id%2Fstm-0ea5e9?style=flat-square" alt="Portal Kuliah"></a>
  <a href="https://mctm.web.id/stm/rps"><img src="https://img.shields.io/badge/RPS-Silabus%20Lengkap-8b5cf6?style=flat-square" alt="Silabus"></a>
  <a href="https://mctm.web.id/stm/rules"><img src="https://img.shields.io/badge/Rules-Kontrak%20Kuliah-ec4899?style=flat-square" alt="Kontrak Kuliah"></a>
  <a href="https://chat.whatsapp.com/E7FaWI7HCyX0Fg4P6p8Ds9"><img src="https://img.shields.io/badge/WhatsApp-Grup%20Kelas-22c55e?style=flat-square&logo=whatsapp" alt="Grup WhatsApp"></a>
</p>

---

## 📌 Identitas Perkuliahan

| Informasi | Keterangan |
|---|---|
| **Mata Kuliah** | **Sistem Teknologi Multimedia** (*Multimedia: System & Technology*) |
| **Kode Mata Kuliah** | **IF25-40305** *(Kurikulum Sebelumnya: IF4021)* |
| **Bobot SKS** | **3 (tiga) SKS** — Beban belajar 135 jam/semester (Peraturan Rektor ITERA No. 2 Th 2024) |
| **Dosen Penanggung Jawab** | **Martin C.T. Manullang, S.T., M.T., Ph.D.** |
| **Jadwal Kuliah** | **Kamis, 13.00 – 15.40 WIB** |
| **Ruang Kuliah** | Menyesuaikan (TBA / Gedung Kuliah Umum ITERA) |
| **Portal Resmi** | [mctm.web.id/course/if25-40305](https://mctm.web.id/course/if25-40305) |
| **LMS Perkuliahan** | [kuliah2.itera.ac.id](https://kuliah2.itera.ac.id) |

---

## 📖 Deskripsi Mata Kuliah

Mata kuliah **IF25-40305 Sistem Teknologi Multimedia (STM)** membahas konsep-konsep data multimedia mencakup data teks, audio, citra, dan video, serta integrasinya ke dalam sistem multimedia terpadu. Pendekatan perkuliahan berlandaskan **teori multimedia yang kokoh** dan **praktik langsung (*hands-on*)** menggunakan perangkat lunak standar industri.

Mahasiswa diajak mengeksplorasi representasi diskrit, kuantisasi, teorema Nyquist-Shannon, rantai konversi ADC-DAC, pengolahan sinyal dan filter audio, pemrosesan citra digital, spatial filtering, motion estimation pada video, pelacakan berbasis Vision AI (facial landmarks, pose estimation, hand tracking), jaringan multimedia & adaptive streaming (DASH/HLS), hingga teknologi mutakhir seperti *remote photoplethysmography* (rPPG).

---

## 💻 Lingkungan Komputasi & Toolchain

Praktikum dan tugas hands-on menggunakan ekosistem **Python 3.12**:

- **Python Version**: `Python 3.12` *(Disarankan menggunakan package manager [`uv`](https://docs.astral.sh/uv/) untuk isolasi environment yang cepat & ringan; alternatif `conda` / `venv` diperbolehkan)*.
- **Library Inti**:
  - **Audio Processing & MIR**: `librosa`, `soundfile`, `scipy`, `pyloudnorm`
  - **Citra & Kompresi**: `opencv-python` (`cv2`), `numpy`, `matplotlib`, `pillow`
  - **Vision AI, Landmark & Tracking**: `mediapipe`, `dlib`, `pyzbar`
  - **Tools & Multimedia Backend**: `jupyter` / `ipykernel`, `ffmpeg`

---

## 🗺️ Silabus & Rencana Pembelajaran Mingguan (16 Minggu)

| Minggu | Topik Pembahasan | Pokok Bahasan & Fokus Hands-on |
|:---:|---|---|
| **01** | **Pengantar Multimedia & Data Digital** | Elemen multimedia (teks, gambar, audio, video); representasi digital (bit, sampling, kuantisasi); Teorema Nyquist; bit depth & PCM; representasi teks (ASCII/Unicode); raster vs vektor. |
| **02** | **Akuisisi Media & Rantai ADC–DAC** | Citra sebagai matriks piksel, ruang warna BGR vs RGB & Grayscale; struktur video (frame, FPS); ekstraksi metadata; sampling diskrit; tahapan konversi ADC (LPF, S&H, kuantisasi, SQNR, PCM); rekonstruksi DAC (ZOH, filter rekonstruksi). |
| **03** | **Audio Fundamentals & Equalization** | Resampling (upsampling/downsampling); jenis filter audio (low-pass, high-pass, band-pass, notch); parameter filter (cutoff, roll-off, Q); perancangan equalizer (rumble removal, isolasi vokal, noise cleanup). |
| **04** | **Loudness & Dynamic Processing** | Metrik loudness: Peak (dBFS), RMS, LUFS; normalisasi loudness platform (Spotify −14 LUFS); gain control: fade in/out, crossfade; compressor (threshold, ratio, attack, release), limiter, noise gate. |
| **05** | **Music Information Retrieval (MIR)** | Pitch shifting & time stretching; deteksi pitch, tempo, onset, dan beat tracking; Chromagram: prediksi nada Mayor/Minor & kunci lagu; chord recognition dan audio fingerprinting. |
| **06** | **Struktur Citra Digital & Warna** | Struktur array NumPy (H × W × C); operasi I/O dasar (imread, imshow, imwrite); konversi ruang warna (BGR, RGB, HSV, Grayscale); channel splitting dan analisis histogram distribusi warna. |
| **07** | **Transformasi Geometris & Anotasi** | Transformasi affine (translasi, rotasi, scaling, flipping); perhitungan canvas baru anti-cropping; metode interpolasi (nearest, bilinear, bicubic); anotasi teks & bounding box pada citra. |
| **08** | **Filtering Spasial, Tepi & Kontur** | Konvolusi kernel; Gaussian blur untuk reduksi noise; sharpening; ekstraksi fitur bentuk melalui deteksi tepi (Canny/Sobel) dan deteksi kontur objek. |
| **09** | **UTS — Ujian Tengah Semester** | **Tes Tulis (Bobot 10%)** mencakup pemahaman konsep representasi digital & media (W01–W02), audio (W03–W05), dan citra digital (W06–W08). |
| **10** | **Video Digital & Motion Estimation** | Struktur video I/P/B-frame; VideoCapture & VideoWriter; motion estimation (block matching & Lucas-Kanade optical flow); chroma keying (green screen); object tracking klasik (CSRT/KCF). |
| **11** | **Deteksi Wajah, Landmark & Kode** | Komparasi Haar Cascades vs Dlib HOG/CNN; 68-point facial landmark & MediaPipe Face Mesh; Eye Aspect Ratio (EAR) untuk deteksi kedipan; decoding QR/barcode; image overlay (AR sederhana). |
| **12** | **Pose, Hand Tracking & Gestur** | Pose estimation 33 titik skeleton & sudut vektor sendi; hand tracking 21 titik (MediaPipe); logika gestur tangan (fist vs open palm, finger counter); proyek interaktif Virtual Painter. |
| **13** | **Multimedia Networking & Streaming** | Karakteristik jaringan (bandwidth, latency, jitter); prinsip QoS/QoE; adaptive streaming (DASH & HLS); bitrate adaptation, buffering, dan Content Delivery Network (CDN). |
| **14** | **Remote Photoplethysmography (rPPG)** | Interaksi cahaya-kulit & Blood Volume Pulse (BVP); ROI face tracking stabil; ekstraksi sinyal warna RGB; detrending, bandpass filtering (0.7–3.0 Hz / 42–180 BPM), estimasi Heart Rate non-contact. |
| **15** | **Integrasi Proyek & Showcase** | Integrasi modul audio/citra/video; troubleshooting & optimasi performa (FPS, latensi); deployment prototype; demo showcase di depan kelas dan peer review. |
| **16** | **UAS — Ujian Akhir Semester** | Pengumpulan laporan akhir tugas besar, evaluasi capaian akhir pembelajaran, dan penilaian final. |

---

## 📊 Sistem Penilaian

Rentang nilai mengacu pada **Standar Penilaian Pembelajaran ITERA** (*Nomor Dokumen: ITERA/LPMPP/SPMI/STD/A-03/2024*):

| Komponen Evaluasi | Bobot | Keterangan |
|---|:---:|---|
| **Tugas Individu** | **35%** | Tugas hands-on mandiri berkala (eksplorasi kode, analisis, dan problem-solving) |
| **Proyek (Tugas Besar)** | **30%** | Pengembangan aplikasi multimedia terpadu berbasis tim/individu beserta showcase |
| **Tes Tulis (UTS)** | **10%** | Ujian tertulis komprehensif konsep multimedia materi Minggu 1–8 |
| **Partisipasi Aktif** | **10%** | Keaktifan dalam menjawab, berdiskusi, dan menanggapi materi di kelas/grup (bukan presensi) |
| **Kuis Pra-UTS** | **7.5%** | Kuis evaluasi berkala paruh pertama semester |
| **Kuis Pra-Final** | **7.5%** | Kuis evaluasi berkala paruh kedua semester |

> **Batas Kelulusan**: Nilai minimum untuk lulus adalah **C (50)**. Pembulatan menggunakan 0 desimal (≥ 0,5 dibulatkan ke atas).

---

## 📜 Ringkasan Kontrak Kuliah & Integritas Akademik

1. **Toleransi Keterlambatan**: Maksimal **5 (lima) menit** sejak jadwal perkuliahan dimulai. Hadir melewati toleransi **tidak dapat melakukan presensi** (berlaku setara untuk dosen dan mahasiswa).
2. **Batas Ketidakhadiran**: Mahasiswa hanya diperkenankan tidak hadir maksimal **3 (tiga) kali** dalam satu semester. Melewati batas ini otomatis tidak diperkenankan mengikuti UAS.
3. **Keterlambatan Tugas**: Keterlambatan dengan alasan apa pun (termasuk kendala teknis/jaringan/laptop) **tidak mendapat toleransi dan bernilai 0 (nol)**. Disarankan mengumpulkan paling lambat H-6 jam.
4. **Kebijakan Penggunaan AI**: Mahasiswa diperkenankan menggunakan AI/LLM untuk belajar konsep dan *debugging*. Namun implementasi kode, eksperimen, analisis hasil, dan kesimpulan harus merupakan karya mandiri. Penggunaan AI untuk men-generate seluruh tugas tanpa kontribusi mandiri menyebabkan nilai 0 pada kategori tugas tersebut.
5. **Plagiarisme & Joki**: Sanksi atas joki atau plagiarisme adalah **nilai akhir E (diskualifikasi permanen)** dan pencatatan nama pada daftar hitam (*blacklist*).
6. **Etika Komunikasi (Bebas Feodalisme)**: Komunikasi dibangun atas dasar saling menghormati. Sampaikan pesan secara **singkat, jelas (*to the point*), dan santun**. Hindari basa-basi panjang, penggunaan kata *"izin/maaf mengganggu"*, atau penggunaan emoji berlebihan seperti 🙏.

---

## 📚 Arsip Semester Sebelumnya (*Branch Archives*)

Materi hands-on dan modul semester-semester lampau tetap tersimpan rapi dan dapat diakses pada branch masing-masing:

| Semester | Branch GitHub | Sorotan Materi |
|---|---|---|
| **2025/2026 Genap** | [`2025_2`](https://github.com/informatika-itera/IF25-40305-handson/tree/2025_2) | Audio Processing, Image Processing, Video Processing, Landmark Detection (Week 1–14) |
| **2025/2026 Ganjil** | [`2025_1`](https://github.com/informatika-itera/IF25-40305-handson/tree/2025_1) | Audio Module, Image Processing, Face/Pose Tracking, Interactive Web Visualizations |
| **2024/2025 Genap** | [`2024_2`](https://github.com/informatika-itera/IF25-40305-handson/tree/2024_2) | Audio Manipulation, Image Compression, Video Processing, Facial Landmarks & rPPG |
| **2024/2025 Ganjil** | [`2024_1`](https://github.com/informatika-itera/IF25-40305-handson/tree/2024_1) | Audio FFT, Equalizer, Filter Audio, JPEG Compression, Respiratory Tracking |

> **Navigasi Lokal:**
> ```bash
> git switch <nama_branch>
> # Contoh untuk membuka materi 2025_2:
> git switch 2025_2
> # Untuk kembali ke materi aktif:
> git switch main
> ```

---

## 🚀 Panduan Unduh untuk Mahasiswa

Untuk mengunduh repositori materi semester aktif saat ini:

```bash
# Shallow clone (sangat cepat & hemat kuota internet):
git clone --depth 1 https://github.com/informatika-itera/IF25-40305-handson.git
cd IF25-40305-handson
```

---

## 🔗 Tautan Penting

- 🌐 **Portal Perkuliahan STM**: [https://mctm.web.id/course/if25-40305](https://mctm.web.id/course/if25-40305)
- 📋 **Silabus / RPS Lengkap**: [https://mctm.web.id/stm/rps](https://mctm.web.id/stm/rps)
- 📜 **Kontrak Kuliah Lengkap**: [https://mctm.web.id/stm/rules](https://mctm.web.id/stm/rules)
- 💬 **Grup Koordinasi WhatsApp**: [https://chat.whatsapp.com/E7FaWI7HCyX0Fg4P6p8Ds9](https://chat.whatsapp.com/E7FaWI7HCyX0Fg4P6p8Ds9)
- 🏫 **Website Jurusan / Prodi**: [Teknik Informatika ITERA](https://if.itera.ac.id)
