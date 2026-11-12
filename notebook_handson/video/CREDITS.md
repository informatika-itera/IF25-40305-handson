# Kredit & Lisensi Aset Video

Video pada direktori ini digunakan untuk keperluan pendidikan pada mata kuliah
**IF25-40305 Sistem Teknologi Multimedia**, Institut Teknologi Sumatera (ITERA), Modul 10–11.

| Berkas | Isi | Pencipta / Sumber | Lisensi | Modifikasi |
|---|---|---|---|---|
| `bola_green_screen.mp4` | Bola pantai memantul di depan layar hijau (640×360, 30 fps, 4 detik; satu segmen bola sengaja berwarna hijau) | Dibuat secara prosedural oleh pengampu (Martin C.T. Manullang) dengan OpenCV | CC0 | — |
| `pejalan_kaki.mp4` | Pejalan kaki di persimpangan kampus | `vtest.avi` dari repositori sampel OpenCV (https://github.com/opencv/opencv/tree/4.x/samples/data) | Apache-2.0 | 150 frame pertama, diperkecil 768×576 → 640×480 (INTER_AREA), disimpan ulang dengan codec `mp4v` |
| `wawancara_kedip.mp4` | Wawancara Dr. Kelsey Young (ilmuwan NASA Goddard) tentang misi Artemis II — 20 detik, untuk deteksi kedipan Modul 11 | NASA Scientific Visualization Studio, *NASA Interview Opportunity: Moonbound! NASA's Artemis II Mission Days From Launch* (SVS 14934, https://svs.gsfc.nasa.gov/14934), via Wikimedia Commons | Domain publik (karya pemerintah AS) | Detik 100–120 dari klip "Kelsey Young SOTs"; dipotong 1280×720 di sekitar wajah lalu diperkecil ke 640×360 (INTER_AREA), 23,976 fps, codec `mp4v`, tanpa audio |

Model `../model/face_detection_yunet_2023mar.onnx` (YuNet, untuk `10_hidung_badut.py`) berasal dari
OpenCV Zoo (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), lisensi MIT.

Model tambahan untuk Modul 11:
- `../model/face_landmarker.task` — MediaPipe Face Landmarker (float16, v1), Google, lisensi Apache-2.0
  (https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker).
- `../model/haarcascade_frontalface_default.xml`, `../model/haarcascade_eye.xml` — Haar cascade dari repositori OpenCV
  (https://github.com/opencv/opencv/tree/4.x/data/haarcascades), Intel License Agreement (BSD-3-Clause).
