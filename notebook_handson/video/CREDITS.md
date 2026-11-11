# Kredit & Lisensi Aset Video

Video pada direktori ini digunakan untuk keperluan pendidikan pada mata kuliah
**IF25-40305 Sistem Teknologi Multimedia**, Institut Teknologi Sumatera (ITERA), Modul 10.

| Berkas | Isi | Pencipta / Sumber | Lisensi | Modifikasi |
|---|---|---|---|---|
| `bola_green_screen.mp4` | Bola pantai memantul di depan layar hijau (640×360, 30 fps, 4 detik; satu segmen bola sengaja berwarna hijau) | Dibuat secara prosedural oleh pengampu (Martin C.T. Manullang) dengan OpenCV | CC0 | — |
| `pejalan_kaki.mp4` | Pejalan kaki di persimpangan kampus | `vtest.avi` dari repositori sampel OpenCV (https://github.com/opencv/opencv/tree/4.x/samples/data) | Apache-2.0 | 150 frame pertama, diperkecil 768×576 → 640×480 (INTER_AREA), disimpan ulang dengan codec `mp4v` |

Model `../model/face_detection_yunet_2023mar.onnx` (YuNet, untuk `10_hidung_badut.py`) berasal dari
OpenCV Zoo (https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), lisensi MIT.
