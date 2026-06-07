# 🔥 FirefighterPi

> Smart autonomous fire suppression robot — detects fire with AI, aims a water nozzle using pan/tilt servos, and extinguishes it automatically.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%20%7C%20Windows-green)
![Model](https://img.shields.io/badge/Model-YOLOv8n%20ONNX-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Demo

| State | Description |
|-------|-------------|
| 🔍 **IDLE** | Kamera aktif, model AI memindai frame terus-menerus |
| 🎯 **LOCKING** | Api terdeteksi, menunggu posisi stabil 0.5 detik |
| ↗️ **AIMING** | Perintah servo dikirim ke Arduino, menunggu pergerakan mekanik 1 detik |
| 💧 **EXTINGUISHING** | Pompa aktif, tracking posisi api secara dinamis |

---

## System Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  USB Webcam │───▶│  YOLOv8n ONNX    │───▶│ Perspective         │
│  640 × 480  │    │  (320 × 320)     │    │ Transform → mm      │
└─────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                       │
                                                       ▼
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ Servo       │◀───│  Arduino Uno     │◀───│ calculate_angles()  │
│ Pan + Tilt  │    │  UART 9600 baud  │    │ IDW / atan2 math    │
└─────────────┘    └──────────────────┘    └─────────────────────┘
       │
       ▼
┌─────────────┐
│ Relay Pump  │  M0 = ON  │  M1 = OFF
└─────────────┘
```

**Serial Protocol ke Arduino:**
```
P<0-180>   → Pan servo angle
T<0-180>   → Tilt servo angle
M0 / M1    → Pompa ON / OFF
```

---

## Hardware

| Komponen | Spesifikasi |
|----------|-------------|
| Board | Raspberry Pi 3B+ |
| Kamera | USB Webcam (mount di plafon, menghadap ke bawah) |
| Mikrokontroler | Arduino Uno (kontrol servo + relay via UART) |
| Servo | 2× servo 0–180° (pan = horizontal, tilt = vertikal) |
| Pompa | Pompa air + relay |

---

## Software Stack

- **Inference:** ONNX Runtime (CPU) — tidak butuh GPU
- **Computer Vision:** OpenCV 4.x
- **Web Stream:** Flask (MJPEG stream di port 5000)
- **Serial:** PySerial (auto-detect COM* / /dev/tty*)

---

## Instalasi

### Windows (dev / test)

```bash
conda create -n firefighter_pi python=3.12
conda activate firefighter_pi
pip install -r requirements.txt
```

### Raspberry Pi (produksi)

> ⚠️ Gunakan **Raspberry Pi OS 64-bit** — onnxruntime tidak memiliki wheel untuk ARM32.

```bash
pip install -r requirements-pi.txt
```

---

## Menjalankan

```bash
# Sistem utama — deteksi + kontrol servo + stream web
python app.py
# Buka http://<ip>:5000
```

```bash
# Tool kalibrasi (jalankan terpisah, STOP app.py dulu)
python calibration_server.py
# Buka http://<ip>:5001
```

```bash
# Capture dataset gambar (butuh display)
python shutter.py
# SPACE = simpan frame  |  q = keluar
```

---

## Kalibrasi

Kalibrasi dilakukan dua tahap via `calibration_server.py` (port 5001).  
**Penting:** hentikan `app.py` terlebih dahulu — keduanya tidak bisa berbagi serial port bersamaan.

### Mode A — Homography (Wajib)

Memetakan pixel kamera ke koordinat lantai nyata (mm).

1. Buka **Mode A** di browser
2. Klik 4 sudut area lantai yang terlihat kamera, berurutan:
   `Kiri-Atas → Kanan-Atas → Kanan-Bawah → Kiri-Bawah`
3. Salin hasil `SRC_POINTS` ke `app.py` dan `calibration_server.py`
4. Update konstanta fisik sesuai ukuran area (ukur dengan meteran):

```python
LEBAR_FISIK_ATAS   = 490.0   # mm
LEBAR_FISIK_BAWAH  = 500.0   # mm
TINGGI_FISIK_KIRI  = 270.0   # mm
TINGGI_FISIK_KANAN = 290.0   # mm
CEILING_HEIGHT     = 500.0   # mm (tinggi plafon ke lantai)
```

### Mode B — Servo Aim (Opsional, meningkatkan akurasi)

Membangun ground-truth mapping `(pan, tilt) → koordinat lantai` dari pengukuran langsung, menggantikan model atan2 murni dengan IDW interpolation.

1. Buka **Mode B** di browser
2. Set sudut servo (misal Pan=90, Tilt=90) → klik **Kirim ke Servo**
3. Tunggu 2 detik → klik posisi nozzle/laser di gambar kamera
4. Ulangi untuk ~9 posisi (grid 3×3, contoh: pan 60/90/120 × tilt 70/90/110)
5. Klik **Simpan JSON** → file `servo_calibration.json` terbentuk
6. Restart `app.py` — kalibrasi otomatis ter-load

> Jika `servo_calibration.json` tidak ada atau kurang dari 3 titik, `app.py` fallback ke model trigonometri (atan2).

---

## Struktur Proyek

```
FirefighterPi/
├── app.py                    ← Entry point utama
├── calibration_server.py     ← Tool kalibrasi (port 5001)
├── shutter.py                ← Capture dataset gambar
├── config.yaml               ← Referensi konfigurasi (tidak di-load runtime)
├── requirements.txt          ← Deps Windows
├── requirements-pi.txt       ← Deps Raspberry Pi (64-bit)
├── models/
│   └── fire_detector-3/
│       └── weights/
│           └── best.onnx     ← Model aktif (tidak di-track git)
├── archive/                  ← Iterasi lama (main.py – main5.py)
└── datasets/                 ← Gambar capture (tidak di-track git)
```

---

## Model

Dilatih dengan **Ultralytics YOLOv8n**, diekspor ke ONNX untuk inference tanpa dependensi Ultralytics:

```bash
model.export(format='onnx', imgsz=320)
```

Untuk melatih ulang: anotasi gambar dalam format YOLO (`.txt` per gambar), lalu ekspor ke ONNX.

---

## Lisensi

MIT License — bebas digunakan dan dimodifikasi.
