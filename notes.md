# Firefighter Project — Knowledge Base

## Konsep
Raspberry Pi 3B+ dengan webcam (POV dari atas ruangan) mendeteksi api menggunakan YOLOv8n. Bounding box api diterjemahkan ke koordinat (x,y), lalu 2 servo pan/tilt (0-180°) mengarahkan semprotan air ke sumber api — semuanya dipasang di plafon.

## Arsitektur Sistem
```
Webcam → Deteksi YOLOv8n → Koordinat (x,y) → Servo Pan/Tilt → Pump + Solenoid Valve
```

## Hardware
- **Board:** Raspberry Pi 3B+
- **Kamera:** USB Webcam (mount di plafon, menghadap ke bawah)
- **Servo:** 2 buah, range 0-180° (Pan = horizontal, Tilt = vertikal)
- **Water Spray:** Pompa air + solenoid valve, kontrol via relay GPIO
- **Catu Daya:** (TBD)

## Software Stack
- **OS:** Raspberry Pi OS (Linux)
- **Language:** Python
- **Computer Vision:** YOLOv8n (Ultralytics), ONNX Runtime, OpenCV
- **GPIO:** RPi.GPIO / pigpio untuk servo PWM
- **Koordinat:** Mapping pixel (frame kamera) → sudut servo

## Struktur Proyek
```
firefighter/
├── main.py           # Main application (1 file, detection + control)
├── shutter.py        # Tool capture dataset gambar
├── train.ipynb       # Notebook training YOLOv8 (akan dibuat nanti)
├── datasets/         # Raw captured images
├── labeled/          # Annotated images (YOLO format)
├── models/           # YOLO weights (.pt / .onnx)
├── config.yaml       # Konfigurasi parameter
├── requirements.txt  # Python dependencies
└── notes.md          # File ini
```

## Coordinate System
- Kamera di plafon, pointing ke bawah
- Bounding box centroid (x_pixel, y_pixel) → mapping ke servo angle
- Pan: 0-180° (kiri-kanan), Tilt: 0-180° (maju-mundur relatif dari plafon)
- FOV kamera menentukan batas area yang bisa dijangkau
- Asumsi: kamera dan servo co-located

## State Machine (Main Loop)
1. **IDLE** — Kamera aktif, deteksi berjalan, tidak ada api
2. **TARGET_DETECTED** — Api terdeteksi, hitung koordinat
3. **TRACKING** — Servo bergerak mengikuti api, lock target
4. **SPRAYING** — Pump aktif + solenoid terbuka, semprot air
5. **CLEAR** — Api padam, matikan pump, kembali ke IDLE

## Dataset
- Format: YOLO (.txt label per image)
- Capture via shutter.py, simpan ke datasets/
- Anotasi manual (tools: labelImg, CVAT, atau Roboflow)
- Training via train.ipynb

## Safety Notes
- Servo jangan dipaksa di luar 0-180°
- Pump + solenoid hanya aktif ketika lock stable (debounce)
- Frame skipping untuk performa di RPi 3B+ (proses setiap N frame)
