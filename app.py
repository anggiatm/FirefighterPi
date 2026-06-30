from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import time
import math
import serial
import sys
import os
import json

# ==============================================================================
# 1. KONFIGURASI SISTEM
# ==============================================================================
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAMERA_ID = 0

_BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "fire_detector-3", "weights", "best.onnx")
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.40
IMG_SIZE = 320

CEILING_HEIGHT = 500.0
OFFSET_X = 0.0
OFFSET_Y = -30.0

# Titik sudut kalibrasi area lantai (gunakan calibration_server.py untuk mendapatkan nilai ini)
SRC_POINTS = np.array([
    [0, 0], [640, 0], [609, 480], [8, 480]
], dtype=np.float32)

# Ukuran fisik lapangan nyata (mm)
LEBAR_FISIK_ATAS   = 490.0
LEBAR_FISIK_BAWAH  = 500.0
TINGGI_FISIK_KIRI  = 270.0
TINGGI_FISIK_KANAN = 290.0

ALPHA_SMOOTHING = 0.25
STABLE_THRESHOLD_PX = 6.0

# ==============================================================================
# 2. PERSPECTIVE TRANSFORMATION & HOMOGRAPHY MATRIX
# ==============================================================================
DST_POINTS = np.array([
    [0.0, 0.0],
    [LEBAR_FISIK_ATAS, 0.0],
    [LEBAR_FISIK_BAWAH, TINGGI_FISIK_KANAN],
    [0.0, TINGGI_FISIK_KIRI]
], dtype=np.float32)

HOMOGRAPHY_MATRIX = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

CENTER_REAL_X = (LEBAR_FISIK_ATAS + LEBAR_FISIK_BAWAH) / 4.0
CENTER_REAL_Y = (TINGGI_FISIK_KIRI + TINGGI_FISIK_KANAN) / 4.0

# ==============================================================================
# 3. INISIALISASI PORT SERIAL KE ARDUINO
# ==============================================================================
def _init_serial():
    # Port candidates: Windows pakai COM*, Linux/Pi pakai /dev/tty*
    if sys.platform == "win32":
        candidates = [f"COM{i}" for i in range(3, 13)]
    else:
        candidates = ["/dev/ttyUSB0", "/dev/ttyS0", "/dev/ttyAMA0"]

    for port in candidates:
        try:
            conn = serial.Serial(port, 9600, timeout=1)
            print(f"Serial terhubung di {port}")
            return conn
        except Exception:
            continue

    print("Peringatan: Tidak ada port serial ditemukan. Mode simulasi (tanpa Arduino).")
    return None

arduino = _init_serial()

last_sent_pan = -1
last_sent_tilt = -1
last_sent_pump = -1

# ==============================================================================
# 4. GLOBAL STATE MACHINE & TELEMETRI
# ==============================================================================
current_pan = 90
current_tilt = 90
current_pump = 1  # M1 = Pompa MATI

telemetry_centroid_px = (0, 0)
telemetry_real_mm = (0.0, 0.0)

current_state = 0
state_timer = 0.0

smoothed_cx = None
smoothed_cy = None

# ==============================================================================
# 5. INITIALIZE MODEL AI & KAMERA
# ==============================================================================
print("Loading ONNX model...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = os.cpu_count() or 4
session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
print(f"Model loaded: {MODEL_PATH}")

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

# ==============================================================================
# 5b. SERVO CALIBRATION — load dari servo_calibration.json jika tersedia
# ==============================================================================
CALIB_JSON_PATH = os.path.join(_BASE_DIR, "servo_calibration.json")
_servo_calib: list | None = None

def _load_servo_calib() -> None:
    global _servo_calib
    if not os.path.exists(CALIB_JSON_PATH):
        print("Servo calib: tidak ada file kalibrasi, pakai math.")
        return
    with open(CALIB_JSON_PATH) as f:
        data = json.load(f)
    pts = data.get("calib_points", [])
    if len(pts) < 3:
        print(f"Servo calib: hanya {len(pts)} titik (butuh >= 3), fallback ke math.")
        return
    # Peringatan jika src_points JSON tidak cocok dengan SRC_POINTS di atas
    json_src = data.get("src_points")
    if json_src and json_src != SRC_POINTS.tolist():
        print("Servo calib: ⚠️  SRC_POINTS di JSON berbeda dari app.py — hasil kalibrasi mungkin tidak akurat.")
    _servo_calib = pts
    print(f"Servo calib loaded: {len(pts)} titik dari {CALIB_JSON_PATH}")

_load_servo_calib()

app = Flask(__name__)

# ==============================================================================
# 6. AI INFERENCE PROCESSING PIPELINE
# ==============================================================================
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.transpose(2, 0, 1) / 255.0
    img = img[np.newaxis, :, :, :].astype(np.float32)
    return img

def postprocess(outputs, orig_shape):
    raw_boxes = []
    confidences = []
    detections = outputs[0]

    if len(detections.shape) == 3:
        detections = detections[0]

    if detections.shape[0] < detections.shape[1]:
        detections = detections.T

    h_orig, w_orig = orig_shape

    for det in detections:
        if len(det) >= 5:
            x, y, w, h = det[0:4]
            conf = float(det[4])
        else:
            continue

        if conf < CONF_THRESHOLD:
            continue

        x1 = max(0, int(((x - w / 2) / IMG_SIZE) * w_orig))
        y1 = max(0, int(((y - h / 2) / IMG_SIZE) * h_orig))
        x2 = min(w_orig, int(((x + w / 2) / IMG_SIZE) * w_orig))
        y2 = min(h_orig, int(((y + h / 2) / IMG_SIZE) * h_orig))

        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(conf)

    final_boxes = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        if len(indices) > 0:
            for i in np.array(indices).flatten():
                x1, y1, w_b, h_b = raw_boxes[i]
                final_boxes.append([x1, y1, x1 + w_b, y1 + h_b, confidences[i]])

    return final_boxes

def send_serial_command(command_char, value):
    global arduino
    if arduino and arduino.is_open:
        try:
            arduino.write(f"{command_char}{value}".encode("utf-8"))
        except Exception as e:
            print(f"Gagal kirim serial {command_char}: {e}")

def _calculate_angles_idw(real_x: float, real_y: float, pts: list) -> tuple[int, int]:
    """Inverse Distance Weighting interpolation dari data servo_calibration.json."""
    xs    = np.array([p["real_x"] for p in pts], dtype=np.float64)
    ys    = np.array([p["real_y"] for p in pts], dtype=np.float64)
    pans  = np.array([p["pan"]    for p in pts], dtype=np.float64)
    tilts = np.array([p["tilt"]   for p in pts], dtype=np.float64)
    dists = np.sqrt((xs - real_x) ** 2 + (ys - real_y) ** 2)

    # Snap ke titik terdekat jika jarak < 5 mm
    exact = np.where(dists < 5.0)[0]
    if len(exact):
        i = exact[0]
        return int(pans[i]), int(tilts[i])

    weights  = 1.0 / (dists ** 2)
    pan_deg  = float(np.sum(weights * pans)  / np.sum(weights))
    tilt_deg = float(np.sum(weights * tilts) / np.sum(weights))
    return int(np.clip(pan_deg, 0, 180)), int(np.clip(tilt_deg, 0, 180))


def calculate_angles(real_x: float, real_y: float) -> tuple[int, int]:
    if _servo_calib is not None and len(_servo_calib) >= 3:
        return _calculate_angles_idw(real_x, real_y, _servo_calib)

    # Fallback: model trigonometri murni
    dx = (real_x - CENTER_REAL_X) + OFFSET_X
    dy = (CENTER_REAL_Y - real_y) + OFFSET_Y
    pan_deg  = 90.0 + math.degrees(math.atan2(dx, dy))
    tilt_deg = 90.0 + math.degrees(math.atan2(math.sqrt(dx**2 + dy**2), CEILING_HEIGHT))
    return int(max(0.0, min(180.0, pan_deg))), int(max(0.0, min(180.0, tilt_deg)))

def update_action_state_machine(target_boxes):
    global current_state, state_timer, current_pan, current_tilt, current_pump
    global smoothed_cx, smoothed_cy, telemetry_centroid_px, telemetry_real_mm
    global last_sent_pan, last_sent_tilt, last_sent_pump

    if len(target_boxes) == 0:
        if current_state != 0:
            print("--- KONDISI AMAN: Api Hilang. Kembali ke Standby dan Matikan Pompa ---")
        current_state = 0
        current_pan = 90
        current_tilt = 90
        current_pump = 1

        if last_sent_pan != current_pan:
            send_serial_command('P', current_pan)
            last_sent_pan = current_pan
        if last_sent_tilt != current_tilt:
            send_serial_command('T', current_tilt)
            last_sent_tilt = current_tilt
        if last_sent_pump != current_pump:
            send_serial_command('M', current_pump)
            last_sent_pump = current_pump

        smoothed_cx, smoothed_cy = None, None
        telemetry_centroid_px = (0, 0)
        telemetry_real_mm = (0.0, 0.0)
        return

    x1, y1, x2, y2, _ = target_boxes[0]
    raw_cx = float(x1 + (x2 - x1) / 2)
    raw_cy = float(y1 + (y2 - y1) / 2)

    if smoothed_cx is None:
        smoothed_cx, smoothed_cy = raw_cx, raw_cy
    else:
        smoothed_cx = ALPHA_SMOOTHING * raw_cx + (1.0 - ALPHA_SMOOTHING) * smoothed_cx  # type: ignore[operator]
        smoothed_cy = ALPHA_SMOOTHING * raw_cy + (1.0 - ALPHA_SMOOTHING) * smoothed_cy  # type: ignore[operator]

    telemetry_centroid_px = (int(smoothed_cx), int(smoothed_cy))

    px_point = np.array([[[smoothed_cx, smoothed_cy]]], dtype=np.float32)
    real_point = cv2.perspectiveTransform(px_point, HOMOGRAPHY_MATRIX)
    real_x = float(real_point[0][0][0])
    real_y = float(real_point[0][0][1])
    telemetry_real_mm = (real_x, real_y)

    now = time.time()

    if current_state == 0:
        print(f"-> TARGET TERDETEKSI: Centroid Px ({int(smoothed_cx)}, {int(smoothed_cy)})")
        current_state = 1
        state_timer = now

    elif current_state == 1:
        if math.sqrt((raw_cx - smoothed_cx)**2 + (raw_cy - smoothed_cy)**2) > STABLE_THRESHOLD_PX:
            state_timer = now
        elif (now - state_timer) >= 0.5:
            current_pan, current_tilt = calculate_angles(real_x, real_y)
            if last_sent_pan != current_pan:
                send_serial_command('P', current_pan)
                last_sent_pan = current_pan
            if last_sent_tilt != current_tilt:
                send_serial_command('T', current_tilt)
                last_sent_tilt = current_tilt
            print(f"--- TARGET STABIL: Servo [P{current_pan}, T{current_tilt}] ---")
            current_state = 2
            state_timer = now

    elif current_state == 2:
        if (now - state_timer) >= 1.0:
            current_pump = 0
            if last_sent_pump != current_pump:
                send_serial_command('M', current_pump)
                last_sent_pump = current_pump
            print("--- SERVO MANTAP: Pompa NYALA [M0] ---")
            current_state = 3

    elif current_state == 3:
        current_pan, current_tilt = calculate_angles(real_x, real_y)
        if last_sent_pan != current_pan:
            send_serial_command('P', current_pan)
            last_sent_pan = current_pan
        if last_sent_tilt != current_tilt:
            send_serial_command('T', current_tilt)
            last_sent_tilt = current_tilt

def generate_frames():
    prev_time = time.time()
    state_labels = {
        0: "IDLE (Searching)",
        1: "LOCKING (Stabilizing)",
        2: "AIMING (Servo Move)",
        3: "EXTINGUISHING (Pump ON)"
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img_input = preprocess_frame(frame)
        outputs = session.run(None, {input_name: img_input})
        boxes = postprocess(outputs, frame.shape[:2])

        update_action_state_machine(boxes)

        pts = SRC_POINTS.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f'FIRE {conf:.2f}', (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if current_state > 0:
            cx, cy = telemetry_centroid_px
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        pump_label = "NYALA" if current_pump == 0 else "MATI"
        cv2.rectangle(frame, (5, 5), (320, 115), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS    : {fps:.1f}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"STATE  : {state_labels[current_state]}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"CENTROID: X:{telemetry_centroid_px[0]}, Y:{telemetry_centroid_px[1]} px", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(frame, f"REAL MM : X:{telemetry_real_mm[0]:.1f}, Y:{telemetry_real_mm[1]:.1f}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(frame, f"SERVO  : PAN: {current_pan}, TILT: {current_tilt}", (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(frame, f"RELAY  : POMPA = {pump_label}", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255) if current_pump == 0 else (128, 128, 128), 1)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Smart Firefighter Robot System</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: #fff; margin:0; padding:20px; }
                h1 { color: #ff3333; margin-bottom: 5px; }
                .container { display: flex; justify-content: center; margin-top: 15px; }
                img { border: 2px solid #ff3333; border-radius: 6px; box-shadow: 0 0 15px rgba(255,0,0,0.4); }
                .info { text-align: left; margin: 15px auto; width: 640px; background: #2a2a2a; padding: 12px; border-radius: 6px; font-size: 13px; color: #ccc;}
            </style>
        </head>
        <body>
            <h1>Smart Firefighter Robot System</h1>
            <p>Deteksi Api · Tracking Centroid · Kontrol Serial Arduino</p>
            <div class="container"><img src="/video_feed"></div>
            <div class="info">Gunakan <b>calibration_server.py</b> (port 5001) untuk mengatur ulang SRC_POINTS sesuai posisi kamera.</div>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(f"Platform  : {sys.platform}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Serial    : {'terhubung' if arduino else 'tidak terhubung (mode simulasi)'}")
    print("Server    : http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
