# main.py - BAGIAN 1
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import time
import math
import serial  # Library untuk komunikasi serial TX/RX

# ==============================================================================
# 1. KONFIGURASI SISTEM (Diadopsi penuh dari main4.py Anda)
# ==============================================================================
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAMERA_ID = 0

MODEL_PATH = "/home/pi/firefighter/models/fire_detector-3/weights/best.onnx"
CONF_THRESHOLD = 0.1
NMS_THRESHOLD = 0.40
IMG_SIZE = 320

CEILING_HEIGHT = 500.0
OFFSET_X = 0.0
OFFSET_Y = -30.0

# Titik sudut kalibrasi area lantai pilihan Anda
SRC_POINTS = np.float32([
    [0, 0], [640, 0], [609, 480], [8, 480]
])

# Ukuran fisik lapangan nyata dari meteran Anda (mm)
LEBAR_FISIK_ATAS  = 490.0
LEBAR_FISIK_BAWAH = 500.0
TINGGI_FISIK_KIRI  = 270.0
TINGGI_FISIK_KANAN = 290.0

ALPHA_SMOOTHING = 0.25
STABLE_THRESHOLD_PX = 6.0

# ==============================================================================
# 2. PERSPECTIVE TRANSFORMATION & HOMOGRAPHY MATRIX
# ==============================================================================
DST_POINTS = np.float32([
    [0.0, 0.0],                                 
    [LEBAR_FISIK_ATAS, 0.0],                    
    [LEBAR_FISIK_BAWAH, TINGGI_FISIK_KANAN],    
    [0.0, TINGGI_FISIK_KIRI]                    
])

HOMOGRAPHY_MATRIX = cv2.getPerspectiveTransform(SRC_POINTS, DST_POINTS)

CENTER_REAL_X = (LEBAR_FISIK_ATAS + LEBAR_FISIK_BAWAH) / 4.0
CENTER_REAL_Y = (TINGGI_FISIK_KIRI + TINGGI_FISIK_KANAN) / 4.0

# ==============================================================================
# 3. INISIALISASI PORT SERIAL TX/RX TO ARDUINO UNO (9600 Baudrate)
# ==============================================================================
try:
    arduino = serial.Serial('/dev/ttyS0', 9600, timeout=1)
except Exception as e:
    print("Mencoba jalur port serial alternatif /dev/ttyAMA0...")
    try:
        arduino = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    except Exception as err:
        print("Peringatan: Komunikasi serial ke Arduino dinonaktifkan (Port tidak ditemukan).")
        arduino = None

# Variable tracker agar tidak mengirimkan karakter duplikat ke serial buffer
last_sent_pan = -1
last_sent_tilt = -1
last_sent_pump = -1

# ==============================================================================
# 4. GLOBAL STATE MACHINE & TELEMETRI VARIABLE
# ==============================================================================
current_pan = 90
current_tilt = 90
current_pump = 1  # Inisialisasi awal ke 1 (Sesuai kode Arduino: M1 = Pompa Mati)

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
opts.intra_op_num_threads = 4
session = ort.InferenceSession(MODEL_PATH, opts)
input_name = session.get_inputs()[0].name
print("Model loaded:", MODEL_PATH)

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

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

        x1 = int(((x - w / 2) / IMG_SIZE) * w_orig)
        y1 = int(((y - h / 2) / IMG_SIZE) * h_orig)
        x2 = int(((x + w / 2) / IMG_SIZE) * w_orig)
        y2 = int(((y + h / 2) / IMG_SIZE) * h_orig)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)

        raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(conf)

    indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w_b, h_b = raw_boxes[i]
            conf = confidences[i]
            final_boxes.append([x1, y1, x1 + w_b, y1 + h_b, conf])

    return final_boxes
# main.py - BAGIAN 2

def send_serial_command(command_char, value):
    """
    Mengirim data string perintah ke Arduino (Format: P90, T120, M0)
    """
    global arduino
    if arduino and arduino.is_open:
        try:
            payload = f"{command_char}{value}"
            arduino.write(payload.encode('utf-8'))
        except Exception as e:
            print(f"Gagal kirim data serial {command_char}: {e}")

def calculate_angles(real_x, real_y):
    dx = (real_x - CENTER_REAL_X) + OFFSET_X
    dy = (CENTER_REAL_Y - real_y) + OFFSET_Y

    pan_rad = math.atan2(dx, dy)
    pan_deg = 90.0 + math.degrees(pan_rad)

    ground_distance = math.sqrt(dx**2 + dy**2)
    tilt_rad = math.atan2(ground_distance, CEILING_HEIGHT)
    tilt_deg = 90.0 - math.degrees(tilt_rad)

    pan_deg = max(0.0, min(180.0, pan_deg))
    tilt_deg = max(0.0, min(180.0, tilt_deg))

    return int(pan_deg), int(tilt_deg)

def update_action_state_machine(target_boxes):
    global current_state, state_timer, current_pan, current_tilt, current_pump
    global smoothed_cx, smoothed_cy, telemetry_centroid_px, telemetry_real_mm
    global last_sent_pan, last_sent_tilt, last_sent_pump

    # Jika tidak ada api, posisikan ke Standby (90,90) dan Matikan Pompa (M1)
    if len(target_boxes) == 0:
        if current_state != 0:
            print("--- KONDISI AMAN: Api Hilang. Kembali ke Standby dan Matikan Pompa ---")
        current_state = 0
        current_pan = 90
        current_tilt = 90
        current_pump = 1  # M1 berdasarkan skrip Arduino Anda = Pompa MATI
        
        # Kirim data reset ke Arduino jika ada perubahan status
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

    box = target_boxes[0]
    x1, y1, x2, y2, _ = box

    raw_cx = float(x1 + (x2 - x1) / 2)
    raw_cy = float(y1 + (y2 - y1) / 2)

    if smoothed_cx is None or smoothed_cy is None:
        smoothed_cx = raw_cx
        smoothed_cy = raw_cy
    else:
        smoothed_cx = (ALPHA_SMOOTHING * raw_cx) + ((1.0 - ALPHA_SMOOTHING) * smoothed_cx)
        smoothed_cy = (ALPHA_SMOOTHING * raw_cy) + ((1.0 - ALPHA_SMOOTHING) * smoothed_cy)

    telemetry_centroid_px = (int(smoothed_cx), int(smoothed_cy))

    px_point = np.array([[[smoothed_cx, smoothed_cy]]], dtype=np.float32)
    real_point = cv2.perspectiveTransform(px_point, HOMOGRAPHY_MATRIX)
    real_x = float(real_point[0][0][0])
    real_y = float(real_point[0][0][1])
    telemetry_real_mm = (real_x, real_y)

    now = time.time()

    # --- STATE 0: BARU MENEMUKAN TARGET ---
    if current_state == 0:
        print(f"-> TARGET TERDETEKSI: Centroid Px ({int(smoothed_cx)}, {int(smoothed_cy)})")
        current_state = 1
        state_timer = now

    # --- STATE 1: LOCKING (Menunggu Stabil 0.5 Detik) ---
    elif current_state == 1:
        distance_movement = math.sqrt((raw_cx - smoothed_cx)**2 + (raw_cy - smoothed_cy)**2)

        if distance_movement > STABLE_THRESHOLD_PX:
            state_timer = now

        elif (now - state_timer) >= 0.5:
            target_pan, target_tilt = calculate_angles(real_x, real_y)
            current_pan = target_pan
            current_tilt = target_tilt
            
            # Kirim data gerak ke Servo Arduino via Serial
            if last_sent_pan != current_pan:
                send_serial_command('P', current_pan)
                last_sent_pan = current_pan
            if last_sent_tilt != current_tilt:
                send_serial_command('T', current_tilt)
                last_sent_tilt = current_tilt
                
            print(f"--- TARGET STABIL: Mengirim Perintah Servo [P{current_pan}, T{current_tilt}] ---")
            current_state = 2
            state_timer = now

    # --- STATE 2: AIMING (Menunggu Pergerakan Mekanik Servo 1.0 Detik) ---
    elif current_state == 2:
        if (now - state_timer) >= 1.0:
            current_pump = 0  # M0 berdasarkan skrip Arduino Anda = Pompa NYALA
            if last_sent_pump != current_pump:
                send_serial_command('M', current_pump)
                last_sent_pump = current_pump
            print("--- SERVO MANTAP: Mengirim Perintah Nyala Pompa! [M0] ---")
            current_state = 3

    # --- STATE 3: EXTINGUISHING (Pompa Aktif & Tracking Dinamis) ---
    elif current_state == 3:
        target_pan, target_tilt = calculate_angles(real_x, real_y)
        current_pan = target_pan
        current_tilt = target_tilt
        
        # Kirim update sudut berkelanjutan jika api bergeser lambat saat disiram
        if last_sent_pan != current_pan:
            send_serial_command('P', current_pan)
            last_sent_pan = current_pan
        if last_sent_tilt != current_tilt:
            send_serial_command('T', current_tilt)
            last_sent_tilt = current_tilt

def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_shape = frame.shape[:2]

        img_input = preprocess_frame(frame)
        outputs = session.run(None, {input_name: img_input})
        boxes = postprocess(outputs, orig_shape)

        update_action_state_machine(boxes)

        # Poligon Kalibrasi Kuning
        pts = SRC_POINTS.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 1)

        # Bounding Box Merah AI
        for box in boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(frame, f'FIRE {conf:.2f}', (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Crosshair Biru Centroid
        if current_state > 0:
            cx, cy = telemetry_centroid_px
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (255, 0, 0), 1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (255, 0, 0), 1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        state_labels = {0: "IDLE (Searching)", 1: "LOCKING (Stabilizing)", 2: "AIMING (Servo Move)", 3: "EXTINGUISHING (Pump ON)"}

        # Menampilkan status telemetri terintegrasi di monitor video web
        cv2.rectangle(frame, (5, 5), (320, 115), (0, 0, 0), -1)
        pump_label = "NYALA" if current_pump == 0 else "MATI"
        
        cv2.putText(frame, f"FPS    : {fps:.1f}", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"STATE  : {state_labels[current_state]}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"CENTROID: X:{telemetry_centroid_px[0]}, Y:{telemetry_centroid_px[1]} px", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(frame, f"REAL MM : X:{telemetry_real_mm[0]:.1f}, Y:{telemetry_real_mm[1]:.1f}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(frame, f"SERVO  : PAN: {current_pan}, TILT: {current_tilt}", (12, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(frame, f"RELAY  : POMPA = {pump_label}", (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255) if current_pump == 0 else (128, 128, 128), 1)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Advanced Fire Tracking System</title>
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
            <p>Sistem Deteksi, Tracking Centroid, dan Kontrol Aktif Serial TX/RX to Arduino Uno</p>
            <div class="container"><img src="/video_feed"></div>
            <div class="info">Kalibrasi: SRC_POINTS sudah menggunakan koordinat custom 480p yang Anda tentukan di lapangan.</div>
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
