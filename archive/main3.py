# main.py
import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, Response
import time

# =========================
# CONFIG
# =========================
MODEL_PATH = "/home/pi/firefighter/models/fire_detector-3/weights/best.onnx"   # path ke ONNX model
CONF_THRESHOLD = 0.25          # Ambang batas kepercayaan deteksi
NMS_THRESHOLD = 0.40           # Batas overlap NMS (semakin kecil, semakin ketat menyaring kotak overlap)
IMG_SIZE = 320                  # ukuran input YOLO
CAMERA_ID = 0                   # 0 = default camera

# =========================
# LOAD MODEL
# =========================
print("Loading ONNX model...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # Mengoptimalkan thread CPU di Raspberry Pi
session = ort.InferenceSession(MODEL_PATH, opts)
input_name = session.get_inputs()[0].name
print("Model loaded:", MODEL_PATH)

# =========================
# INIT CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# Set resolusi kamera lebih rendah agar FPS lebih ringan di RPi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

def preprocess_frame(frame):
    # Konversi BGR bawaan OpenCV ke RGB standar YOLO
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # HWC -> CHW dan Normalisasi /255.0
    img = img.transpose(2, 0, 1) / 255.0   
    img = img[np.newaxis, :, :, :].astype(np.float32)
    return img

def postprocess(outputs, orig_shape):
    raw_boxes = []
    confidences = []
    detections = outputs[0]

    # Menghilangkan batch dimension jika ada
    if len(detections.shape) == 3:
        detections = detections[0]

    # Transpose jika format kolom dan baris terbalik (Standar YOLOv8/v11)
    if detections.shape[0] < detections.shape[1]:
        detections = detections.T

    h_orig, w_orig = orig_shape

    for det in detections:
        if len(det) >= 5:
            x, y, w, h = det[0:4]
            conf = float(det[4]) # Ambil nilai skor kelas pertama (api)
        else:
            continue

        # Lewati jika di bawah threshold keyakinan awal
        if conf < CONF_THRESHOLD:
            continue

        # Hitung koordinat piksel awal berdasarkan ukuran input model (IMG_SIZE)
        x1 = int(((x - w / 2) / IMG_SIZE) * w_orig)
        y1 = int(((y - h / 2) / IMG_SIZE) * h_orig)
        x2 = int(((x + w / 2) / IMG_SIZE) * w_orig)
        y2 = int(((y + h / 2) / IMG_SIZE) * h_orig)

        # Batasi koordinat agar tidak keluar dari pixel frame asli kamera
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)

        # Perhitungan lebar dan tinggi kotak untuk input fungsi NMS OpenCV
        width_box = x2 - x1
        height_box = y2 - y1

        raw_boxes.append([x1, y1, width_box, height_box])
        confidences.append(conf)

    # ==========================================
    # PROSES NMS (Non-Maximum Suppression)
    # ==========================================
    indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    final_boxes = []
    if len(indices) > 0:
        # Menangani kompabilitas indeks output NMS OpenCV di berbagai versi
        for i in indices.flatten():
            x1, y1, w_b, h_b = raw_boxes[i]
            conf = confidences[i]
            # Kembalikan ke format x1, y1, x2, y2 untuk digambar ke frame
            final_boxes.append([x1, y1, x1 + w_b, y1 + h_b, conf])

    return final_boxes

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf = box
        # Menggunakan ketebalan 1 agar bounding box terlihat tipis dan rapi
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # Teks dengan skala kecil 0.4 dan ketebalan garis 1
        cv2.putText(frame, f'FIRE {conf:.2f}', (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame

def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_shape = frame.shape[:2] # Ambil (height, width) asli kamera

        # =========================
        # PREPROCESS
        # =========================
        img_input = preprocess_frame(frame)

        # =========================
        # INFERENCE
        # =========================
        outputs = session.run(None, {input_name: img_input})

        # =========================
        # POSTPROCESS
        # =========================
        boxes = postprocess(outputs, orig_shape)

        # =========================
        # DRAW BOXES
        # =========================
        frame = draw_boxes(frame, boxes)

        # =========================
        # FPS CALCULATION
        # =========================
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        # Teks indikator FPS tipis berwarna hijau di pojok kiri atas
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1
        )

        # =========================
        # JPEG ENCODE
        # =========================
        ret, buffer = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        frame_bytes = buffer.tobytes()

        # =========================
        # STREAM
        # =========================
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Fire Detection Server</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #222; color: #fff; }
                img { border: 2px solid #ff4444; border-radius: 6px; margin-top: 20px; max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Real-time Fire Detection</h1>
            <p>Status: Active (NMS Filtered)</p>
            <img src="/video_feed">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    print("Starting Flask server at http://0.0.0")
    app.run(host='0.0.0.0', port=5000, debug=False)
