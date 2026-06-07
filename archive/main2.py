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
CONF_THRESHOLD = 0.25          # Ambang batas kepercayaan deteksi (sesuaikan kebutuhan)
IMG_SIZE = 320                  # ukuran input YOLO
CAMERA_ID = 0                   # 0 = default camera

# =========================
# LOAD MODEL
# =========================
print("Loading ONNX model...")
# Memaksa ONNX Runtime menggunakan CPU secara optimal di Raspberry Pi
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
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
    boxes = []
    detections = outputs[0]

    # Menghilangkan batch dimension jika ada [1, 5, 8400] -> [5, 8400]
    if len(detections.shape) == 3:
        detections = detections[0]

    # Standar YOLOv8/v11 outputnya berbentuk (properti, jumlah_kandidat), misal (5, 2100)
    # Kita balik (Transpose) agar menjadi (jumlah_kandidat, properti) agar bisa di-loop per baris
    if detections.shape[0] < detections.shape[1]:
        detections = detections.T

    h_orig, w_orig = orig_shape

    for det in detections:
        # Format det: [x_center, y_center, width, height, class_score1, class_score2...]
        if len(det) >= 5:
            x, y, w, h = det[0:4]
            # Ambil nilai skor kelas tertinggi (indeks ke-4 jika hanya ada 1 kelas api)
            conf = float(det[4]) 
        else:
            continue

        # Lewati jika di bawah threshold
        if conf < CONF_THRESHOLD:
            continue

        # Hitung koordinat XYXY berdasarkan rasio ukuran gambar input YOLO (IMG_SIZE)
        # Catatan: Jika model mengekspor koordinat ternormalisasi (0-1), ganti IMG_SIZE di bawah dengan angka 1
        x1 = int(((x - w / 2) / IMG_SIZE) * w_orig)
        y1 = int(((y - h / 2) / IMG_SIZE) * h_orig)
        x2 = int(((x + w / 2) / IMG_SIZE) * w_orig)
        y2 = int(((y + h / 2) / IMG_SIZE) * h_orig)

        # Batasi koordinat agar tidak keluar dari pixel frame asli kamera
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_orig, x2)
        y2 = min(h_orig, y2)

        boxes.append([x1, y1, x2, y2, conf])

    return boxes

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf = box
        # Gambar kotak merah untuk objek api
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, f'FIRE {conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return frame

def generate_frames():
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_shape = frame.shape[:2] # Ambil (height, width) asli frame kamera

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

        # Overlay teks FPS di layar berwarna hijau
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # =========================
        # JPEG ENCODE
        # =========================
        # Kualitas diturunkan sedikit ke 70 agar beban streaming web di RPi lebih enteng
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
                img { border: 4px solid #ff4444; border-radius: 8px; margin-top: 20px; max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Real-time Fire Detection</h1>
            <p>Status: Running</p>
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
