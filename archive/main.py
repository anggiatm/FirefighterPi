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
CONF_THRESHOLD = 0.1           # confidence threshold
IMG_SIZE = 320                  # ukuran input YOLO
CAMERA_ID = 0                   # 0 = default camera

# =========================
# LOAD MODEL
# =========================
print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("Model loaded:", MODEL_PATH)

# =========================
# INIT CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.transpose(2,0,1) / 255.0   # HWC -> CHW
    img = img[np.newaxis, :, :, :].astype(np.float32)
    return img

def postprocess(outputs, orig_shape):

    boxes = []

    # ambil output pertama
    detections = outputs[0]

    # jika ada batch dimension
    if len(detections.shape) == 3:
        detections = detections[0]

    for det in detections:

        # support output 5 atau 6 values
        if len(det) == 6:
            x, y, w, h, conf, cls = det
        elif len(det) == 5:
            x, y, w, h, conf = det
            cls = 0
        else:
            continue

        conf = float(conf)

        if conf < CONF_THRESHOLD:
            continue

        # convert normalized xywh -> xyxy
        x1 = int((x - w / 2) * orig_shape[1])
        y1 = int((y - h / 2) * orig_shape[0])
        x2 = int((x + w / 2) * orig_shape[1])
        y2 = int((y + h / 2) * orig_shape[0])

        boxes.append([x1, y1, x2, y2, conf])

    return boxes

def draw_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2, conf = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, f'fire {conf:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    return frame
def generate_frames():

    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        orig_shape = frame.shape[:2]

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

        # overlay FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
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
        <head><title>Fire Detection</title></head>
        <body>
            <h1>Fire Detection Raspberry Pi</h1>
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
    print("Starting Flask server at http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
