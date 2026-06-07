import cv2
from flask import Flask, Response

CAMERA_ID = 0
CAM_WIDTH  = 640
CAM_HEIGHT = 480

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")
print(f"Kamera terkoneksi: ID={CAMERA_ID}, resolusi={CAM_WIDTH}x{CAM_HEIGHT}")

app = Flask(__name__)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Camera Test Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ background: #111; color: #fff; font-family: Arial, sans-serif;
                display: flex; flex-direction: column; align-items: center;
                min-height: 100vh; padding: 20px; }}
        h1 {{ font-size: 1.2em; color: #0cf; margin-bottom: 4px; letter-spacing: 1px; }}
        .badge {{ font-size: 12px; color: #aaa; margin-bottom: 16px; }}
        .badge span {{ background: #222; border: 1px solid #333; border-radius: 4px;
                       padding: 2px 8px; margin: 0 3px; }}
        .stream-wrap {{ position: relative; border: 2px solid #0cf;
                        border-radius: 8px; overflow: hidden;
                        box-shadow: 0 0 20px rgba(0,200,255,0.2); }}
        img {{ display: block; max-width: 100%; width: 640px; }}
        .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                background: #0f0; margin-right: 6px;
                animation: blink 1.2s infinite; }}
        @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.2}} }}
        .status {{ margin-top: 12px; font-size: 13px; color: #aaa; }}
    </style>
</head>
<body>
    <h1>📷 Camera Test Stream</h1>
    <div class="badge">
        <span>Camera ID: {CAMERA_ID}</span>
        <span>{CAM_WIDTH} × {CAM_HEIGHT}</span>
    </div>
    <div class="stream-wrap">
        <img src="/video_feed" alt="stream">
    </div>
    <div class="status"><span class="dot"></span>Live</div>
</body>
</html>"""

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Stream: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
