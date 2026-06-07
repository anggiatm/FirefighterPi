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
    return '<html><body style="margin:0;background:#000"><img src="/video_feed" style="width:100%"></body></html>'

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Stream: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
