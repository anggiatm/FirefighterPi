import os
import cv2
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string

SAVE_DIR = "datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lock = threading.Lock()
latest_frame = None


def capture_thread():
    global latest_frame
    while True:
        ret, frame = cam.read()
        if ret:
            with lock:
                latest_frame = frame.copy()


threading.Thread(target=capture_thread, daemon=True).start()


def gen_frames():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/")
def index():
    count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
    return render_template_string(HTML_PAGE, count=count, save_dir=SAVE_DIR)


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture", methods=["POST"])
def capture():
    with lock:
        frame = latest_frame
    if frame is None:
        return jsonify({"ok": False, "error": "Kamera belum siap"}), 503

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, frame)

    count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
    return jsonify({"ok": True, "filename": filename, "total": count})


@app.route("/count")
def count():
    n = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
    return jsonify({"total": n})


HTML_PAGE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Capture</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #111; color: #eee; font-family: sans-serif; display: flex;
         flex-direction: column; align-items: center; min-height: 100vh; padding: 24px 16px; }
  h1 { font-size: 1.3rem; margin-bottom: 16px; letter-spacing: 1px; color: #fff; }
  #preview { border: 2px solid #333; border-radius: 8px; width: 100%; max-width: 640px; display: block; }
  #flash { position: fixed; inset: 0; background: white; opacity: 0;
           pointer-events: none; transition: opacity 0.05s; z-index: 10; }
  #flash.active { opacity: 0.6; }
  .controls { margin-top: 20px; display: flex; flex-direction: column; align-items: center; gap: 12px; width: 100%; max-width: 640px; }
  #btn-shutter { width: 100%; padding: 18px; font-size: 1.2rem; font-weight: bold;
                 background: #e74c3c; color: white; border: none; border-radius: 10px;
                 cursor: pointer; letter-spacing: 1px; transition: background 0.15s; }
  #btn-shutter:hover { background: #c0392b; }
  #btn-shutter:active { background: #922b21; }
  #btn-shutter:disabled { background: #555; cursor: not-allowed; }
  #status { font-size: 0.95rem; color: #aaa; min-height: 1.2em; }
  #counter { font-size: 1rem; color: #2ecc71; }
  #log { width: 100%; max-width: 640px; margin-top: 16px; background: #1a1a1a;
         border: 1px solid #333; border-radius: 6px; padding: 10px;
         font-size: 0.8rem; color: #888; max-height: 150px; overflow-y: auto; font-family: monospace; }
  #log p { margin: 2px 0; }
  #log p.ok { color: #2ecc71; }
  #log p.err { color: #e74c3c; }
</style>
</head>
<body>
<div id="flash"></div>
<h1>Dataset Capture</h1>
<img id="preview" src="/stream" alt="Camera Stream">
<div class="controls">
  <button id="btn-shutter">CAPTURE</button>
  <div id="counter">Total tersimpan: <span id="total">{{ count }}</span> gambar</div>
  <div id="status">Siap — folder: {{ save_dir }}/</div>
</div>
<div id="log"></div>

<script>
const btn = document.getElementById('btn-shutter');
const totalEl = document.getElementById('total');
const statusEl = document.getElementById('status');
const flash = document.getElementById('flash');
const log = document.getElementById('log');

function addLog(msg, type='ok') {
  const p = document.createElement('p');
  p.className = type;
  p.textContent = new Date().toLocaleTimeString() + ' — ' + msg;
  log.prepend(p);
  if (log.children.length > 50) log.lastChild.remove();
}

async function doCapture() {
  btn.disabled = true;
  flash.classList.add('active');
  setTimeout(() => flash.classList.remove('active'), 150);

  try {
    const res = await fetch('/capture', { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      totalEl.textContent = data.total;
      statusEl.textContent = 'Tersimpan: ' + data.filename;
      addLog('Saved: ' + data.filename);
    } else {
      statusEl.textContent = 'Error: ' + data.error;
      addLog('Error: ' + data.error, 'err');
    }
  } catch(e) {
    statusEl.textContent = 'Koneksi gagal';
    addLog('Koneksi gagal', 'err');
  }
  btn.disabled = false;
}

btn.addEventListener('click', doCapture);

document.addEventListener('keydown', (e) => {
  if ((e.code === 'Space' || e.code === 'Enter') && !btn.disabled) {
    e.preventDefault();
    doCapture();
  }
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    connected = cam.isOpened()
    print(f"Kamera  : {'terhubung' if connected else 'TIDAK DITEMUKAN'}")
    print(f"Simpan  : {SAVE_DIR}/")
    print(f"Akses   : http://<ip-pi>:5002")
    if not connected:
        print("WARNING: Kamera tidak terbuka. Pastikan webcam terpasang.")
    app.run(host="0.0.0.0", port=5002, threaded=True)
