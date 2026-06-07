import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import json
import sys
import os
import threading
import time
import serial
from datetime import datetime

# ==============================================================================
# KONFIGURASI KAMERA
# ==============================================================================
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAMERA_ID  = 0

# ==============================================================================
# KONSTANTA FISIK LANTAI — must match app.py
# ==============================================================================
LEBAR_FISIK_ATAS   = 490.0
LEBAR_FISIK_BAWAH  = 500.0
TINGGI_FISIK_KIRI  = 270.0
TINGGI_FISIK_KANAN = 290.0
CEILING_HEIGHT     = 500.0

CALIB_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_calibration.json")

# ==============================================================================
# STATE MODE A — Homography Calibration
# ==============================================================================
# Urutan: Kiri-Atas, Kanan-Atas, Kanan-Bawah, Kiri-Bawah
src_points = [
    [100, 100], [540, 100], [580, 380], [60, 380]
]

# ==============================================================================
# STATE MODE B — Servo Calibration
# ==============================================================================
calib_pan    = 90
calib_tilt   = 90
calib_points = []   # list of {pan, tilt, real_x, real_y}

arduino_conn  = None
_arduino_lock = threading.Lock()

# ==============================================================================
# HELPERS
# ==============================================================================
def _get_serial():
    global arduino_conn
    with _arduino_lock:
        if arduino_conn and arduino_conn.is_open:
            return arduino_conn
        candidates = [f"COM{i}" for i in range(3, 13)] if sys.platform == "win32" \
                     else ["/dev/ttyS0", "/dev/ttyAMA0", "/dev/ttyUSB0"]
        for port in candidates:
            try:
                arduino_conn = serial.Serial(port, 9600, timeout=1)
                print(f"[Calib] Serial terhubung di {port}")
                return arduino_conn
            except Exception:
                continue
        print("[Calib] Tidak ada port serial ditemukan.")
        return None


def _build_homography():
    if len(src_points) < 4 or src_points[0] == [0, 0]:
        return None
    src = np.array(src_points, dtype=np.float32)
    dst = np.array([
        [0.0, 0.0],
        [LEBAR_FISIK_ATAS, 0.0],
        [LEBAR_FISIK_BAWAH, TINGGI_FISIK_KANAN],
        [0.0, TINGGI_FISIK_KIRI]
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_mm(px_x, px_y):
    H = _build_homography()
    if H is None:
        return None, None
    pt = np.array([[[float(px_x), float(px_y)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    real_x = float(out[0][0][0])
    real_y = float(out[0][0][1])
    # Validasi dalam area lantai
    max_x = max(LEBAR_FISIK_ATAS, LEBAR_FISIK_BAWAH)
    max_y = max(TINGGI_FISIK_KIRI, TINGGI_FISIK_KANAN)
    if real_x < -50 or real_x > max_x + 50 or real_y < -50 or real_y > max_y + 50:
        return None, None
    return real_x, real_y


# ==============================================================================
# KAMERA
# ==============================================================================
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

app = Flask(__name__)


def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Gambar polygon kalibrasi homography (Mode A)
        pts = np.array(src_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
        for idx, pt in enumerate(src_points):
            cv2.circle(frame, (pt[0], pt[1]), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(idx + 1), (pt[0] + 8, pt[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Overlay info servo (Mode B)
        cv2.rectangle(frame, (5, 5), (220, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"SERVO: P{calib_pan} T{calib_tilt}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ==============================================================================
# ROUTES — MODE A (existing)
# ==============================================================================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_points')
def get_points():
    return jsonify({"points": src_points})


@app.route('/update_point', methods=['POST'])
def update_point():
    data = request.get_json()
    idx = data['index']
    src_points[idx] = [data['x'], data['y']]
    return jsonify({"status": "success"})


@app.route('/reset_points', methods=['POST'])
def reset_points():
    global src_points
    src_points = [[0, 0], [0, 0], [0, 0], [0, 0]]
    return jsonify({"status": "success"})


# ==============================================================================
# ROUTES — MODE B (servo calibration)
# ==============================================================================
@app.route('/calib/set_servo', methods=['POST'])
def calib_set_servo():
    global calib_pan, calib_tilt
    data = request.get_json()
    pan  = int(data.get('pan',  90))
    tilt = int(data.get('tilt', 90))
    pan  = max(0, min(180, pan))
    tilt = max(0, min(180, tilt))
    calib_pan  = pan
    calib_tilt = tilt

    conn = _get_serial()
    if conn:
        try:
            conn.write(f"P{pan}".encode('utf-8'))
            time.sleep(0.02)
            conn.write(f"T{tilt}".encode('utf-8'))
        except Exception as e:
            return jsonify({"status": "serial_error", "error": str(e)}), 500

    return jsonify({"status": "ok", "pan": pan, "tilt": tilt,
                    "serial": "connected" if conn else "disconnected"})


@app.route('/calib/record_click', methods=['POST'])
def calib_record_click():
    data  = request.get_json()
    px_x  = float(data['px_x'])
    px_y  = float(data['px_y'])
    real_x, real_y = pixel_to_mm(px_x, px_y)

    if real_x is None or real_y is None:
        return jsonify({"status": "error", "message": "Klik di luar area kalibrasi lantai"}), 400

    point = {
        "pan":    calib_pan,
        "tilt":   calib_tilt,
        "real_x": round(real_x, 1),
        "real_y": round(real_y, 1)
    }
    calib_points.append(point)
    return jsonify({"status": "ok", "point": point, "total": len(calib_points)})


@app.route('/calib/save', methods=['POST'])
def calib_save():
    payload = {
        "version": 1,
        "created": datetime.now().isoformat(timespec='seconds'),
        "src_points": src_points,
        "floor_mm": {
            "lebar_fisik_atas":   LEBAR_FISIK_ATAS,
            "lebar_fisik_bawah":  LEBAR_FISIK_BAWAH,
            "tinggi_fisik_kiri":  TINGGI_FISIK_KIRI,
            "tinggi_fisik_kanan": TINGGI_FISIK_KANAN
        },
        "ceiling_height": CEILING_HEIGHT,
        "calib_points": calib_points
    }
    with open(CALIB_JSON_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    return jsonify({"status": "saved", "path": CALIB_JSON_PATH, "count": len(calib_points)})


@app.route('/calib/load', methods=['GET'])
def calib_load():
    global calib_points
    if not os.path.exists(CALIB_JSON_PATH):
        return jsonify({"status": "not_found", "calib_points": []})
    with open(CALIB_JSON_PATH) as f:
        data = json.load(f)
    calib_points = data.get("calib_points", [])
    return jsonify({"status": "loaded", "count": len(calib_points), "calib_points": calib_points})


@app.route('/calib/undo', methods=['POST'])
def calib_undo():
    if calib_points:
        removed = calib_points.pop()
        return jsonify({"status": "ok", "removed": removed, "remaining": len(calib_points)})
    return jsonify({"status": "empty"})


@app.route('/calib/get_points', methods=['GET'])
def calib_get_points():
    return jsonify({"calib_points": calib_points})


# ==============================================================================
# MAIN HTML
# ==============================================================================
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Firefighter Calibration</title>
        <style>
            * { box-sizing: border-box; }
            body { font-family: Arial, sans-serif; text-align: center; background: #1a1a1a; color: #fff; margin:0; padding:16px; }
            h1 { color: #ffff00; margin-bottom: 4px; font-size: 1.3em; }

            /* Tabs */
            .tabs { display: flex; justify-content: center; gap: 8px; margin: 12px 0; }
            .tab-btn { padding: 8px 24px; border: 2px solid #555; background: #2a2a2a; color: #aaa;
                       border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: bold; }
            .tab-btn.active { border-color: #ffff00; color: #ffff00; background: #333; }

            /* Panels */
            .panel { display: none; }
            .panel.active { display: block; }

            /* Video */
            .video-wrap { position: relative; display: inline-block; cursor: crosshair; margin-top: 10px; }
            #stream-img { border: 2px solid #ffff00; border-radius: 4px; display: block; }

            /* Mode A */
            .info { text-align: left; margin: 12px auto; width: 640px; background: #2a2a2a;
                    padding: 12px; border-radius: 6px; font-size: 13px; line-height: 1.6; }
            pre { color: #0f0; margin: 0; font-size: 12px; }

            /* Mode B controls */
            .ctrl-box { width: 640px; margin: 12px auto; background: #2a2a2a; padding: 14px; border-radius: 6px; text-align: left; }
            .ctrl-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
            .ctrl-row label { width: 40px; font-weight: bold; color: #0cf; }
            input[type=range] { flex: 1; }
            .val-display { width: 36px; text-align: right; font-weight: bold; color: #fff; }
            .quick-btns { display: flex; gap: 6px; margin: 8px 0; }
            .quick-btn { padding: 4px 10px; background: #444; border: 1px solid #666; color: #fff;
                         border-radius: 3px; cursor: pointer; font-size: 12px; }
            .quick-btn:hover { background: #555; }
            .send-btn { width: 100%; padding: 10px; background: #0077cc; border: none; color: #fff;
                        font-size: 14px; font-weight: bold; border-radius: 4px; cursor: pointer; margin-top: 6px; }
            .send-btn:hover { background: #0099ff; }
            .send-btn:disabled { background: #444; color: #888; cursor: not-allowed; }
            .countdown { text-align: center; font-size: 13px; color: #ffa; margin: 6px 0; min-height: 20px; }

            /* Warning banner */
            .warning { background: #5a2000; border: 1px solid #f80; color: #fda; padding: 8px 12px;
                       border-radius: 4px; margin: 8px auto; width: 640px; font-size: 13px; text-align: left; }

            /* Table */
            .calib-table { width: 640px; margin: 10px auto; border-collapse: collapse; font-size: 13px; }
            .calib-table th { background: #333; padding: 6px 10px; color: #0cf; text-align: center; }
            .calib-table td { padding: 5px 10px; border-bottom: 1px solid #333; text-align: center; color: #ddd; }
            .calib-table tr:hover td { background: #2a2a2a; }

            .action-btns { display: flex; gap: 8px; justify-content: center; margin: 10px 0; }
            .action-btn { padding: 7px 18px; border: none; border-radius: 4px; cursor: pointer;
                          font-size: 13px; font-weight: bold; }
            .btn-undo   { background: #664400; color: #fda; }
            .btn-save   { background: #006633; color: #afa; }
            .btn-load   { background: #004466; color: #acf; }
            .btn-undo:hover  { background: #885500; }
            .btn-save:hover  { background: #008844; }
            .btn-load:hover  { background: #005588; }

            #click-hint { color: #aaa; font-size: 12px; margin: 4px 0; min-height: 18px; }
        </style>
    </head>
    <body>
        <h1>Firefighter Calibration Tool</h1>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('a')">Mode A — Homography</button>
            <button class="tab-btn"        onclick="switchTab('b')">Mode B — Servo Aim</button>
        </div>

        <!-- ================================================================ -->
        <!-- MODE A                                                            -->
        <!-- ================================================================ -->
        <div id="panel-a" class="panel active">
            <div class="video-wrap">
                <img src="/video_feed" id="stream-a" onclick="getCoordinates(event)">
            </div>
            <div class="info">
                <b>Urutan Klik:</b><br>
                1. Klik <b>Kiri-Atas</b> &nbsp; 2. Klik <b>Kanan-Atas</b>
                &nbsp; 3. Klik <b>Kanan-Bawah</b> &nbsp; 4. Klik <b>Kiri-Bawah</b><br><br>
                <b>Salin hasil ke SRC_POINTS di app.py:</b>
                <pre id="coord-display"></pre>
            </div>
            <button class="action-btn btn-undo" onclick="resetPoints()" style="margin-top:6px">Reset Klik</button>
        </div>

        <!-- ================================================================ -->
        <!-- MODE B                                                            -->
        <!-- ================================================================ -->
        <div id="panel-b" class="panel">
            <div class="warning">
                ⚠️ <b>Pastikan app.py sudah dimatikan</b> sebelum menggunakan tool ini
                (keduanya tidak bisa pakai serial port bersamaan).
            </div>

            <div class="video-wrap">
                <img src="/video_feed" id="stream-b" onclick="recordClick(event)">
            </div>
            <div id="click-hint">← Klik pada gambar untuk merekam posisi nozzle</div>

            <div class="ctrl-box">
                <div class="ctrl-row">
                    <label>Pan</label>
                    <input type="range" id="slider-pan" min="0" max="180" value="90"
                           oninput="document.getElementById('val-pan').innerText=this.value">
                    <span class="val-display" id="val-pan">90</span>°
                </div>
                <div class="ctrl-row">
                    <label>Tilt</label>
                    <input type="range" id="slider-tilt" min="0" max="180" value="90"
                           oninput="document.getElementById('val-tilt').innerText=this.value">
                    <span class="val-display" id="val-tilt">90</span>°
                </div>
                <div class="quick-btns">
                    <b style="color:#aaa;font-size:12px">Cepat:</b>
                    <button class="quick-btn" onclick="setAngles(0,90)">P0</button>
                    <button class="quick-btn" onclick="setAngles(45,90)">P45</button>
                    <button class="quick-btn" onclick="setAngles(90,90)">Center</button>
                    <button class="quick-btn" onclick="setAngles(135,90)">P135</button>
                    <button class="quick-btn" onclick="setAngles(180,90)">P180</button>
                    <button class="quick-btn" onclick="setAngles(90,60)">T60</button>
                    <button class="quick-btn" onclick="setAngles(90,120)">T120</button>
                </div>
                <button class="send-btn" id="send-btn" onclick="sendServo()">
                    Kirim ke Servo
                </button>
                <div class="countdown" id="countdown-msg"></div>
            </div>

            <div class="action-btns">
                <button class="action-btn btn-undo" onclick="doUndo()">Undo Terakhir</button>
                <button class="action-btn btn-save" onclick="doSave()">Simpan JSON</button>
                <button class="action-btn btn-load" onclick="doLoad()">Muat JSON</button>
            </div>

            <table class="calib-table">
                <thead><tr>
                    <th>#</th><th>Pan (°)</th><th>Tilt (°)</th>
                    <th>Real X (mm)</th><th>Real Y (mm)</th>
                </tr></thead>
                <tbody id="calib-tbody"></tbody>
            </table>
        </div>

        <!-- ================================================================ -->
        <!-- JAVASCRIPT                                                        -->
        <!-- ================================================================ -->
        <script>
        let modeA_clickCount = 0;
        let canClick = true;   // false selama countdown servo

        // ---------- Tab switching ----------
        function switchTab(mode) {
            document.querySelectorAll('.tab-btn').forEach((b,i) =>
                b.classList.toggle('active', (i === 0 && mode==='a') || (i===1 && mode==='b')));
            document.getElementById('panel-a').classList.toggle('active', mode==='a');
            document.getElementById('panel-b').classList.toggle('active', mode==='b');
            if (mode === 'b') refreshCalibTable();
        }

        // ---------- Mode A: Homography ----------
        function updateDisplayA() {
            fetch('/get_points').then(r => r.json()).then(data => {
                document.getElementById('coord-display').innerText =
                    'SRC_POINTS = np.float32(\\n' + JSON.stringify(data.points) + '\\n)';
            });
        }

        function getCoordinates(event) {
            if (modeA_clickCount >= 4) {
                alert('Sudah 4 titik! Klik Reset jika ingin mengulang.');
                return;
            }
            const img  = document.getElementById('stream-a');
            const rect = img.getBoundingClientRect();
            const x = Math.round((event.clientX - rect.left) * (img.naturalWidth  / rect.width));
            const y = Math.round((event.clientY - rect.top)  * (img.naturalHeight / rect.height));
            fetch('/update_point', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: modeA_clickCount, x, y})
            }).then(() => { modeA_clickCount++; updateDisplayA(); });
        }

        function resetPoints() {
            modeA_clickCount = 0;
            fetch('/reset_points', {method: 'POST'}).then(() => updateDisplayA());
        }

        // ---------- Mode B: Servo Calibration ----------
        function setAngles(pan, tilt) {
            document.getElementById('slider-pan').value  = pan;
            document.getElementById('slider-tilt').value = tilt;
            document.getElementById('val-pan').innerText  = pan;
            document.getElementById('val-tilt').innerText = tilt;
        }

        function sendServo() {
            const pan  = parseInt(document.getElementById('slider-pan').value);
            const tilt = parseInt(document.getElementById('slider-tilt').value);
            const btn  = document.getElementById('send-btn');
            const msg  = document.getElementById('countdown-msg');

            btn.disabled = true;
            canClick = false;
            document.getElementById('click-hint').innerText = '⏳ Menunggu servo bergerak...';

            fetch('/calib/set_servo', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pan, tilt})
            }).then(r => r.json()).then(data => {
                const serial = data.serial === 'connected' ? '✅ Serial OK' : '⚠️ No Serial';
                let secs = 2;
                msg.innerText = `${serial} — Servo bergerak... siap klik dalam ${secs}s`;
                const iv = setInterval(() => {
                    secs--;
                    if (secs <= 0) {
                        clearInterval(iv);
                        msg.innerText = '✅ Siap! Klik posisi nozzle di gambar.';
                        document.getElementById('click-hint').innerText = '← Klik pada gambar untuk merekam posisi nozzle';
                        canClick = true;
                        btn.disabled = false;
                    } else {
                        msg.innerText = `${serial} — Servo bergerak... siap klik dalam ${secs}s`;
                    }
                }, 1000);
            });
        }

        function recordClick(event) {
            if (!canClick) return;
            const img  = document.getElementById('stream-b');
            const rect = img.getBoundingClientRect();
            const px_x = Math.round((event.clientX - rect.left) * (img.naturalWidth  / rect.width));
            const px_y = Math.round((event.clientY - rect.top)  * (img.naturalHeight / rect.height));

            fetch('/calib/record_click', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({px_x, px_y})
            }).then(r => r.json()).then(data => {
                if (data.status === 'error') {
                    alert('❌ ' + data.message);
                } else {
                    document.getElementById('countdown-msg').innerText =
                        `✅ Titik #${data.total} direkam: real_x=${data.point.real_x}, real_y=${data.point.real_y}`;
                    refreshCalibTable();
                }
            });
        }

        function refreshCalibTable() {
            fetch('/calib/get_points').then(r => r.json()).then(data => {
                const tbody = document.getElementById('calib-tbody');
                tbody.innerHTML = '';
                data.calib_points.forEach((p, i) => {
                    tbody.innerHTML += `<tr>
                        <td>${i+1}</td>
                        <td>${p.pan}</td><td>${p.tilt}</td>
                        <td>${p.real_x}</td><td>${p.real_y}</td>
                    </tr>`;
                });
            });
        }

        function doUndo() {
            fetch('/calib/undo', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.status === 'empty') alert('Tidak ada data untuk dihapus.');
                else refreshCalibTable();
            });
        }

        function doSave() {
            fetch('/calib/save', {method: 'POST'}).then(r => r.json()).then(data => {
                alert(`✅ Tersimpan: ${data.count} titik\\n${data.path}`);
            });
        }

        function doLoad() {
            fetch('/calib/load').then(r => r.json()).then(data => {
                alert(`✅ Dimuat: ${data.count} titik dari JSON`);
                refreshCalibTable();
            });
        }

        // Init
        updateDisplayA();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    print(f"Platform  : {sys.platform}")
    print(f"Calib JSON: {CALIB_JSON_PATH}")
    print("Server    : http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
