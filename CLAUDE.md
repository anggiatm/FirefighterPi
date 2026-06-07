# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart firefighter robot system. A ceiling-mounted USB webcam feeds frames through a YOLOv8n ONNX model to detect fire. Detected fire coordinates are transformed from pixel space to real-world millimeters via perspective homography, then converted to pan/tilt servo angles sent to an Arduino Uno over UART serial (9600 baud). A relay-controlled water pump fires when the servo reaches the target.

The codebase runs on **both Windows (dev/test) and Raspberry Pi 3B+ (production)**.

## System Architecture

```
Webcam (640x480) → ONNX Inference (320x320) → Perspective Transform → Angle Calculation → Serial TX → Arduino Uno → Servo Pan/Tilt + Relay Pump
```

**Serial protocol to Arduino:** Single-char prefix + integer value, no delimiter:
- `P<0-180>` — pan angle
- `T<0-180>` — tilt angle
- `M0` / `M1` — pump ON / OFF

**Serial port auto-detection:** Windows tries `COM3`–`COM12`; Linux/Pi tries `/dev/ttyS0`, `/dev/ttyAMA0`, `/dev/ttyUSB0`. If none found, app runs in simulation mode (no crash).

**Coordinate pipeline:**
1. Raw pixel centroid (640×480 frame)
2. Exponential smoothing (`ALPHA_SMOOTHING = 0.25`)
3. `cv2.perspectiveTransform` via `HOMOGRAPHY_MATRIX` (calibrated with `SRC_POINTS`) → real-world mm
4. `calculate_angles()` → pan/tilt degrees clamped to [0, 180]
   - If `servo_calibration.json` exists with ≥ 3 points: uses IDW interpolation
   - Otherwise: falls back to `math.atan2` trigonometric model

**State machine (4 states):**
- `0` IDLE — no fire, servos at 90/90, pump off
- `1` LOCKING — fire detected, wait 0.5s stability (movement < `STABLE_THRESHOLD_PX = 6px`)
- `2` AIMING — servo command sent, wait 1.0s for mechanical movement
- `3` EXTINGUISHING — pump on, continuous angle tracking

## File Roles

| File | Purpose |
|------|---------|
| `app.py` | **Production entry point** — full pipeline with serial, perspective transform, state machine, Flask stream |
| `calibration_server.py` | Web tool (port 5001) — Mode A: click 4 floor corners for homography; Mode B: servo aim calibration that builds `servo_calibration.json` |
| `shutter.py` | Dataset capture tool — SPACE to save frame to `datasets/`, `q` to quit (requires display) |
| `config.yaml` | Reference config (GPIO pins, FOV, thresholds) — **not loaded at runtime** |
| `models/fire_detector-3/weights/best.onnx` | Active ONNX model |
| `servo_calibration.json` | Runtime artifact — ground-truth servo→coordinate mapping, auto-loaded by `app.py` |
| `archive/` | Older main*.py iterations, kept for reference |

## Running

### Windows (dev/test)

```bash
conda activate firefighter_pi
pip install -r requirements.txt

python app.py              # http://localhost:5000
python calibration_server.py  # http://localhost:5001
```

### Raspberry Pi (production)

> **Requirement:** Use Raspberry Pi OS **64-bit** — onnxruntime has no ARM32 wheel.

```bash
pip install -r requirements-pi.txt   # uses opencv-python-headless

python app.py              # http://<pi-ip>:5000
python calibration_server.py  # http://<pi-ip>:5001
```

`shutter.py` requires a display — run it on a desktop machine or Pi with GUI, not headless.

## Calibration Workflow

Two-step calibration, both done via `calibration_server.py` (stop `app.py` first — serial conflict):

1. **Mode A — Homography:** Click 4 floor corners in order (TL, TR, BR, BL) → copy `SRC_POINTS` into `app.py` and the matching constants in `calibration_server.py`
2. **Mode B — Servo Aim:** Command servos to ~9 positions (3×3 grid), click where nozzle points on each → Save → restarts `app.py` auto-loads the file

## Key Constants to Tune Per Deployment

All hardcoded in `app.py` and duplicated in `calibration_server.py` (must stay in sync):

- `SRC_POINTS` — pixel coordinates of the 4 floor corners
- `LEBAR_FISIK_*` / `TINGGI_FISIK_*` — physical floor area dimensions in mm
- `CEILING_HEIGHT` — camera/nozzle height from floor in mm
- `OFFSET_X`, `OFFSET_Y` — mechanical offset nozzle vs camera center in mm
- `CONF_THRESHOLD` — detection sensitivity (0.1 = very sensitive)
- `ALPHA_SMOOTHING` — centroid smoothing (0.25 = heavily smoothed)

## Model Training

Trained with Ultralytics YOLOv8n. The ONNX export (`best.onnx`) runs at inference with `onnxruntime` only — no Ultralytics dependency at runtime.

To retrain: annotate images in YOLO format, then `model.export(format='onnx', imgsz=320)`.
