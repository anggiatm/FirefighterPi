import cv2
import os
from datetime import datetime

SAVE_DIR = "datasets"
os.makedirs(SAVE_DIR, exist_ok=True)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("ERROR: Cannot open camera")
    exit(1)

count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])
print("=== SHUTTER — Dataset Capture Tool ===")
print(f"  SPACE   : capture image")
print(f"  q       : quit")
print(f"  Saved to: {SAVE_DIR}/")
print(f"  Existing: {count} images")
print("=" * 40)

while True:
    ret, frame = cam.read()
    if not ret:
        print("ERROR: Cannot read frame")
        break

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    info = f"Captured: {count}  |  [SPACE] save  [q] quit"
    cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Shutter — Firefighter Dataset", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = os.path.join(SAVE_DIR, f"fire_{ts}.jpg")
        cv2.imwrite(fname, frame)
        count += 1
        print(f"  [SAVED] {fname}")
    elif key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print(f"\nDone. Total images captured: {count}")
