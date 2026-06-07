import serial
import time

# Sesuaikan baudrate dengan Arduino
arduino = serial.Serial(
    port="/dev/serial0",
    baudrate=9600,
    timeout=1
)

print("Serial connected")

time.sleep(2)  # tunggu serial stabil

while True:
    # Kirim data
    try:
        payload = "P_10"
        arduino.write(payload.encode('utf-8'))
    except Exception as e:
        print(f"Gagal kirim data serial: {e}")
