import cv2
import time
import threading
import RPi.GPIO as GPIO
from flask import Flask, Response, jsonify
from ultralytics import YOLO
import atexit
import requests
import os

app = Flask(__name__)
model = YOLO("burung_ncnn_model")

cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
latest_frame = None

BUZZER_PIN = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

last_bird_time = 0
HOLD_TIME = 5
bird_detected_status = False
image_sent = False 

API_URL = "https://tkj-3b.com/tkj-3b.com/opengate/upload.php"
STATIC_STATUS = "terdeteksi"
STATIC_USER_ID = "1"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def detect_objects():
    global latest_frame, last_bird_time, bird_detected_status, image_sent
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        results = model(frame, conf=0.7)

        annotated_frame = results[0].plot()


        detected_bird = any(
            int(box.cls[0]) == 0 for result in results for box in result.boxes
        )

        if detected_bird:
            last_bird_time = time.time()

        
        if detected_bird or (time.time() - last_bird_time < HOLD_TIME):
            bird_detected_status = True
            GPIO.output(BUZZER_PIN, GPIO.HIGH)

            if detected_bird and not image_sent:
                timestamp = int(time.time())
                image_path = os.path.join(UPLOAD_DIR, f"detected_{timestamp}.jpg")
                cv2.imwrite(image_path, annotated_frame)
                threading.Thread(
                    target=send_detection_to_api, args=(image_path,), daemon=True
                ).start()

                image_sent = True
        else:
            bird_detected_status = False
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            image_sent = False

        
        fps = 1 / (time.time() - prev_time) if prev_time else 0
        prev_time = time.time()

        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            latest_frame = annotated_frame

        
        time.sleep(0.01)


def send_detection_to_api(image_path):
    try:
        with open(image_path, 'rb') as img_file:
            files = {'image': (os.path.basename(image_path), img_file, 'image/jpeg')}
            data = {
                'status_hama': "Terdeteksi",
                'user_id': "1"
            }
            response = requests.post(API_URL, files=files, data=data)
            
            print("Response code:", response.status_code)
            print("Response text:", response.text)

            if response.status_code == 200:
                print("? Berhasil mengirim data ke API")
            else:
                print(f"? Gagal mengirim ke API, status code: {response.status_code}")
    except Exception as e:
        print(f"? Error kirim API: {e}")


threading.Thread(target=detect_objects, daemon=True).start()

@app.route('/')
def index():
    return """
    <html><body>
    <h2>Live Monitoring</h2>
    <img src="/video_feed" width="640" height="480">
    </body></html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    remaining_time = max(0, HOLD_TIME - (time.time() - last_bird_time)) if bird_detected_status else 0
    return jsonify({
        "bird_detected": bird_detected_status,
        "hold_time_remaining": round(remaining_time, 2)
    })

def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@atexit.register
def cleanup_gpio():
    GPIO.cleanup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
