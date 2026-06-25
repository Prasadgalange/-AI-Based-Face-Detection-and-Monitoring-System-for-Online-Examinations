

import cv2
import time
import os
from datetime import datetime

def initialize_cascades():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return face_cascade

def setup_logging():
    if not os.path.exists("exam_logs"):
        os.makedirs("exam_logs")
    return open("exam_logs/exam_log.txt", "a")

def process_frame(frame, face_cascade, log_file):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if len(faces) == 0:
        log_alert(log_file, timestamp, "No face detected!", frame, "no_face")
    elif len(faces) > 1:
        log_alert(log_file, timestamp, "Multiple faces detected!", frame, "multiple_faces")
    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def log_alert(log_file, timestamp, message, frame, alert_type):
    full_message = f"[{timestamp}] ALERT: {message}"
    print(full_message)
    log_file.write(full_message + "\n")
    cv2.imwrite(f"exam_logs/{alert_type}_{int(time.time())}.jpg", frame)

def main():
    face_cascade = initialize_cascades()
    log_file = setup_logging()
    video_cap = cv2.VideoCapture(0)
    
    print("Exam monitoring started... Press 'q' to quit.")

    try:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                break

            process_frame(frame, face_cascade, log_file)
            cv2.imshow("Exam Monitoring - Press 'q' to Quit", frame)

            if cv2.waitKey(10) == ord('q'):
                break

    finally:
        video_cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print("Monitoring ended. Logs saved to exam_logs/exam_log.txt")

if __name__ == "__main__":
    main()