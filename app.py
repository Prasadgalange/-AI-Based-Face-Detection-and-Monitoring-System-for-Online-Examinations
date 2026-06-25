from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import threading

app = Flask(__name__)

# Create necessary directories
if not os.path.exists("exam_logs"):
    os.makedirs("exam_logs")
if not os.path.exists("exam_data"):
    os.makedirs("exam_data")

# Sample exam questions
EXAM_QUESTIONS = [
    {
        "id": 1,
        "question": "What is the capital of France?",
        "options": ["London", "Berlin", "Paris", "Madrid"],
        "correct_answer": "Paris"
    },
    {
        "id": 2,
        "question": "Which planet is known as the Red Planet?",
        "options": ["Venus", "Mars", "Jupiter", "Saturn"],
        "correct_answer": "Mars"
    },
    {
        "id": 3,
        "question": "What is 2 + 2?",
        "options": ["3", "4", "5", "6"],
        "correct_answer": "4"
    }
]

# Store exam session data
exam_sessions = {}

class BackgroundDetector:
    def __init__(self):
        self.video_capture = None
        self.active = False
        self.thread = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.session_id = None
        self.log_file = None
        self.reference_face_feature = None
        self.reference_background_gray = None
        self.registered_face_box = None
        
        # Timing and state variables for 2.5s verification gap
        self.last_process_time = 0.0
        self.last_faces = []
        self.last_status_text = ""
        self.last_status_color = (0, 255, 0)    
        
        # Load the built-in deep-learning face detector and recognizer models
        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (640, 480), score_threshold=0.6)
        self.recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

    def start(self):
        with self.lock:
            if self.active:
                return
            self.video_capture = cv2.VideoCapture(0)
            self.active = True
            self.session_id = str(int(time.time()))
            self.reference_face_feature = None  # Reset for new session
            self.reference_background_gray = None
            self.registered_face_box = None
            
            # Reset state variables
            self.last_process_time = 0.0
            self.last_faces = []
            self.last_status_text = ""
            self.last_status_color = (0, 255, 0)
            
            if not os.path.exists("exam_logs"):
                os.makedirs("exam_logs")
            self.log_file = open("exam_logs/exam_log.txt", "a")

            # Initialize session data with answers dict
            exam_sessions[self.session_id] = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_detections": [],
                "status": "in_progress",
                "answers": {}
            }

            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            print(f"Background detection thread started for session {self.session_id}")

    def stop(self):
        with self.lock:
            if not self.active:
                return
            self.active = False
            
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
        with self.lock:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            cv2.destroyAllWindows()
            self.reference_face_feature = None
            self.reference_background_gray = None
            self.registered_face_box = None
            
            # Reset state variables
            self.last_process_time = 0.0
            self.last_faces = []
            self.last_status_text = ""
            self.last_status_color = (0, 255, 0)
            print("Background detection thread stopped cleanly.")

    def register_face(self):
        with self.lock:
            cap = self.video_capture
            active = self.active
            
        if cap is None or not active:
            # If camera is not running, temporarily capture
            temp_cap = cv2.VideoCapture(0)
            ret, frame = temp_cap.read()
            temp_cap.release()
        else:
            ret, frame = cap.read()
            
        if not ret or frame is None:
            return False, "Failed to capture image. Please check your camera connection."
            
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        
        if faces is None or len(faces) == 0:
            return False, "No face detected. Please position yourself in front of the camera."
            
        if len(faces) > 1:
            return False, "Multiple faces detected. Please make sure you are alone during registration."
            
        # Extract and align features
        face = faces[0]
        x, y, fw, fh = map(int, face[0:4])
        aligned_face = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned_face)
        
        # Capture reference background in registration frame
        ref_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ref_gray_blur = cv2.GaussianBlur(ref_gray, (21, 21), 0)
        
        with self.lock:
            self.reference_face_feature = feature
            self.reference_background_gray = ref_gray_blur
            self.registered_face_box = (x, y, fw, fh)
            
        return True, "Face registered successfully"

    def _detection_loop(self):
        while True:
            with self.lock:
                if not self.active:
                    break
                cap = self.video_capture
                ref_feature = self.reference_face_feature
                ref_bg_gray = self.reference_background_gray
                reg_face_box = self.registered_face_box

            if cap is None:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            # If 2.5 seconds have elapsed, execute heavy face detection and matching
            if current_time - self.last_process_time >= 2.5:
                self.last_process_time = current_time
                
                h_frame, w_frame, _ = frame.shape
                self.detector.setInputSize((w_frame, h_frame))
                _, detected_faces = self.detector.detect(frame)
                
                # Filter detections by area to ignore background noise
                faces = []
                if detected_faces is not None and len(detected_faces) > 0:
                    sorted_faces = sorted(detected_faces, key=lambda f: f[2] * f[3], reverse=True)
                    largest_area = sorted_faces[0][2] * sorted_faces[0][3]
                    
                    faces.append(sorted_faces[0])
                    
                    for f in sorted_faces[1:]:
                        area = f[2] * f[3]
                        if area >= 0.3 * largest_area:
                            faces.append(f)

                # Check background changes for motion/intrusion detection
                bg_alert = False
                change_ratio = 0.0
                if ref_bg_gray is not None:
                    # Convert current frame to grayscale and blur
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    curr_gray_blur = cv2.GaussianBlur(curr_gray, (21, 21), 0)
                    
                    # Compute absolute difference
                    frame_delta = cv2.absdiff(ref_bg_gray, curr_gray_blur)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    
                    # Create mask of the same size, initialized to 255 (monitoring zone)
                    mask = np.ones((h_frame, w_frame), dtype=np.uint8) * 255
                    
                    # Exclude the user's face and body zone
                    ex_box = None
                    if len(faces) == 1:
                        ex_box = map(int, faces[0][0:4])
                    elif reg_face_box is not None:
                        ex_box = reg_face_box
                        
                    if ex_box is not None:
                        fx, fy, fw, fh = ex_box
                        # Define exclusion column centered on the face
                        ex_x1 = max(0, fx - int(fw * 0.6))
                        ex_x2 = min(w_frame, fx + fw + int(fw * 0.6))
                        ex_y1 = max(0, fy - int(fh * 0.6))
                        ex_y2 = h_frame  # Exclude all the way to the bottom to cover the torso/chair
                        
                        # Set exclusion zone to 0 in mask
                        cv2.rectangle(mask, (ex_x1, ex_y1), (ex_x2, ex_y2), 0, -1)
                        
                    # Apply mask to thresholded difference
                    bg_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
                    monitoring_area = cv2.countNonZero(mask)
                    if monitoring_area > 0:
                        changed_pixels = cv2.countNonZero(bg_thresh)
                        change_ratio = changed_pixels / monitoring_area
                        # If more than 6% of the background has changed, mark intrusion
                        if change_ratio > 0.06:
                            bg_alert = True

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with self.lock:
                    session_data = exam_sessions.get(self.session_id)
                    log_f = self.log_file

                if len(faces) == 0:
                    message = f"[{timestamp}] ALERT: No face detected!"
                    print(message)
                    if log_f:
                        log_f.write(message + "\n")
                        log_f.flush()
                    cv2.imwrite(f"exam_logs/no_face_{int(time.time())}.jpg", frame)
                    
                    self.last_faces = []
                    self.last_status_text = "No Face Detected!"
                    self.last_status_color = (0, 0, 255)
                    
                    if session_data:
                        session_data["face_detections"].append({
                            "timestamp": timestamp,
                            "status": "no_face",
                            "image_path": f"exam_logs/no_face_{int(time.time())}.jpg"
                        })

                elif len(faces) > 1:
                    message = f"[{timestamp}] ALERT: Multiple faces detected!"
                    print(message)
                    if log_f:
                        log_f.write(message + "\n")
                        log_f.flush()
                    cv2.imwrite(f"exam_logs/multiple_faces_{int(time.time())}.jpg", frame)
                    
                    self.last_faces = []
                    self.last_status_text = "Multiple Faces Detected!"
                    self.last_status_color = (0, 0, 255)
                    
                    if session_data:
                        session_data["face_detections"].append({
                            "timestamp": timestamp,
                            "status": "multiple_faces",
                            "image_path": f"exam_logs/multiple_faces_{int(time.time())}.jpg"
                        })

                else:
                    # Exactly one face detected
                    face = faces[0]
                    x, y, w, h = map(int, face[0:4])
                    self.last_faces = [(x, y, w, h)]
                    
                    if ref_feature is not None:
                        # Align, crop and match
                        aligned_face = self.recognizer.alignCrop(frame, face)
                        live_feature = self.recognizer.feature(aligned_face)
                        similarity = self.recognizer.match(ref_feature, live_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        
                        # Set threshold to 50% matching (cosine similarity threshold = 0.50)
                        if similarity < 0.50:
                            message = f"[{timestamp}] ALERT: Face verification failed! (Similarity: {similarity:.3f})"
                            print(message)
                            if log_f:
                                log_f.write(message + "\n")
                                log_f.flush()
                            cv2.imwrite(f"exam_logs/unmatched_face_{int(time.time())}.jpg", frame)
                            
                            self.last_status_text = f"UNMATCHED FACE! ({similarity:.2f})"
                            self.last_status_color = (0, 0, 255)
                            
                            if session_data:
                                session_data["face_detections"].append({
                                    "timestamp": timestamp,
                                    "status": "unmatched_face",
                                    "image_path": f"exam_logs/unmatched_face_{int(time.time())}.jpg"
                                })
                        elif bg_alert:
                            message = f"[{timestamp}] ALERT: Background activity detected! (Ratio: {change_ratio:.3f})"
                            print(message)
                            if log_f:
                                log_f.write(message + "\n")
                                log_f.flush()
                            cv2.imwrite(f"exam_logs/background_activity_{int(time.time())}.jpg", frame)
                            
                            self.last_status_text = "BG Activity Detected!"
                            self.last_status_color = (0, 0, 255)
                            
                            if session_data:
                                session_data["face_detections"].append({
                                    "timestamp": timestamp,
                                    "status": "background_activity",
                                    "image_path": f"exam_logs/background_activity_{int(time.time())}.jpg"
                                })
                        else:
                            self.last_status_text = f"Verified ({similarity:.2f})"
                            self.last_status_color = (0, 255, 0)
                    else:
                        self.last_status_text = "Registering face..."
                        self.last_status_color = (255, 255, 0)

            # Draw overlay features on every frame to keep visual feedback smooth
            for (x, y, w, h) in self.last_faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), self.last_status_color, 2)
            
            if self.last_status_text:
                cv2.putText(frame, self.last_status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.last_status_color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                with self.lock:
                    self.latest_frame = frame_bytes

            time.sleep(0.04)

detector = BackgroundDetector()

def generate_frames():
    while True:
        with detector.lock:
            active = detector.active
            frame = detector.latest_frame
        
        if not active:
            break
            
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        time.sleep(0.05)

@app.route('/')
def index():
    detector.start()
    return render_template('index.html', questions=EXAM_QUESTIONS)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_face', methods=['POST'])
def register_face():
    success, message = detector.register_face()
    return jsonify({'success': success, 'message': message})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.get_json()
    question_id = data.get('question_id')
    answer = data.get('answer')
    
    question = next((q for q in EXAM_QUESTIONS if q['id'] == question_id), None)
    
    session_id = detector.session_id
    if session_id and session_id in exam_sessions:
        exam_sessions[session_id]["answers"][str(question_id)] = {
            "answer": answer,
            "correct": answer == question['correct_answer'] if question else False,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    if question and answer == question['correct_answer']:
        return jsonify({'correct': True, 'message': 'Correct answer!'})
    else:
        return jsonify({'correct': False, 'message': 'Incorrect answer. Try again!'})

@app.route('/submit_exam', methods=['POST'])
def submit_exam():
    session_id = detector.session_id
    if not session_id or session_id not in exam_sessions:
        return jsonify({'success': False, 'message': 'No active session found'})
        
    exam_sessions[session_id]["status"] = "completed"
    exam_sessions[session_id]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_questions = len(EXAM_QUESTIONS)
    answers_dict = exam_sessions[session_id].get("answers", {})
    correct_answers = sum(1 for answer in answers_dict.values() if answer["correct"])
    score = (correct_answers / total_questions) * 100
    
    exam_sessions[session_id]["score"] = score
    
    filename = f"exam_data/exam_session_{session_id}.json"
    with open(filename, 'w') as f:
        json.dump(exam_sessions[session_id], f, indent=4)
    
    detector.stop()
    
    return jsonify({
        'success': True,
        'message': 'Exam submitted successfully!',
        'score': score,
        'total_questions': total_questions,
        'correct_answers': correct_answers
    })

@app.route('/stop_detection', methods=['POST'])
def stop_detection_route():
    detector.stop()
    return jsonify({'success': True, 'message': 'Face detection stopped successfully'})

@app.route('/get_status')
def get_status():
    with detector.lock:
        status = detector.last_status_text
        color = detector.last_status_color
        active = detector.active
        
    color_class = "text-success"
    if color == (0, 0, 255):
        color_class = "text-danger"
    elif color == (255, 255, 0):
        color_class = "text-warning"
        
    return jsonify({
        'status': status if active else 'Stopped',
        'class': color_class
    })

if __name__ == '__main__':
    app.run(debug=True) 