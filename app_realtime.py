import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime, date
from flask import Flask, render_template, Response, url_for
import sqlite3

app = Flask(__name__)

# --- Database and Face Recognition Setup (Mostly unchanged) ---

# Path to known images
KNOWN_FACES_DIR = "known_faces"
known_face_encodings = []
known_face_names = []

# Setup SQLite attendance DB
def init_db():
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# Function to load known faces from the 'known_faces' directory
def load_known_faces():
    print("Loading known faces...")
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(os.path.splitext(filename)[0])
                    print(f"Loaded face: {os.path.splitext(filename)[0]}")
            except Exception as e:
                print(f"Error loading {image_path}: {e}")

# Function to mark attendance in the database
def mark_attendance(name):
    date_today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    # Avoid duplicate entries for the same person on the same day
    c.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, date_today))
    if c.fetchone() is None:
        c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date_today, time_now))
        conn.commit()
        print(f"Attendance marked for {name} at {time_now}")
    conn.close()

# --- Real-time Video Processing ---

def generate_frames():
    """Generator function to capture frames, perform recognition, and stream."""
    video_capture = cv2.VideoCapture(0) # Use 0 for the default webcam

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"  # Default value

            # If a match was found in known_face_encodings, use the first one.
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    mark_attendance(name)
            
            face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')# --- Flask Routes ---

@app.route("/")
def index():
    """Main page route that renders the webcam feed and attendance log."""
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute("SELECT name, date, time FROM attendance WHERE date = ? ORDER BY time DESC", (today,))
    attendance_data = c.fetchall()
    conn.close()
    return render_template("index_realtime.html", attendance_data=attendance_data)

@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    init_db()
    load_known_faces()
    app.run(host="0.0.0.0", port=5000, debug=True)