import pickle
import cv2
import face_recognition
import numpy as np
from fastapi import FastAPI, UploadFile, File
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase (Render Secret File path)
cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-bb63c-default-rtdb.firebaseio.com/"
})

# Load encodings
with open("/app/EncodeFile.p", "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)

app = FastAPI(title="Face Attendance System API")

@app.get("/")
def home():
    return {"status": "Face Attendance API running"}

@app.post("/mark-attendance")
async def mark_attendance(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_encodings:
        return {"message": "No face detected"}

    matches = face_recognition.compare_faces(encodeListKnown, face_encodings[0])
    distances = face_recognition.face_distance(encodeListKnown, face_encodings[0])

    match_index = np.argmin(distances)

    if matches[match_index]:
        student_id = studentIds[match_index]
        ref = db.reference(f"Students/{student_id}")
        student = ref.get()

        ref.child("total_attendance").set(student["total_attendance"] + 1)

        return {
            "status": "success",
            "student_id": student_id,
            "name": student["name"]
        }

    return {"status": "unknown"}

