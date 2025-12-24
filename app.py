import pickle
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

import firebase_admin
from firebase_admin import credentials, db
from scipy.spatial.distance import cdist


cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://faceattendancerealtime-bb63c-default-rtdb.firebaseio.com/"
})


with open("/app/EncodeFile.p", "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)

encodeListKnown = np.array(encodeListKnown)

app = FastAPI(title="Face Attendance API")


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/mark-attendance")
async def mark_attendance(file: UploadFile = File(...)):

    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OpenCV face detector
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return {"message": "No face detected"}

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    face_resized = cv2.resize(face_img, (128, 128))
    face_vector = face_resized.flatten().reshape(1, -1)

    distances = cdist(face_vector, encodeListKnown, metric="euclidean")[0]
    match_index = np.argmin(distances)

    if distances[match_index] < 5000:

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

