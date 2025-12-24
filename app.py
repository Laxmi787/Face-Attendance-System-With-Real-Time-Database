import pickle
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

import firebase_admin
from firebase_admin import credentials, db

from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist

# Firebase
cred = credentials.Certificate("/etc/secrets/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendancerealtime-bb63c-default-rtdb.firebaseio.com/"
})

# Load encodings
with open("/app/EncodeFile.p", "rb") as f:
    encodeListKnown, studentIds = pickle.load(f)

encodeListKnown = np.array(encodeListKnown)

# InsightFace (CPU + lighter model)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(
    ctx_id=-1,                     # CPU ONLY
    det_size=(320, 320)            # smaller detector = lower memory
)

app = FastAPI()


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

    faces = face_app.get(img)

    if not faces:
        return {"message": "No face detected"}

    face_encoding = faces[0].normed_embedding.reshape(1, -1)

    distances = cdist(face_encoding, encodeListKnown, metric="euclidean")[0]
    match_index = np.argmin(distances)

    if distances[match_index] < 0.9:
        student_id = studentIds[match_index]

        ref = db.reference(f"Students/{student_id}")
        student = ref.get()

        if not student:
            return {"status": "student record missing"}

        total = student.get("total_attendance", 0)
        ref.child("total_attendance").set(total + 1)

        return {
            "status": "success",
            "student_id": student_id,
            "name": student.get("name", "Unknown")
        }

    return {"status": "unknown"}

