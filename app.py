from fastapi import FastAPI, UploadFile, File
import face_recognition
from PIL import Image
import numpy as np
import io

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Face detection API is running"}

@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()

    # Convert bytes to numpy image
    image = face_recognition.load_image_file(io.BytesIO(contents))

    # Detect face locations
    face_locations = face_recognition.face_locations(image)

    # Optional: convert to (x,y,w,h) style
    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append({
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "left": int(left)
        })

    return {
        "faces_detected": len(faces),
        "faces": faces
    }

