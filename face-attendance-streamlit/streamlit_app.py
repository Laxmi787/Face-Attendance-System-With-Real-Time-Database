import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

st.set_page_config(page_title="Face Attendance System")

st.title("Face Attendance System (Streamlit)")

# ---------- Firebase ----------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendancerealtime-bb63c-default-rtdb.firebaseio.com/"
})

# ---------- Load Encode File ----------
with open("EncodeFile.p", "rb") as f:
    encode_list_known, student_ids = pickle.load(f)

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    success, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(frame_rgb)
    encodes = face_recognition.face_encodings(frame_rgb, faces)

    for encode_face, face_loc in zip(encodes, faces):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            student_id = student_ids[match_index]

            ref = db.reference(f"Students/{student_id}")
            ref.child("last_attendance_time").set(str(datetime.now()))

            top, right, bottom, left = face_loc
            cv2.rectangle(frame_rgb, (left, top), (right, bottom), (0,255,0), 2)
            cv2.putText(frame_rgb, student_id, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    FRAME_WINDOW.image(frame_rgb)

cap.release()

