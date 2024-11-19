
import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from cvzone import cornerRect
import firebase_admin
from firebase_admin import credentials



import os

import json
import firebase_admin
from firebase_admin import credentials

# Load the JSON file contents into a Python dict
with open(r"C:\Users\hp\New folder[face_recognition with real time database]\serviceAccountKey.json") as f:
    service_account_info = json.load(f)

# Initialize the Firebase app with the service account credentials from dict
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred)

print("Firebase initialized successfully")





from code import encodeListKnownWithIds, faceCurFrame, encodeCurFrame, imgBackground

# Use the default camera (usually the built-in webcam) with index 0
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set the width of the video frame
cap.set(4, 480)  # Set the height of the video frame

imageBackground = cv2.imread("Resources/background.png")

#Importing the Mode Images into a list
folderModePath = 'Resources/Modes'

modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

print(len(imgModeList))

# Load the encoding file
print("Loading Encoding File...")
with open("EncodeFile.p", 'rb') as file:
    encodeListKnownWithIds = pickle.load(file)

encodeListKnown, studentIds = encodeListKnownWithIds
print(f"Student IDs: {studentIds}")
print("Encoding File Loaded...")


while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imageBackground[162:162+480,55:55+640] = img
    imageBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]


    for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print("matches", matches)
        print("faceDis", faceDis)

        matchIndex = np.argmin(faceDis)
        print("match Index", matchIndex)

        if matches[matchIndex]:
            #print("Known Face Detected")
            #print(studentIds[matchIndex])
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            bbox = 55+x1,162+y1,x2-x1,y2-y1
            imgBackground = cvzone,cornerRect(imageBackground,bbox,rt=0)
    if not success:
        print("Failed to capture image")
        break

    #cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imageBackground)

    # Wait for 1 ms and break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window when done
cap.release()
cv2.destroyAllWindows()










