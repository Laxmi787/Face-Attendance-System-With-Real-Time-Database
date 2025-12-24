# Face Attendance System with Real-Time Database

##  Overview
The **Face Attendance System** is a real-time attendance management application that uses
**facial recognition technology** to automatically identify individuals and mark their
attendance. The system eliminates manual attendance processes, reduces proxy attendance,
and securely stores attendance records in a **real-time cloud database**.

This project demonstrates practical use of **Computer Vision, Python, and database integration**
in a real-world scenario.

---

##  Features
- Real-time face detection and recognition
- Automatic attendance marking
- Secure cloud database storage
- Fast and contactless attendance system
- Scalable and easy to extend
- High accuracy using encoded facial data

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Face Recognition:** face-recognition library  
- **Database:** Firebase / Real-time Database  
- **Libraries:** NumPy, Pickle  

---

## 📂 Project Structure
Face-Attendance-System/
│── main.py # Main application file
│── AddDatatoDatabase.py # Uploads attendance data to database
│── EncodeGenerator.py # Generates and stores face encodings
│── EncodeFile.p # Encoded face data
│── README.md # Project documentation


---

## ⚙️ How It Works
1. The system captures live video using a webcam.
2. Faces are detected and encoded using facial recognition techniques.
3. The detected face is compared with stored encodings.
4. If a match is found, attendance is automatically marked.
5. Attendance records are stored in a real-time cloud database.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Face-Attendance-System.git
cd Face-Attendance-System
pip install -r requirements.txt
python main.py

