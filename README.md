# Face Attendance System with Real-Time Database
https://www.linkedin.com/posts/laxmi-chaudhary-2b57042ba_facialrecognition-realtimedatabase-machinelearning-activity-7265364217065578497-tvNs?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAEyikKkB72N4T6Gt2cFDr0Y6DI9OcCSO10k

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
6. <img width="368" height="184" alt="image" src="https://github.com/user-attachments/assets/0650c1f6-8472-497b-abdb-373597b3fcfc" />
<img width="367" height="185" alt="image" src="https://github.com/user-attachments/assets/1a6ac1cf-6bac-481e-be03-7b0ad773e8f0" />
<img width="374" height="188" alt="image" src="https://github.com/user-attachments/assets/b8068e80-28a7-4150-88c2-adcc20b08bcd" />
<img width="374" height="185" alt="image" src="https://github.com/user-attachments/assets/fb594d07-7f19-4313-8cc1-27716af7e80c" />
<img width="373" height="157" alt="image" src="https://github.com/user-attachments/assets/8bf86113-4c4d-4d93-a710-08d2eed942c2" />





---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Face-Attendance-System.git
cd Face-Attendance-System
pip install -r requirements.txt
python main.py

