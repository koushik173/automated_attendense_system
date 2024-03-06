from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory="template")

# Global variables
camera_status = False
video_capture = None
encodeListKnown = []  

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

# Load images and encode known faces
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

def markAttendance(name):
    with open('Attendance.csv', 'a+') as f:
        f.seek(0)  # Move to the beginning of the file
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')

def generate_frames():
    global camera_status
    global video_capture
    global encodeListKnown

    while camera_status:
        success, img = video_capture.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
async def index(request: Request):
    global video_capture
    if video_capture is not None:
        video_capture.release()
    return {"message": "Camera not started"}

@app.get("/initialize")
async def initialize(request: Request):
    return templates.TemplateResponse("http://localhost:5173/tc/classroom/Home", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/start_video")
async def start_video():
    global video_capture
    global camera_status

    
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera_status = True
    return {"message": "Video started"}

@app.get("/stop_video")
async def stop_video():
    global video_capture
    global camera_status

    camera_status = False

    if video_capture is not None:
        video_capture.release()

        
 
    csv_file_path = 'Attendance.csv'

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)

        for row in csv_reader:
            print(row)
    with open(csv_file_path, 'w', newline='') as file:
        pass

    return {"message": "Video stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8001)
