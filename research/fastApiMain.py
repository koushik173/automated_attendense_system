from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import threading
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
camera_status = False
camera_thread = None
encodeListKnown = []  # Store encoding list globally

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

encodeListKnown = findEncodings(images)  # Store encodings globally

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [entry.split(',')[0] for entry in myDataList]

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def start_camera():
    global camera_status 

    # Use encodeListKnown globally
    global encodeListKnown  

    cap = cv2.VideoCapture(0)

    while camera_status:
        success, img = cap.read()
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

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

@app.get('/')
async def ddnot():
    return {"welcome": "welcome to attendense system"}



@app.get('/start_camera')
async def start_camera_route(background_tasks: BackgroundTasks):
    global camera_status, camera_thread

    if not camera_status:
        camera_status = True
        background_tasks.add_task(start_camera)
        return JSONResponse(content={'status': 'Camera started'})
    else:
        return JSONResponse(content={'status': 'Camera already running'})

@app.get('/stop_camera')
async def stop_camera_route():
    global camera_status, camera_thread

    if camera_status:
        camera_status = False
        return JSONResponse(content={'status': 'Camera stopped'})
    else:
        return JSONResponse(content={'status': 'Camera not running'})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
    