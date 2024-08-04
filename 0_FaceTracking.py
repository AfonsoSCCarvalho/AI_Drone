import cv2
import numpy as np
from djitellopy import tello
from time import sleep

# Initialize Tello drone
me = tello.Tello()
me.connect()
print(me.get_battery())

# Start video stream and take off
me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 25, 0)
sleep(1)

# Set frame size and control parameters
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0]
pError = 0


def findFace(img):
    """
    Detects faces in the image and returns the image with face bounding boxes and the coordinates and area of the largest face.
    """
    faceCascade = cv2.CascadeClassifier("Resources/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):
    """
    Tracks the face by adjusting the drone's position based on the face coordinates and area.
    """
    area = info[1]
    x, y = info[0]
    fb = 0
    error = x - w // 2  # Error in x relative to the center
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if fbRange[0] < area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    me.send_rc_control(0, fb, 0, speed)
    return error


# Main loop
while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    # Find and track face
    img, info = findFace(img)
    pError = trackFace(info, w, pid, pError)

    # Display the output image
    cv2.imshow("Output", img)

    # Land the drone if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break
