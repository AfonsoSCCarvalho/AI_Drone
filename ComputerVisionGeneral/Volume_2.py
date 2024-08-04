import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
#########################
wCam, hCam = 640, 480
#########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detectionCon =.7, maxHands=1)

#audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume.GetMute()
#volume.GetMasterVolumeLevel()
volRange =volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar= 400
volPer= 0
area = 0
colorVol = (0,0,0)
#fps
pTime = 0
cTime = 0

while True:
        success, img = cap.read()

        #Find hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw = True)

        if len(lmList) != 0:
            #Filter based on size
            area = (bbox[2]-bbox[0])* (bbox[3]-bbox[1])//100
            #print(area)
            if 250 < area < 1000:
            #print('yes')
            #Find distance between index and Thumb
                length, img , lineInfo = detector.findDistance(4,8,img)

                #Convert Volume
                vol = np.interp(length, [50, 300], [minVol, maxVol])
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])

                #Reduce resolution to make it smoother
                smoothness = 5
                volPer = smoothness* round(volPer/smoothness)
                #CHeck if fingers up
                fingers = detector.fingersUp()
                print(fingers)
                #If pinly is down set volume
                if not fingers[4]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.line(img, (lineInfo[0], lineInfo[1]), (lineInfo[2], lineInfo[3]), (0, 255, 0), 3)
                    colorVol = (0,255,0)
                else:
                    colorVol = (255, 0, 0)
                if length < 50:
                    cv2.circle(img, (lineInfo[4],lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

        # drawings
        cv2.rectangle(img, (50,150), (85,400), (0,255,0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f' {int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 3,(0, 255, 0), 3)

        cVol = int(volume.GetMasterVolumeLevelScalar()*100)
        cv2.putText(img, f'Vol Set: {int(cVol)}%', (400, 50), cv2.FONT_HERSHEY_PLAIN, 2,colorVol, 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)