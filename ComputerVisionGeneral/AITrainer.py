import cv2
import time
from Modules import ModulePose as pm
import numpy as np

cap = cv2.VideoCapture(0)#"Trainer/curls.mp4")

detector = pm.poseDetector()
count = 0.5 #ver o acerto
dir = 1 #0 baixo, 1 cima
color = (255,0, 255)
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280,720))
    #img = cv2.imread("Trainer/test.jpg")
    img = detector.findPose(img, False)

    lmList = detector.findPosition(img, False)
    #print(lmList)
    if len(lmList) != 0:
         #Right Arm
        #angle = detector.findAngle(img, 12,14,16)
        #Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (45, 150), (100, 0))
        bar = np.interp(angle, (45,150), (100,650))

        color = (255,0, 255)
        if per == 100:
            color = (0, 0, 255)
            if dir == 1:
                count += 0.5
                dir = 0
        if per ==0:
            color = (0, 0, 255)
            if dir == 0:
                count += 0.5
                dir = 1
    #print (count)

        #draw count
        cv2.rectangle (img,(0,450),(250,720),(0,255,0), cv2.FILLED)
        cv2.putText(img, f'{int(count)}', (47, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)


        #draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'fps:', (50, 180), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
    cv2.putText(img, f'{int(fps)}', (50,260), cv2.FONT_HERSHEY_PLAIN, 5 , (255,0,0), 5)
    cv2.imshow("image",img)
    cv2.waitKey(1)