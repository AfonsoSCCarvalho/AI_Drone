import cv2
import time
import mediapipe as mp
import math
import numpy as np
from djitellopy import tello


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        # Convert BGR image to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                # Draw landmarks on the image
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosePosition(self, img, draw=True, BBOX=True):
        self.lmList = []
        xList = []
        yList = []
        bbox = []
        area = 0
        okay = False
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])  # Only x and y coordinates
                if id < 12 or id == 23 or id == 24:  # Track only up to the hands
                    xList.append(cx)
                    yList.append(cy)
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)
                    bbox = xmin, ymin, xmax, ymax
                if id == 23 and lm.visibility > 0.15:
                    okay = True
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            if BBOX:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
                cv2.rectangle(img, (bbox[0] - 10, bbox[1] - 50), (bbox[2] + 10, bbox[3]), (0, 255, 0), 2)
                cx = (bbox[2] + bbox[0]) // 2
                cy = (bbox[3] + bbox[1]) // 2
                cv2.rectangle(img, (cx - 50, cy - 50), (cx + 50, cy + 50), (255, 255, 0), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList, bbox, okay, area

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angles
        angle1_2 = math.degrees(math.atan2(y1 - y2, x1 - x2))
        if angle1_2 < 0:
            angle1_2 += 360
        angle2_3 = math.degrees(math.atan2(y3 - y2, x3 - x2))
        if angle2_3 < 0:
            angle2_3 += 360

        angle = abs(angle2_3 - angle1_2)
        if angle < 0:
            angle += 360

        # Draw angles on the image
        if draw:
            cv2.ellipse(img, (x2, y2), (50, 50), 0, angle1_2, angle2_3, (0, 0, 255), 10)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, f'{int(angle)}', (x2 + 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            if angle >= 100:
                cv2.putText(img, 'o', (x2 + 80, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            else:
                cv2.putText(img, 'o', (x2 + 60, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return angle

    def trackBody(self, okay, lmList, area, w, h, pid, pErrorx, pErrory):
        fbRange = [70, 100]
        x, y = lmList[0][1], lmList[0][2]
        fb = 0
        error_x = x - w // 2  # Error in x relative to the center
        error_y = -(y - h // 2 + 40)
        speed = pid[0] * error_x + pid[1] * (error_x - pErrorx)
        speed = int(np.clip(speed, -100, 100))  # Clip speed

        sobe = pid[3] * error_y + pid[4] * (error_y - pErrory)
        sobe = int(np.clip(sobe, -80, 80))

        if fbRange[0] < area < fbRange[1]:
            fb = 0
        elif area > fbRange[1]:
            fb = -20
        elif area < fbRange[0] and area != 0:
            fb = 20
        if x == 0:
            error_x = 0
            error_y = 0
            sobe = 0
            speed = 0

        return error_x, error_y, fb, sobe, speed


def main():
    # Capture video from file
    cap = cv2.VideoCapture('Imgs/1.mp4')
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Find pose and landmarks in the frame
        img = detector.findPose(img)
        lmList, bbox, okay, area = detector.findPosePosition(img, draw=False)

        if lmList:
            # Draw circle on a specific landmark
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
