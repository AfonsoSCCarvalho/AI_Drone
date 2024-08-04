import cv2
import time
from Modules import ModulePose as pm

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

# Initialize video writer for saving the output video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
result = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# Initialize pose detector
detector = pm.poseDetector()
color = (255, 0, 255)
pTime = 0  # Initialize previous time for FPS calculation

# Set parameters for frame size and PID control
w, h = 360, 240
pid = [0.4, 0.4, 0, 0.4, 0.4, 0]
pErrorx, pErrory = 0, 0

img_counter = 0  # Initialize image counter for photo capture
last_recorded_time = time.time()  # Initialize last recorded time for photo capture
Foto = False  # Initialize flag for photo capture

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect pose in the frame
    img = detector.findPose(img)
    lmList, bbox, okay, area = detector.findPosePosition(img, draw=False)

    if len(lmList) != 0 and okay:
        # Track body and control drone movement
        pErrorx, pErrory, fb, sobe, speed = detector.trackBody(okay, lmList, area, w, h, pid, pErrorx, pErrory)

        # Calculate angles for both arms
        angle_esq = detector.findAngle(img, 11, 12, 14, draw=False)  # Right Arm
        angle_d = detector.findAngle(img, 12, 11, 13, draw=False)  # Left Arm

        # Check if both arms are extended to trigger photo capture
        if 170 < angle_esq < 190 and 170 < angle_d < 190:
            Foto = True
            print("meio")

        # Check if both hands are above the shoulders
        elif lmList[16][2] < lmList[12][2] and lmList[15][2] < lmList[11][2]:
            print("cima")

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Capture photo if both arms are extended for at least 1 second
    if cTime - last_recorded_time >= 1.0 and Foto:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
        Foto = False
        last_recorded_time = cTime

    # Display FPS on the image
    cv2.putText(img, 'FPS:', (50, 180), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
    cv2.putText(img, str(int(fps)), (50, 260), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    # Display the image
    cv2.imshow("image", img)

    # Break the loop if 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Release the video capture and video writer objects
cap.release()
result.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("The video was successfully saved")

