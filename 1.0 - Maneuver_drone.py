import cv2
import time
from Modules import ModulePose as pm
from djitellopy import tello

# Initialize Tello drone
me = tello.Tello()
me.connect()
print(me.get_battery())

# Start video stream and take off
me.streamon()
me.takeoff()
# me.send_rc_control(0,0,20,0)
# sleep(1)

# Initialize pose detector
detector = pm.poseDetector()
color = (255, 0, 255)
pTime = 0

# Set parameters for frame size and flight control
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0, 0.4, 0.4, 0]
pErrorx, pErrory = 0, 0

# Initialize variables for photo capture timing
last_recorded_time = 0
cTime = 0  # Current time
out_of_shoot_time = 0
img_counter = 0
Foto = False

while True:
    # Read frame from Tello
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))

    # Detect pose in the frame
    img = detector.findPose(img)
    lmList, bbox, okay, area = detector.findPosePosition(img, False)

    if len(lmList) != 0 and okay:
        # Check angles of both arms
        angle_esq = detector.findAngle(img, 11, 12, 14, draw=False)  # Right Arm
        angle_d = detector.findAngle(img, 12, 11, 13, draw=False)  # Left Arm

        # If both arms are extended, set Foto flag
        if 170 < angle_esq < 190 and 170 < angle_d < 190:
            Foto = True
            last_recorded_time = cTime

        # Control drone movement based on body tracking
        pErrorx, pErrory, fb, sobe, speed = detector.trackBody(okay, lmList, area, w, h, pid, pErrorx, pErrory)
        me.send_rc_control(0, fb, sobe, speed)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Save photo if Foto flag is set and at least 2 seconds have passed
    if cTime - last_recorded_time >= 2.0 and Foto:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
        Foto = False

    # Display image
    cv2.imshow("image", img)

    # Land the drone if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

# Release resources and close windows
cv2.destroyAllWindows()
