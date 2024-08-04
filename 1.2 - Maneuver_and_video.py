import cv2
import time
from Modules import ModuleHolistic as hm
from djitellopy import tello

# Connect to Tello drone
me = tello.Tello()
me.connect()
print(me.get_battery())

# Start video stream and take off
me.streamon()
me.takeoff()
time.sleep(1)

# Get video capture from Tello
vid = me.get_video_capture()

# Prepare output video
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps = int(vid.get(cv2.CAP_PROP_FPS))
vid_width, vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('results.avi', codec, vid_fps, (vid_width, vid_height))

# Initialize holistic detector
detector = hm.holisticDetector()
color = (255, 0, 255)
pTime = 0

# Set parameters for flight control
w, h = 360, 240
fbRange = [6200, 6800]
pid = [0.4, 0.4, 0, 0.4, 0.4, 0]
pErrorx, pErrory = 0, 0

# Initialize variables for photo and maneuver timing
last_recorded_time_photo = 0
last_recorded_time_manobra = 0
img_counter = 0
Foto = False

cTime = 0  # Case where someone already has arms extended

modo_Maneuvers = False  # Mode for maneuvers

while True:
    # Read frame from video
    success, img = vid.read()
    img = cv2.resize(img, (w, h))

    # Detect pose and hand positions
    img = detector.findPose(img)
    lmList, bbox, okay, area = detector.findPosePosition(img, draw=False, BBOX=True)
    lmListHand, bboxh = detector.findHandPosi(img)
    cTime = time.time()

    if len(lmList) != 0 and okay:
        # Check arm angles
        angle_d = detector.findAngle(img, 11, 12, 14, draw=False)
        angle_esq = detector.findAngle(img, 12, 11, 13, draw=False)

        # If both arms are extended, set Foto flag
        if 170 < angle_esq < 190 and 170 < angle_d < 190:
            Foto = True
            last_recorded_time_photo = cTime

        # Control drone movement based on body tracking
        pErrorx, pErrory, fb, sobe, speed = detector.trackBody(okay, lmList, area, w, h, pid, pErrorx, pErrory)
        me.send_rc_control(0, fb, sobe, speed)

        # Check for hand above head for maneuver mode toggle
        if lmList[19][2] < lmList[1][2] and cTime - last_recorded_time_manobra >= 1.0:
            print()
            modo_Maneuvers = not modo_Maneuvers
            last_recorded_time_manobra = cTime

        # Execute maneuvers if in maneuver mode
        if modo_Maneuvers:
            dentro = detector.dentroBBox(bboxh, modo_Maneuvers, img)
            if len(lmListHand) != 0 and dentro:
                fingersUp = detector.fingersUp()
                if fingersUp == [0, 0, 0, 0, 0]:
                    me.land()
                    break
                if fingersUp == [0, 1, 0, 0, 0]:
                    me.flip_forward()

    # Calculate and display FPS
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Save photo if Foto flag is set and at least 1 second has passed
    if cTime - last_recorded_time_photo >= 1.0 and Foto:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, img)
        print("{} written!".format(img_name))
        img_counter += 1
        Foto = False

    # Display image and write to output video
    cv2.imshow("image", img)
    out.write(img)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

# Release video and close windows
vid.release()
out.release()
cv2.destroyAllWindows()
