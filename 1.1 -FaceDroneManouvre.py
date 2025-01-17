import cv2
import mediapipe as mp
import time

# Initialize MediaPipe drawing and holistic modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Initialize previous time for FPS calculation
pTime = 0

# Open webcam for input
cap = cv2.VideoCapture(0)

# Set up the holistic model with minimum detection and tracking confidence
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read frame from webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for selfie-view and convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        results = holistic.process(image)

        # Draw landmark annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        # Draw left hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Draw right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Display FPS on the image
        cv2.putText(image, f'fps:', (50, 180), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
        cv2.putText(image, f'{int(fps)}', (50, 260), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Display the image
        cv2.imshow('MediaPipe Holistic', image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam
cap.release()
