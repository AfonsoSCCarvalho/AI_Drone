import cv2
import mediapipe as mp
import time

# Initialize MediaPipe drawing and holistic modules
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

pTime = 0  # Initialize previous time for FPS calculation

# Open webcam for input
cap = cv2.VideoCapture(0)  # Change '0' to the path of a video file if you want to use a video

# Set up the holistic model with minimum detection and tracking confidence
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect the landmarks
        results = holistic.process(image)

        # Convert the RGB image back to BGR
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

        cv2.putText(image, 'FPS:', (50, 180), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
        cv2.putText(image, str(int(fps)), (50, 260), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

        # Display the image
        cv2.imshow('MediaPipe Holistic', image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam
cap.release()
cv2.destroyAllWindows()
