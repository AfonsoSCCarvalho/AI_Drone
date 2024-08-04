# Python program to save a
# video using OpenCV

import cv2

# Create an object to read from the camera
video = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not video.isOpened():
    print("Error reading video file")

# Get the resolution of the video frames
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the size (resolution) of the video frames
size = (frame_width, frame_height)

# Create a VideoWriter object to save the video
# The output is stored in 'filename.avi' file with MJPG codec at 10 FPS
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

# Loop to read and display frames from the camera
while True:
    ret, frame = video.read()  # Read a frame from the camera

    if ret:  # Check if the frame is read correctly
        result.write(frame)  # Write the frame into the file 'filename.avi'
        cv2.imshow('Frame', frame)  # Display the frame

        # Press 's' on the keyboard to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    else:
        break  # Break the loop if the frame is not read correctly

# Release the video capture and video write objects
video.release()
result.release()

# Close all the OpenCV windows
cv2.destroyAllWindows()

print("The video was successfully saved")
