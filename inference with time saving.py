import cv2
from ultralytics import YOLO
import time

# Create an object to read from camera
video = cv2.VideoCapture("input_videos/video 7.mp4")

# Loading the model
model = YOLO("best (6).pt")

# We need to check if the camera is opened previously or not
if not video.isOpened():
    print("Error reading video file")
    exit(0)

# We need to set resolutions, so convert them from float to integer
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Define VideoWriter object to create a frame of the output video
result = cv2.VideoWriter('glitches_detected_result.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)

# Create a text file to save the detection time, count, and frame number
with open('detection_results.txt', 'w') as txt_file:
    frame_number = 0  # Initialize frame number
    detections = 0
    start_time = time.time()  # Start time for the video

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            frame_number += 1  # Increment frame number

            # Perform detection
            res = model.predict(frame)

            detections = len(res[0].boxes.data.tolist())

            # Draw and save the frame with detections
            frame = res[0].plot()

            # Saving the
            result.write(frame)

            # Display the frame saved in the file
            cv2.imshow('Frame', frame)

        else:
            break

        end_time = time.time()  # End time for the video

        # Calculate and write the time of the video where detection was done to the text file
        elapsed_time = end_time - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Only save the annotation if the detection is not empty
        if detections != 0:
            txt_file.write(f"Frame {frame_number}: Time = {int(hours)}h {int(minutes)}m {int(seconds)}s : Detections = {detections}\n")

        # Press S on the keyboard to stop the process
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# When everything is done, release the video capture and video write objects
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
