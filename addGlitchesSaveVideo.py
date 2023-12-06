import glitchart

import cv2

video = cv2.VideoCapture("input_videos/video 5.mp4")
video_name = "tom_cruse"

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(f'{video_name}_glitched.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, size)


def save_annotation(glitched_image, binary_image, annotation_path, glitched_image_path):
    # Create a file to save YOLO annotations
    with open(annotation_path, 'w') as file:

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for all the glitches
        for contour in contours:

            # Detect bog glitches not small
            if cv2.contourArea(contour) > 700:  # You can adjust this threshold as needed
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate the YOLO format values
                center_x = (x + w / 2) / binary_image.shape[1]
                center_y = (y + h / 2) / binary_image.shape[0]
                width = w / binary_image.shape[1]
                height = h / binary_image.shape[0]

                # Write the annotation to the file
                file.write(f'0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n')

    # Save the resulting image with bounding boxes
    cv2.imwrite(glitched_image_path, glitched_image)


frames_count = 0
saving_count = 0

while video.isOpened():

    ret, frame = video.read()

    if ret:

        # Displaying the original frame
        cv2.imshow("frame", frame)

        # Save it
        cv2.imwrite("test.jpg", frame)

        # Make the glitched version
        glitchart.jpeg("test.jpg", min_amount=0, max_amount=10)

        # Load the glitched image
        glitched = cv2.imread("test_glitch.jpg")

        # Make the difference
        difference = cv2.absdiff(frame, glitched)

        # Get the threshold images
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Saving every 50th frame
        if frames_count % 30 == 0:
            result.write(glitched)
        else:
            result.write(frame)

        # Count the frame number
        frames_count += 1

        # Display it
        cv2.imshow("glitched frame", glitched)

        # cv2.imshow("difference", difference)
        cv2.imshow("difference", thresh)

        # Check for the user pressed the q
        if cv2.waitKey(1) == ord('q'):
            break

    else:
        break

video.release()
result.release()
cv2.destroyAllWindows()
