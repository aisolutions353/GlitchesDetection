import cv2
import numpy as np

video = cv2.VideoCapture("input_videos/video 5.mp4")
video_name = "tom_cruse"
glitch_min_size = 700
output_folder = "output"

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
                         10, size)

def addGlitches(image):
    # Define glitch parameters
    glitch_intensity = 20  # Adjust the intensity of the glitch effect
    num_boxes = 100  # Adjust the number of boxes
    box_size_range = (5, 30)  # Adjust the range of box sizes

    # Randomly introduce boxes of pixels
    for _ in range(num_boxes):
        x1, y1 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
        box_width, box_height = np.random.randint(box_size_range[0], box_size_range[1]), np.random.randint(
            box_size_range[0], box_size_range[1])
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        # Create a box of pixels
        x2, y2 = x1 + box_width, y1 + box_height
        image[y1:y2, x1:x2] = color

    return image


def save_annotation(glitched_image, binary_image, annotation_path, glitched_image_path):

    # Create a file to save YOLO annotations
    with open(annotation_path, 'w') as file:

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check for all the glitches
        for contour in contours:

            # Detect bog glitches not small
            if cv2.contourArea(contour) >= glitch_min_size:  # You can adjust this threshold as needed
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
        cv2.imshow(f"{video_name}_frame", frame)

        # Save it
        cv2.imwrite(f"{video_name}_test.jpg", frame)

        # Adding glitches
        glitched = addGlitches(frame)

        # Save glitched image
        cv2.imwrite(f"{video_name}_test_glitch.jpg", glitched)

        frame = cv2.imread(f"{video_name}_test.jpg")
        glitched = cv2.imread(f"{video_name}_test_glitch.jpg")

        # Make the difference
        difference = cv2.absdiff(frame, glitched)

        # Get the threshold images
        gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        try:

            # Saving every 50th frame
            if frames_count % 50 == 0:
                save_annotation(glitched, thresh, f"{output_folder}/{video_name}_{saving_count}.txt",
                                f"{output_folder}/{video_name}_{saving_count}.jpg")
                saving_count += 1
                print(f"{saving_count} frame saved!")

        except:
            print(f"Error: annotations/{video_name}_{saving_count}.jpg not saved")

        # Count the frame number
        frames_count += 1

        # Display it
        cv2.imshow("glitched frame", glitched)

        # cv2.imshow("difference", difference)
        cv2.imshow("difference", thresh)

        result.write(glitched)

        # Check for the user pressed the q
        if cv2.waitKey(10) == ord('q'):
            break
    else:
        break

result.release()
video.release()
cv2.destroyAllWindows()
