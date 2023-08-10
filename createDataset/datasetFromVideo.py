#create frame by frame from dataset
import cv2
import os

# Path to the MP4 video file
video_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetVideo\\hariz.mp4"

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture(video_path)

# Check if the video capture is successful
if not cap.isOpened():
    print("Failed to open video.")
    exit()

# Get the frames per second (fps) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Create the output folder if it doesn't exist
output_folder = "Dvideo"
os.makedirs(output_folder, exist_ok=True)

# Create a loop to read frames from the video and save them as images
frame_count = 0
while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    # Break the loop if there are no more frames
    if not ret:
        break

    # Increment the frame count
    frame_count += 1

    # Process the frame (you can perform any additional operations here)

    # Display the frame (optional)
    cv2.imshow("Frame", frame)

    # Save the frame as an image
    frame_name = os.path.join(output_folder, f"3_{frame_count:04d}hariz.jpg")  # Adjust the filename format as needed
    cv2.imwrite(frame_name, frame)

    # Delay between frames based on the video's fps (optional)
    delay = int(1000 / fps)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
