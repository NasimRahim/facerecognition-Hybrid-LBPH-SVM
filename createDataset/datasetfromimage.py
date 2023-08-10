import cv2
import os

# Load the Haar cascade xml file for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Enter person id
id = input("\nEnter user id: ")
name = input("Enter your name: ")

# Initialize face count
count = 0

# Specify the source folder path where the images are located
source_folder = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\Dvideo"

# Specify the destination folder path where the cropped face images will be saved
destination_folder = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\D2video"

# Iterate over the images in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".jpg") and filename.startswith(f"{id}_"):
        img_path = os.path.join(source_folder, filename)
        img = cv2.imread(img_path)

        # Detect faces in the color image
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces and crop them
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_img = img[y:y+h, x:x+w]

            # Save cropped face image to the destination folder with a new name
            new_filename = f"{id}_{count}_{name}.jpg"
            new_img_path = os.path.join(destination_folder, new_filename)
            cv2.imwrite(new_img_path, face_img)

            count += 1

        cv2.imshow('image', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Display the number of face images captured
destination_image_count = len([filename for filename in os.listdir(destination_folder) if filename.endswith(".jpg") and filename.startswith(f"{id}_")])
print(f"Total number of face images captured for user {id}: {destination_image_count}")
