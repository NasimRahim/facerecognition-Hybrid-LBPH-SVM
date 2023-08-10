import os
import cv2

# Load the Haar cascade xml file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set video width
cap.set(4, 480)  # Set video height

# Enter person id
id = input("\nEnter user id: ")
name = input("Enter your name: ")

# Initialize face count
count = 0   
while True:
    ret, img = cap.read()

    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and crop them
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_img = gray[y:y+h, x:x+w]

        # Save cropped face image to the specified folder
        cv2.imwrite(f"C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetBiasa\\{id}_{count}_{name}.jpg", face_img)

    count += 1

    cv2.imshow('image', img)
    
    # Press ESC to exit
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

    # Set the number of images to capture
    elif count >= 250:
        break

# Display the number of face images captured
dataset_folder = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetBiasa"
image_count = len([filename for filename in os.listdir(dataset_folder) if filename.endswith(".jpg") and filename.startswith(f"{id}_")])

print(f"Total number of face images captured for user {id}: {image_count}")

cap.release()
cv2.destroyAllWindows()
