import cv2
import numpy as np
import joblib


# Load the haarcascade classifier for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the pre-trained LBPH model
lbph_model = cv2.face.LBPHFaceRecognizer_create()
lbph_model.read("lbph_modelFYPHaar.yml")

# Load the trained SVM classifier (recognizer)
recognizer_path = "modelALL\svm_modelFYPHaar.pkl"
recognizer = joblib.load(recognizer_path)

# Create a dictionary to map label numbers to names
label_names = {
    1: "nasim",
    2: "khai",
    3: "hariz",
    16: "mirun",
    20: "tuan arif"
}

# Create a function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces, gray

# Create a function to recognize faces in an image
def recognize_faces(image, gray):
    faces, _ = detect_faces(image)
    predictions = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_region = cv2.resize(face_region, (100, 100))
        label, _ = lbph_model.predict(face_region)
        predictions.append(label)
    return predictions

# Open the camera
video_capture = cv2.VideoCapture(0)

while True:
    # Read the video frame from the camera
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face recognition on the frame
    predictions = recognize_faces(frame, gray)
    
    # Draw bounding boxes and labels on the frame
    faces, _ = detect_faces(frame)
    for (x, y, w, h), label in zip(faces, predictions):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label_name = label_names.get(label, "Unknown")
        cv2.putText(frame, label_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the windows
video_capture.release()
cv2.destroyAllWindows()
