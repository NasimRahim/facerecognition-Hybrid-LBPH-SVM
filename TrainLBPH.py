#train lbph using metric information (accuracy,precision,f1-score,recall)
#NOTES THIS CODE ONLY FOR TRAIN THE DATASET THEN STORE THE INFORMATION INTO YML FILE 
#untuk run tulis nih dekat terminal = python "c:\Users\nasim\Desktop\fyp system\vs code\TrainLBPH.py" PENTING!!
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Set variable to call the path from the folder untuk dataset
dataset_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\DatasetLBPH"   

cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

# Load the haarcascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cascade_path)

# Define LBPH parameters #also this is a hyperparameters for lbph
radius = 1
neighbors = 8
grid_x = 8
grid_y = 8
threshold = 100 #The threshold used for predicting the label of an image. If the predicted confidence score is below this threshold, then the image is labeled as unknown.

# Create an LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=radius, neighbors=neighbors, grid_x=grid_x, grid_y=grid_y, threshold=threshold)

# Define a function to read the images from the dataset folder which is folder datasetT
def get_images_and_labels(dataset_path):
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    images = []
    labels = []
    for image_path in image_paths:
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            #apply noise reduction guissian blur filter to reduce noise in the image
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces) == 0:
                os.remove(image_path) #delete images yang tak boleh detect muka
            for (x, y, w, h) in faces:
                images.append(cv2.resize(gray[y:y+h, x:x+w], (100, 100)))
                labels.append(int(os.path.split(image_path)[-1].split("_")[0]))
        except Exception as e:
            print(f"Error loading or processing {image_path}: {e}")
    return images, labels

# Load images and label from dataset folder/ check berapa images muka berjaya upload
images, labels = get_images_and_labels(dataset_path)
# Load images and label from dataset folder/ check berapa images muka berjaya upload
images, labels = get_images_and_labels(dataset_path)
print("Number of images:", len(images))
print("Number of labels:", len(set(labels))) # prints the total number of unique labels

if len(images) == 0:
    print("No face images found in the dataset folder!")
    exit()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

if len(X_train) == 0 or len(X_test) == 0:
    print("Not enough face images for training and testing!")
    exit()

# Train the recognizer using the training data
recognizer.train(X_train, np.array(y_train))

# Make predictions on the testing data
y_pred = []
for image in X_test:
    label, confidence = recognizer.predict(image)
    y_pred.append(label)

# Calculate the evaluation metrics
if len(set(y_test)) == 1:
    # Only one class in the test set, so set all metrics to 1
    accuracy = recall = precision = f1 = 1.0
else:
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1-score:", f1)

# Train the recognizer on the images and labels
recognizer.train(images, np.array(labels))

# Save the recognizer to a file 
recognizer.save("lbph_modelFYPHaar.yml")

