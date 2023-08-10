import cv2
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt


# Set variable to call the path from the folder untuk datasetq
dataset_path = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\D2video"

# Load the haarcascade classifier for face detection
cascade_path = "fyp\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create a function to read the images from the dataset folder which is folder datasetT
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
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces) == 0:
                os.remove(image_path) #delete images yang tak boleh detect muka
            for (x, y, w, h) in faces:
                img = cv2.resize(gray[y:y+h, x:x+w], (100, 100))
                img = img.flatten() # flatten the image into a 1D array
                images.append(img)
                labels.append(int(os.path.split(image_path)[-1].split("_")[0]))
                #print(img)   #display number dia ja 
        except Exception as e:
            print(f"Error loading or processing {image_path}: {e}")
    return images, labels
    

# Load images and label from dataset folder/ check berapa images muka berjaya upload
images, labels = get_images_and_labels(dataset_path)
print("Number of images:", len(images))
print("Number of labels:", len(set(labels))) # prints the total number of unique labels

# Convert the data into a numpy array
X = np.array(images)
y = np.array(labels)

# Define the number of cross-validation folds
cv_folds = [10, 5, 4, 3]

# Train the SVM classifier using cross-validation
error_rates = []
for cv_fold in cv_folds:
    clf = SVC(kernel='linear')
    scores = cross_val_score(clf, X, y, cv=cv_fold)
    error_rate = 1 - np.mean(scores)
    error_rates.append(error_rate)

# Plot the graph
plt.plot(cv_folds, error_rates, marker='o')
plt.xlabel('Cross-validation fold')
plt.ylabel('Error Rate')
plt.title('Error Rate for Different Cross-validation Folds ')
plt.show()

#save the model
clf.fit(X, y)
with open("svm_modelFYPHaar.pkl", "wb") as f: #wb write binary
    pickle.dump(clf, f)
