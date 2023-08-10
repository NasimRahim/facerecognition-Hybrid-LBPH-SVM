import os
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set video width to 1280
cap.set(4, 720)   # Set video height to 720

# Enter person ID
id = input("\nEnter user ID: ")
name = input("Enter your name: ")

# Initialize face count
count = 0
while True:
    ret, img = cap.read()

    count += 1

    # Save the image to the desired folder
    image_path = f"C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetT\\{id}_{count}_{name}.jpg"
    cv2.imwrite(image_path, img)

    # Resize the image to passport size (e.g., 35x45 pixels)
    resized_img = cv2.resize(img, (35, 45))

    # Save the resized image to a separate folder
    resized_image_path = f"C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\resized_images\\{id}_{count}_{name}.jpg"
    cv2.imwrite(resized_image_path, resized_img)

    cv2.imshow('image', img)

    # Press ESC to exit
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

    # Set the maximum number of images to capture
    elif count >= 100:
        break

# Display the number of face images captured
dataset_folder = "C:\\Users\\nasim\\Desktop\\fyp system\\vs code\\datasetT"
image_count = sum(1 for filename in os.listdir(dataset_folder) if filename.endswith(".jpg") and filename.startswith(f"{id}_"))
print(f"\nTotal number of face images captured for user {id}: {image_count}")

cap.release()
cv2.destroyAllWindows()
