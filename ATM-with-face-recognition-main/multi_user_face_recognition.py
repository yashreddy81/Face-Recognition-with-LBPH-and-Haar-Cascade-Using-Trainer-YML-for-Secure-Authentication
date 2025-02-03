import cv2
import numpy as np
import os
from datetime import datetime

# Load and preprocess images from the 'images' folder
def preprocess_image(image_path, size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, size)
    return img_resized

def load_training_images(username, folder='images'):
    images = []
    labels = []
    user_folder = os.path.join(folder, username)
    
    if not os.path.exists(user_folder):
        print(f"Folder for user '{username}' does not exist!")
        return None, None
    
    for label, img_name in enumerate(os.listdir(user_folder)):
        if img_name.endswith('.jpg') or img_name.endswith('.png'):
            img_path = os.path.join(user_folder, img_name)
            img_resized = preprocess_image(img_path)
            images.append(img_resized)
            labels.append(label)
    return images, labels

# Main face recognition function
def recognize_face(username):
    # Load training data (faces & labels)
    train_images, labels = load_training_images(username)
    if train_images is None or labels is None:
        return  # Exit if no images were found for the user
    
    # Create the face recognizer and train it with images
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(train_images, np.array(labels))

    # Initialize face detection using Haar Cascade
    face_cascade = cv2.CascadeClassifier(r'C:\Users\Dell\Downloads\ATM-with-face-recognition-main (1)\ATM-with-face-recognition-main\haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100, 100))  # Resize face to match the training image size
            label, confidence = recognizer.predict(face_resized)
            
            if confidence < 92:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for match
                cv2.putText(frame, f"User {username}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for no match
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        
        # Break the loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Logging functionality
def log_message(message):
    log_file_path = "recognition_log.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{datetime.now()}: {message}\n")

# Example usage
username = "your_username"
log_message("Face recognition started")
recognize_face(username)
