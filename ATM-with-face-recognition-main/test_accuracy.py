import cv2
import os
import numpy as np

# Path to the images folder where user images are stored
images_folder = r"C:\Users\Dell\Downloads\ATM-with-face-recognition-main (1)\images"

# Function to load images from the images folder based on the username
def load_user_images(username, folder):
    images = []
    labels = []
    user_folder = os.path.join(folder, username)
    
    if not os.path.isdir(user_folder):
        print(f"Error: User folder for {username} not found.")
        return None, None
    
    img_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg') or f.endswith('.png')]
    
    for img_name in img_files:
        img_path = os.path.join(user_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (100, 100)))  # Resize to 100x100 for better resolution
            labels.append(0)  # Assuming only one label for the current user
    
    if len(images) == 0:
        print(f"No images found for {username}.")
        return None, None

    return images, labels

# Train the recognizer using the loaded images
def train_recognizer(X_train, y_train):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, np.array(y_train))
    return recognizer

# Function to train the model with images from the images folder
def train_with_images(folder):
    images = []
    labels = []
    label = 0  # Starting label for the first user

    for username in os.listdir(folder):
        user_folder = os.path.join(folder, username)
        
        # Ensure the folder contains images
        if not os.path.isdir(user_folder):
            continue
        
        for img_name in os.listdir(user_folder):
            if img_name.endswith(".jpg") or img_name.endswith(".png"):
                img_path = os.path.join(user_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (100, 100)))  # Resize to 100x100 for better resolution
                    labels.append(label)
        label += 1  # Increment the label for each new user
    
    recognizer = train_recognizer(images, labels)
    recognizer.save("trainer.yml")  # Save the trained model as 'trainer.yml'
    print("Training complete. Model saved as 'trainer.yml'.")

# Test the recognizer with the webcam feed and the given threshold for confidence
def test_accuracy_with_webcam(username, threshold=170):
    # Load the user's images and labels
    images, labels = load_user_images(username, images_folder)
    
    if images is None or labels is None:
        print(f"Error: Could not load images for {username}.")
        return

    # Train the model using the loaded images
    recognizer = train_recognizer(images, labels)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the webcam feed.")
    
    total_predictions = 0
    correct_predictions = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))  # Increase detection size

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the face from the image
            face_region = gray[y:y + h, x:x + w]

            # Resize the face to match the training images size
            face_resized = cv2.resize(face_region, (100, 100))

            # Predict the label for the detected face
            label, confidence = recognizer.predict(face_resized)

            # If the confidence is below the threshold, consider it a rejection
            if confidence < threshold:
                if label == 0:  # Only accept the label if it's the correct username (0)
                    correct_predictions += 1
            total_predictions += 1

        # Display the resulting frame
        cv2.imshow('Face Recognition', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate and display the accuracy
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No faces detected during the webcam session.")

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter the username: ")  # Prompt user for the username
    threshold = 137  # Set a default threshold for confidence
    
    # Train the model with images from the images folder
    train_with_images(images_folder)
    
    # Test face recognition with the webcam
    test_accuracy_with_webcam(username, threshold)
