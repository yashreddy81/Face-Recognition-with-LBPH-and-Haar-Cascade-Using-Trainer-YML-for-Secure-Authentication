import cv2
import os
import numpy as np
import random

# Function to load images from the given folder
def load_images_from_folder(folder, test_split=0.2):
    images = []
    labels = []
    users = os.listdir(folder)
    
    for label, user in enumerate(users):
        user_folder = os.path.join(folder, user)
        if os.path.isdir(user_folder):
            img_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg') or f.endswith('.png')]
            
            for img_name in img_files:
                img_path = os.path.join(user_folder, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(cv2.resize(img, (100, 100)))  # Resize to 100x100
                    labels.append(label)
    
    # Shuffle and split the dataset into training and testing sets
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    
    # Splitting dataset into training and testing
    split_idx = int(len(images) * (1 - test_split))  # 80% for training
    X_train, X_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    return X_train, X_test, y_train, y_test

# Train the recognizer using training data
def train_recognizer(X_train, y_train):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, np.array(y_train))
    return recognizer

# Test the recognizer with the testing data
def test_recognizer(recognizer, X_test, y_test):
    correct_predictions = 0
    total = len(X_test)

    for i in range(total):
        label, confidence = recognizer.predict(X_test[i])  # Predict the label for the test image
        if label == y_test[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / total) * 100
    print(f"Testing Accuracy: {accuracy:.2f}%")
    return accuracy

def test_accuracy_with_webcam(recognizer, threshold=100):
    cap = cv2.VideoCapture(0)  # Use default webcam (set to the index of the camera, 0 is default)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))  # Changed to (100, 100)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))  # Resize the face image for recognition to 100x100
            label, confidence = recognizer.predict(face_resized)
            
            if confidence < threshold:  # If confidence is lower than threshold, consider it a match
                label_text = f"Person: {label}, Confidence: {confidence:.2f}"
                color = (0, 255, 0)  # Green for match
            else:
                label_text = f"Unknown, Confidence: {confidence:.2f}"
                color = (0, 0, 255)  # Red for unknown
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('Live Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    folder = 'C:/Users/Dell/Downloads/ATM-with-face-recognition-main (1)/images'  # Path to the dataset folder
    test_split = 0.2  # 80% for training, 20% for testing
    
    # Load the dataset and split into training and testing sets
    X_train, X_test, y_train, y_test = load_images_from_folder(folder, test_split)
    
    # Train the model
    recognizer = train_recognizer(X_train, y_train)
    
    # Test the model
    accuracy = test_recognizer(recognizer, X_test, y_test)
    print(f"Final Testing Accuracy: {accuracy:.2f}%")
    
    # Test the model with live webcam feed
    test_accuracy_with_webcam(recognizer, threshold=100)  # Set the threshold as needed

if __name__ == "__main__":
    main()
