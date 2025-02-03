import cv2
import numpy as np
import os

# Path to the images folder
image_folder = 'C:/Users/Dell/Downloads/ATM-with-face-recognition-main (1)/images'

# Initialize LBPH recognizer with adjusted parameters
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)

def load_images_from_folder(folder, test_split=0.2):
    faces = []
    labels = []
    label_dict = {}  # Dictionary to map usernames to integer labels
    label_counter = 0  # Counter for unique integer labels
    
    for person_name in os.listdir(folder):
        person_folder = os.path.join(folder, person_name)
        
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is None:
                    continue  # Skip invalid images
                
                # Apply histogram equalization to improve image contrast
                img = cv2.equalizeHist(img)
                
                faces.append(img)
                
                if person_name not in label_dict:
                    label_dict[person_name] = label_counter
                    label_counter += 1
                
                labels.append(label_dict[person_name])  # Append the integer label
    
    return faces, labels, label_dict

def train_model(faces, labels):
    """
    Trains the LBPH recognizer on the loaded images and labels.
    """
    recognizer.train(faces, np.array(labels))
    recognizer.save('trainer.yml')  # Save the trained model

def detect_face(username):
    """
    Function to detect face from the webcam and compare it with the trained model.
    """
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Crop and resize the detected face to a higher resolution (100x100)
            face_resized = cv2.resize(gray[y:y+h, x:x+w], (100, 100))  # Resize face to 100x100 or higher

            # Predict the label and confidence
            label, confidence = recognizer.predict(face_resized)
            
            # Display label and confidence on the frame
            cv2.putText(frame, f"Label: {label}, Confidence: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if confidence is below threshold for successful match
            if confidence < 130:
                cv2.putText(frame, f"Hello {username}!", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Webcam", frame)
        
        # Exit on pressing 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Load images and train the model
    faces, labels, label_dict = load_images_from_folder(image_folder)
    if faces:
        print("Training the model...")
        train_model(faces, labels)
        print("Training complete!")
    else:
        print("No faces found in the dataset.")
    
    # Test face recognition on webcam
    username = input("Enter your username for webcam recognition: ")
    if username in label_dict:
        detect_face(username)
    else:
        print(f"User '{username}' not found in the dataset.")

if __name__ == '__main__':
    main()
