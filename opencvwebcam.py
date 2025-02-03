import cv2
import os

def detect_face(username):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")  # Ensure the trainer file exists

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    user_folder = os.path.join("images", username)
    if not os.path.exists(user_folder):
        print("No images found for this user.")
        return False

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face, (100, 100))
            label, confidence = recognizer.predict(face_resized)
            
            if confidence < 117.4:  # Adjust threshold as needed
                print(f"Welcome, {username}!")
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print("Face not recognized.")
        
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False
