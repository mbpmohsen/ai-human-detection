import numpy as np
import cv2

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # Check if faces is empty, not ()
    if len(faces) == 0:
        return img
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Fixed typo: cvw -> cv2

    return img

# Open webcam video stream
cap = cv2.VideoCapture(1)  # Adjust index if necessary

while True:
    ret, frame = cap.read()
    if not ret:  # Check if the frame is read correctly
        print("Error: Could not read frame.")
        break

    frame = detect_faces(frame)

    cv2.imshow("Video Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
