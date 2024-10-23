import numpy as np
import cv2

# Initialize the HOG descriptor and set SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open webcam video stream
cap = cv2.VideoCapture(1)  # Adjust index as needed

# Set up video output
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.0,
    (640, 480)
)

# Initialize a counter for detected people
person_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize frame for faster detection
    frame = cv2.resize(frame, (640, 480))

    # Detect people in the image
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Convert boxes to the format expected by OpenCV
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # Track and count detected people
    person_count = len(boxes)  # Update person count based on the number of detected boxes

    # Draw bounding boxes around detected persons and display the count
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Display the count of detected persons
    cv2.putText(frame, f'Count: {person_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Write the output video
    out.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and output
cap.release()
out.release()
cv2.destroyAllWindows()
