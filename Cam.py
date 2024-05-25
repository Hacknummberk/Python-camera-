import cv2
import numpy as np

# Create a synthetic image with predefined objects
image = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Draw a face
cv2.rectangle(image, (200, 200), (300, 300), (0, 255, 0), -1)  # Draw an object
cv2.circle(image, (100, 300), 20, (0, 0, 255), -1)  # Draw a light source

# Convert the synthetic image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load pre-trained models for face detection and object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Detect objects (smiles)
objects = object_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around faces and objects
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
for (x, y, w, h) in objects:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the resulting image to a file
cv2.imwrite('synthetic_image.jpg', image)
