import cv2
import numpy as np
import os #needed for file access

try:
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
except Exception as e:
    print(f"Error loading cascade classifier from standard path: {e}")
    print("Attempting to load from local directory (needs manual download)...")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("FATAL ERROR: Could not open video stream or camera (VideoCapture(0) failed).")
    print("Check if another program is using the camera or if the camera is connected.")
    exit()

print("Camera initialized successfully. Press 'Q' to quit the window.")

while True:
    c_rec, d_image = cap.read()
    
    if not c_rec:
        print("Warning: Could not read frame from video stream. Exiting loop.")
        break
    
    gray_image = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray_image, 
        scaleFactor=1.3, 
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(d_image, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('Face Detector - Press Q to Quit', d_image)
    
    key = cv2.waitKey(1) & 0xff 
    
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Program closed successfully.")