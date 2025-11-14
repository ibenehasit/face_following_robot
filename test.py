import cv2
import numpy as np
import matplotlib.pyplot as plt


#loading in pre-trained haar cascade classifer for faces and eyes
face_cascade = cv2.CascadeClassifier("/content/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('/content/haarcascade_eye.xml')


#detecting face module
def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
        
    return face_img

