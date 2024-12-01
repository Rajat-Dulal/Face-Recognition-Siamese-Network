import cv2
import os
import shutil

def extract_face(img_path):

    faces_list = []

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x,y,w,h) in enumerate(faces):
        # print(i, f"({x},{y},{w},{h})")
        face = img[y:y+h, x:x+w]
        faces_list.append(face)
        
    print(f"{len(faces_list)} face(s) extracted from the image.")
    # print("Faces Extracted from the image.")
    return faces_list












