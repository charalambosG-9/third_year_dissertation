# Imports
import cv2
import os
import numpy as np
import pickle
from pathlib import Path
import mtcnn
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--detector', dest='detector', type=str, help='Haar or mtcnn', default='haar')
parser.add_argument('--recognizer', dest='recognizer', type=str, help='lbph or fisher or eigen', default='eigen')
parser.add_argument('--folder', dest='folder', type=str, help='Add name of input folder', default='')
args = parser.parse_args()

FACE = (args.detector).lower() # "haar" or "mtcnn"
RECOGNIZER = (args.recognizer).lower() # "lbph" or "eigen" or "fisher"
input_folder = args.folder

# Initialize face detector
if FACE == "mtcnn":
    detector = mtcnn.MTCNN()
else:
    face_cascade_front = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
    face_cascade_profile = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

if RECOGNIZER == "lbph":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
elif RECOGNIZER == "eigen":
    recognizer = cv2.face.EigenFaceRecognizer_create()
elif RECOGNIZER == "fisher":
    recognizer = cv2.face.FisherFaceRecognizer_create()

Path("pickles").mkdir(parents=True, exist_ok=True)
Path("recognizers").mkdir(parents=True, exist_ok=True)

current_id = 0
label_ids = {}
y_labels = []
x_train = []
start_time = time.time()
images_kept = 0
images_with_multiple_faces = 0

for folder in os.listdir(input_folder):

    subfolder_path = os.path.join(input_folder, folder)
    images = os.listdir(subfolder_path)
    label = folder
    number_of_images = len(images)
    current_image = 0
    images_kept += number_of_images
    
    for image in images:

        if image.endswith(('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG', 'WEBP', 'webp')):

            current_image += 1
            print("Processing image " + str(current_image) + " of " + str(number_of_images) + " in folder " + folder)

            new_image = os.path.join(subfolder_path, image)
            img = cv2.imread(new_image)
            img_height, img_width, _ = img.shape
            if img_height > 1080 or img_width > 1920:
                img = cv2.resize(img,(1920,1080))
                img_height, img_width, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            if FACE == "mtcnn":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                faces = detector.detect_faces(img_bgr)
            else:
                faces = face_cascade_front.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                faces2 = face_cascade_profile.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                # Combine the two sets of faces
                if len(faces) != 0 and len(faces2) != 0:
                    faces = np.concatenate((faces, faces2))
                elif len(faces) == 0 and len(faces2) != 0:
                    faces = faces2

            if len(faces) == 0:
                print("No faces found in image: " + image)
                images_kept -= 1
                continue

            if len(faces) > 1:
                images_with_multiple_faces +=1

            for face in faces:

                if FACE == "mtcnn":
                    x, y, w, h = face['box']
                else:
                    x, y, w, h = face

                roi = gray[y:y+h, x:x+w]
                if RECOGNIZER == "eigen" or RECOGNIZER == "fisher":
                    roi = cv2.resize(roi, (48,48))
                x_train.append(roi)
                y_labels.append(id_)

with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")

print("Process complete")
print("Time taken: " + str(time.time() - start_time) + " seconds")
print("Images kept for training: " + str(images_kept))
print("Images with multiple faces: " + str(images_with_multiple_faces))

                

