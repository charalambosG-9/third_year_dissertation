# Imports
import cv2
import os
import shutil
from pathlib import Path
from augment_images import augment_images
import pickle   
import mtcnn
import numpy as np
import time
import argparse

# Convert from openCV format YOLO form
def opencv_to_yolo(img_width, img_height, x, y, w, h):
    
    width_norm = round(w/img_width, 6)
    height_norm = round(h/img_height, 6)
    x_center_norm = round((x + w/2)/img_width, 6)
    y_center_norm = round((y + h/2)/img_height, 6)
    
    return width_norm, height_norm, x_center_norm, y_center_norm

# Remove suffix from image name
def remove_suffix(image):

    suffixes = ('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG', 'webp', 'WEBP')

    for suffix in suffixes:
        image = image.removesuffix(suffix)

    return image

# Create file in new folders
def create_file_in_new_folders(folder, img_path,image):
    
    # Copy image to appropriate folder
    shutil.copyfile(img_path, os.path.join(folder + images_path, image))
    img_path = os.path.join(folder + images_path, image)
    # Remove file extension
    image = remove_suffix(image)
    # Create label file
    label_file = open(os.path.join(folder + labels_path, image) + ".txt", "w")
    return img_path, label_file

parser = argparse.ArgumentParser()
parser.add_argument('--detector', dest='detector', type=str, help='Haar or mtcnn', default='haar')
parser.add_argument('--recognizer', dest='recognizer', type=str, help='lbph or fisher or eigen', default='eigen')
parser.add_argument('--folder', dest='folder', type=str, help='Add name of input folder', default='allFrames')
parser.add_argument('--conf', dest='conf', type=float, help='Confidence threshold', default=80)
args = parser.parse_args()


FACE = (args.detector).lower() # "haar" or "mtcnn"
RECOGNIZER = (args.recognizer).lower() # "lbph" or "eigen" or "fisher"

conf_threshold = args.conf # May vary based on which face detection algorithm is used

# Paths
folder_path = args.folder

images_path = "images/"
labels_path = "labels/"

train_folder = "train/"
val_folder = "val/"
test_folder = "test/"

train_folder_images = os.path.join(train_folder, images_path)
train_folder_labels = os.path.join(train_folder, labels_path)
val_folder_images = os.path.join(val_folder, images_path)
val_folder_labels = os.path.join(val_folder, labels_path)
test_folder_images = os.path.join(test_folder, images_path)
test_folder_labels = os.path.join(test_folder, labels_path)

Path(train_folder_images).mkdir(parents=True, exist_ok=True)
Path(train_folder_labels).mkdir(parents=True, exist_ok=True)

Path(val_folder_images).mkdir(parents=True, exist_ok=True)
Path(val_folder_labels).mkdir(parents=True, exist_ok=True)

Path(test_folder_images).mkdir(parents=True, exist_ok=True)
Path(test_folder_labels).mkdir(parents=True, exist_ok=True)

images_kept = 0
start_time = time.time()

# Labels

if RECOGNIZER == "lbph":
    recognizer = cv2.face.LBPHFaceRecognizer_create()
elif RECOGNIZER == "eigen":
    recognizer = cv2.face.EigenFaceRecognizer_create()
elif RECOGNIZER == "fisher":
    recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer.read("recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

if FACE == "mtcnn":
    detector = mtcnn.MTCNN()
else:
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
    face_cascade2 = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')

# Get list of subfolders 
folders = os.listdir(folder_path)

for folder in folders:
    
    subfolder_path = os.path.join(folder_path, folder)
    # Get list of images in subfolder
    images = os.listdir(subfolder_path)

    number_of_images = len(images)

    threshold_for_training = int(number_of_images * 0.7)
    threshold_for_validation = int(number_of_images * 0.9)

    threshold_counter = 0 
    
    for image in images:
    
        # make sure file is an image
        if image.endswith(('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG', 'WEBP', 'webp')):
            img_path = os.path.join(subfolder_path, image) 

            threshold_counter += 1

            print("Image {} out of {} in folder {}".format(threshold_counter, number_of_images, folder))

            if threshold_counter > threshold_for_validation:
                img_path, label_file = create_file_in_new_folders(test_folder, img_path, image)
            elif threshold_counter > threshold_for_training:
                img_path, label_file = create_file_in_new_folders(val_folder, img_path, image)
            else:
                img_path, label_file = create_file_in_new_folders(train_folder, img_path, image)

            # Read input image
            img = cv2.imread(img_path)

            # Get image dimensions
            img_height, img_width, _ = img.shape

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if FACE == "mtcnn":
                # Covert image to BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                faces = detector.detect_faces(img_bgr)
            else:
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
                faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

                if len(faces) != 0 and len(faces2) != 0:
                    faces = np.concatenate((faces, faces2))
                elif len(faces) == 0 and len(faces2) != 0:
                    faces = faces2

            # If no faces are detected, skip the image, and delete image and label file
            if len(faces) == 0:
                print("No faces detected")
                label_file.close()
                os.remove(img_path)
                os.remove(label_file.name)
                continue

            kept = False
            
            # Loop over all the faces detected
            for face in faces: 

                if FACE == "mtcnn":
                    x, y, w, h = face['box']
                else:
                    x, y, w, h = face

                # Create a copy of the image to draw on
                temp_img = img.copy()

                roi_gray = gray[y:y+h, x:x+w] 

                if RECOGNIZER == "eigen" or RECOGNIZER == "fisher":
                        roi_gray = cv2.resize(roi_gray, (48,48))
                
                id_, conf = recognizer.predict(roi_gray)

                print("Confidence: " + str(conf))

                if conf <= conf_threshold: 
                    print(labels[id_])
                    # Convert from openCV format YOLO form
                    width_norm, height_norm, x_center_norm, y_center_norm = opencv_to_yolo(img_width, img_height, x, y, w, h)
                    label_file.writelines(str(id_) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(width_norm) + " " + str(height_norm) + "\n")
        
                    # Increment number of images kept
                    if not kept:
                        images_kept += 1
                        kept = True

                else:
                    print("Face rejected")

        label_file.close()

        # If no faces are kept, delete image and label file
        if os.stat(label_file.name).st_size == 0:
            os.remove(img_path)
            os.remove(label_file.name)


augment_images(images_path = train_folder_images, labels_path = train_folder_labels, images_to_generate = len(os.listdir(train_folder_images)))
augment_images(images_path = val_folder_images, labels_path = val_folder_labels, images_to_generate = len(os.listdir(val_folder_images)))
augment_images(images_path = test_folder_images, labels_path = test_folder_labels, images_to_generate = len(os.listdir(test_folder_images)))

print("Number of images kept: " + str(images_kept))
print("Time taken: " + str(time.time() - start_time) + " seconds")

if os.path.isdir("data/train/") and os.path.isdir("data/val/") and os.path.isdir("data/test/"):
    
    folders = ["train/", "val/", "test/"]
    for folder in folders:
        label_file = "data/" + folder + "labels.cache"
        if os.path.isfile(label_file):
            os.remove(label_file)
        subfolders = os.listdir(folder)
        for subfolder in subfolders:
            new_path = os.path.join(folder, subfolder)
            files = os.listdir(new_path)
            for file in files:
                rel_path = os.path.join(new_path, file)
                shutil.move(rel_path, os.path.join("data/", rel_path))
        shutil.rmtree(folder)
else:

    shutil.move("train/", "data/")
    shutil.move("val/", "data/")
    shutil.move("test/", "data/")
