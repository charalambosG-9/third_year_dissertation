import cv2
import os
import shutil
from pathlib import Path

def opencv_to_yolo(img_width,img_height,x,y,w,h):
    
    # Convert from openCV format YOLO form
    width_norm = round(w/img_width, 6)
    height_norm = round(h/img_height, 6)
    x_center_norm = round((x + w/2)/img_width, 6)
    y_center_norm = round((y + h/2)/img_height, 6)
    
    # Convert from YOLO format to openCV form
    # new_center_x = round(x_center_norm * img_width)
    # new_center_y = round(y_center_norm * img_height)
    # new_width = round(width_norm * img_width)
    # new_height = round(height_norm * img_height)
    # new_x = new_center_x - round(new_width/2)
    # new_y = new_center_y - round(new_height/2)
    
    return width_norm, height_norm, x_center_norm, y_center_norm

# Remove suffix from image name
def remove_suffix(image):

    image = image.removesuffix('.jpg')
    image = image.removesuffix('.jpeg')
    image = image.removesuffix('.png')

    return image

# Paths
folder_path = "results/"

images_path = "images/"
labels_path = "labels/"

train_folder = "train/"
val_folder = "val/"

# Read haar cascades for detection
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Get list of subfolders 
folders = os.listdir(folder_path)

# Path(labels_path).mkdir(parents=True, exist_ok=True)

Path(train_folder + "/" + images_path).mkdir(parents=True, exist_ok=True)
Path(train_folder + "/" + labels_path).mkdir(parents=True, exist_ok=True)

Path(val_folder + "/" + images_path).mkdir(parents=True, exist_ok=True)
Path(val_folder + "/" + labels_path).mkdir(parents=True, exist_ok=True)

# classes_file = open(labels_path + "classes.txt", "w")

class_counter = 0

for folder in folders:
    
    subfolder_path = folder_path + folder
    # Get list of images in subfolder
    images = os.listdir(subfolder_path)

    number_of_images = len(images)

    threshold_for_training = int(number_of_images * 0.7)

    threshold_counter = 0 

    # classes_file.writelines((folder + '\n').replace('+',' '))
    
    for image in images:
    
        # make sure file is an image
        if image.endswith(('.jpg', '.png', 'jpeg')):
            img_path = subfolder_path + "/" + image

            threshold_counter += 1

            if threshold_counter <= threshold_for_training:

                # Copy image to training folder
                shutil.copyfile(img_path, train_folder + "/" + images_path + "/" + image)

                # Remove file extension
                image = remove_suffix(image)

                label_file = open(train_folder + "/" + labels_path + "/" + image + ".txt", "w")
            else:
                # Copy image to validation folder
                shutil.copyfile(img_path, val_folder + "/" + images_path + "/" + image)

                # Remove file extension
                image = remove_suffix(image)

                label_file = open(val_folder + "/" + labels_path + "/" + image + ".txt", "w")

            # Read input image
            img = cv2.imread(img_path)

            # Get image dimensions
            img_height, img_width, _ = img.shape

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects faces in the input image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop over all the faces detected
            for(x,y,w,h) in faces: 

                # Convert from openCV format YOLO form
                width_norm, height_norm, x_center_norm, y_center_norm = opencv_to_yolo(img_width,img_height,x,y,w,h)

                # Draw a rectangle in a face to check accuracy of conversion
                cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 255), 2)
                cv2.putText(img, folder.replace('+',' '), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
               
                label_file.writelines(str(class_counter) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(width_norm) + " " + str(height_norm) + "\n")

                # Display the image in a window
                cv2.imshow('Image',img)
                cv2.waitKey(0)
        
        label_file.close()
    
    class_counter += 1

# classes_file.close()
cv2.destroyAllWindows()