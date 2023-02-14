import cv2
import os
import shutil
from pathlib import Path
import numpy as np
from augment_images import augment_images

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

    suffixes = ('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG')

    for suffix in suffixes:
        image = image.removesuffix(suffix)

    return image

def draw_on_image(temp_img, x, y, w, h, folder):

    cv2.rectangle(temp_img, (x, y),(x + w, y + h),(0, 255, 255), 2)
    cv2.putText(temp_img, folder.replace('+',' '), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(temp_img, "Q - Discard Box", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
    cv2.putText(temp_img, "P - Keep Box", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

# Paths
folder_path = "results/"

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

# Read haar cascades for detection
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
face_cascade2 = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

# Get list of subfolders 
folders = os.listdir(folder_path)

class_counter = 0
total_images = 0

for folder in folders:
    
    subfolder_path = folder_path + folder
    # Get list of images in subfolder
    images = os.listdir(subfolder_path)

    number_of_images = len(images)

    total_images += number_of_images

    threshold_for_training = int(number_of_images * 0.8)
    threshold_for_validation = int(number_of_images * 0.9)

    threshold_counter = 0 
    
    for image in images:
    
        # make sure file is an image
        if image.endswith(('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG')):
            img_path = subfolder_path + "/" + image

            threshold_counter += 1

            if threshold_counter > threshold_for_validation:

                # Copy image to validation folder
                shutil.copyfile(img_path, test_folder + "/" + images_path + "/" + image)
                img_path = test_folder + "/" + images_path + "/" + image
                # Remove file extension
                image = remove_suffix(image)
                label_file = open(test_folder + "/" + labels_path + "/" + image + ".txt", "w")

            elif threshold_counter > threshold_for_training:
                
                # Copy image to validation folder
                shutil.copyfile(img_path, val_folder + "/" + images_path + "/" + image)
                img_path = val_folder + "/" + images_path + "/" + image
                # Remove file extension
                image = remove_suffix(image)
                label_file = open(val_folder + "/" + labels_path + "/" + image + ".txt", "w")
            
            else:

                # Copy image to training folder
                shutil.copyfile(img_path, train_folder + "/" + images_path + "/" + image)
                img_path = train_folder + "/" + images_path + "/" + image
                # Remove file extension
                image = remove_suffix(image)
                label_file = open(train_folder + "/" + labels_path + "/" + image + ".txt", "w")

            # Read input image
            img = cv2.imread(img_path)

            # Get image dimensions
            img_height, img_width, _ = img.shape

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects faces in the input image
            faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors = 7)
            faces2 = face_cascade2.detectMultiScale(gray, scaleFactor = 1.05, minNeighbors =  7)

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

            # Loop over all the faces detected
            for(x,y,w,h) in faces: 

                # Create a copy of the image to draw on
                temp_img = img.copy()

                # Convert from openCV format YOLO form
                width_norm, height_norm, x_center_norm, y_center_norm = opencv_to_yolo(img_width, img_height, x, y, w, h)

                # Draw a rectangle in a face to check accuracy of conversion
                draw_on_image(temp_img, x, y, w, h, folder)

                # Display the image in a window
                cv2.imshow(image, temp_img)
    
                if cv2.waitKey(0) == ord('q'):
                    print("Discard")
                    cv2.destroyAllWindows()
                    continue
                else:
                    print("Keep")
                    label_file.writelines(str(class_counter) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(width_norm) + " " + str(height_norm) + "\n")
                    cv2.destroyAllWindows()


                # Possible solution to the above problem
                # k = cv2.waitKey(0)
                # if k==27:    # Esc key to stop
                #     break
                # elif k==-1:  # normally -1 returned,so don't print it
                #     continue
                # else:
                #     print k # else print its value

        label_file.close()

        # If no faces are kept, delete image and label file
        if os.stat(label_file.name).st_size == 0:
            os.remove(img_path)
            os.remove(label_file.name)

    class_counter += 1

cv2.destroyAllWindows()

augment_images(images_path = train_folder_images, labels_path = train_folder_labels, images_to_generate = len(os.listdir(train_folder_images)))
augment_images(images_path = val_folder_images, labels_path = val_folder_labels, images_to_generate = len(os.listdir(val_folder_images)))
augment_images(images_path = test_folder_images, labels_path = test_folder_labels, images_to_generate = len(os.listdir(test_folder_images)))