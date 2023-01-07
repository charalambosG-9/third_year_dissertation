import cv2
import os

# Resize image
# resize = True
# resize_size = (1920, 1080)

# Paths
folder_path = "results/"
labels_path = "labels/"

# Read haar cascades for detection
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

# Get list of subfolders 
folders = os.listdir(folder_path)

classes_file = open(labels_path + "classes.txt", "w")

counter = 0

for folder in folders:
    
    subfolder_path = folder_path + folder
    # Get list of images in subfolder
    images = os.listdir(subfolder_path)

    classes_file.writelines((folder + '\n').replace('+',' '))
    
    for image in images:
    
        # make sure file is an image
        if image.endswith(('.jpg', '.png', 'jpeg')):
            img_path = subfolder_path + "/" + image

            # Remove file extension (default is .jpg)
            new_image = image.removesuffix('.jpg')
            new_image = new_image.removesuffix('.jpeg')
            new_image = new_image.removesuffix('.png')

            label_file = open(labels_path + new_image + ".txt", "w")

            # Read input image
            img = cv2.imread(img_path)

            img_height, img_width, _ = img.shape

            # if resize:
            #     img = cv2.resize(img, resize_size)

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detects faces in the input image
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop over all the faces detected
            for(x,y,w,h) in faces: 

                # YOLO FORMAT Label_ID_1 X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM

                # X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH
                # Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT
                # WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH
                # HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT

                # Convert from openCV format YOLO format

                width_norm = round(w/img_width, 6)
                height_norm = round(h/img_height, 6)
                x_center_norm = round((x + w/2)/img_width, 6)
                y_center_norm = round((y + h/2)/img_height, 6)

                # Convert from YOLO format to openCV format

                new_center_x = round(x_center_norm * img_width)
                new_center_y = round(y_center_norm * img_height)
                new_width = round(width_norm * img_width)
                new_height = round(height_norm * img_height)
                new_x = new_center_x - round(new_width/2)
                new_y = new_center_y - round(new_height/2)

                # Draw a rectangle in a face
                # cv2.rectangle(img,(x, y),(x + w, y + h),(0, 255, 255), 2)
                cv2.rectangle(img,(new_x, new_y),(new_x + new_width, new_y + new_height),(0, 255, 255), 2)
                cv2.putText(img, "Face", (new_x, new_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                label_file.writelines(str(counter) + " " + str(x_center_norm) + " " + str(y_center_norm) + " " + str(width_norm) + " " + str(height_norm) + "\n")

                # Display the image in a window
                cv2.imshow('Image',img)
                cv2.waitKey(0)

        label_file.close()
    counter += 1

classes_file.close()
cv2.destroyAllWindows()