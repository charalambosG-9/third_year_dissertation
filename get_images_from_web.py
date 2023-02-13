import simple_image_download.simple_image_download as simp
import os
import cv2

results_directory = "results/"

# Resize image
resize = False
resize_size = (1920, 1080)

# Download images from the web
response = simp.Downloader()

# Change the directory to save the images and the format
response.directory = results_directory
response.extensions = ".jpg"

# Keywords to search for
keywords = ["Walter White"]

for kw in keywords:
    response.download(kw.replace(" ", "+"), limit = 300)
    subfolder_path = results_directory + kw.replace(" ", "+")
    images = os.listdir(subfolder_path)

    if resize:
        for image in images:
            filename = subfolder_path + "/" + image
            img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            if not img is None:
                img = cv2.resize(img, resize_size)
                cv2.imwrite(filename, img)
