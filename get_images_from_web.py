from simple_image_download import simple_image_download as simp
import os

# Download images from the web
response = simp.simple_image_download()

keywords = ["lucifer actor"]

for kw in keywords:
    response.download(kw, 10)

# Rename the folder
os.rename("simple_images", "images")
