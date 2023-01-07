import simple_image_download.simple_image_download as simp

# Download images from the web
response = simp.Downloader()

# Change the directory to save the images and the format
response.directory = "results/"
response.extensions = ".jpg"

# Keywords to search for
keywords = ["Will Smith", "Tom Cruise"]

for kw in keywords:
    response.download(kw.replace(" ", "+"), limit = 10)
