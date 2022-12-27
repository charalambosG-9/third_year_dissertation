import simple_image_download.simple_image_download as simp

# Download images from the web
response = simp.Downloader()

response.directory = "web_images/"
response.extensions = ".jpg"

keywords = ["flashpoint batman"]

for kw in keywords:
    response.download(kw.replace(" ", "+"), limit = 10)
