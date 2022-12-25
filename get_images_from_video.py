import cv2
from pathlib import Path
import math

video_name = "film"

actor_name = "John_Doe"
video_filename = "videos/" + video_name + ".MP4"
result_folder = "results/" + actor_name + "/"
img_names = "frame_"

skip_frame = 15
frame_id = 0

resize = True
resize_size = (1920, 1080)

start_minute = 0.01
end_minute = 0.04

# Create result folder if it does not exists
Path(result_folder).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video_filename)

# Get frame rate based on version of opencv
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
else:
  fps = cap.get(cv2.CAP_PROP_FPS)

start_frame = int((math.floor(start_minute) * 60 + ((start_minute - math.floor(start_minute)) * 100)) * fps)
end_frame = int((math.floor(end_minute) * 60 + ((end_minute - math.floor(end_minute)) * 100)) * fps)

if (cap.isOpened() == False):
  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    if frame_id > start_frame and frame_id < end_frame:
      if frame_id % skip_frame == 0:
        # Display the resulting frame
        if resize:
          frame = cv2.resize(frame, resize_size )
        cv2.imshow('Frame', frame)
        cv2.imwrite(result_folder + "/" + img_names + str(frame_id)+'.jpg', frame)

    frame_id +=1

    if frame_id > end_frame:
      break

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    # Break the loop
  else:
    break

    # When everything is done, release the video capture object
cap.release()

  # Closes all the frames
cv2.destroyAllWindows()
