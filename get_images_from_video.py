import cv2
from pathlib import Path
import math

video_name = "trim"

actor_name = "Bryan Cranston"
video_filename = video_name + ".mp4"
result_folder = "results/" + actor_name.replace(" ", "+") + "/"
img_names = "frame_"

skip_frame = 15
frame_id = 0

resize = True
resize_size = (1920, 1080)

# List of tuples with start and end minute
minutes = [(0.20, 3.14)]
current_set = 0

ping_counter = 0

# Create result folder if it does not exists
Path(result_folder).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video_filename)

# Get frame rate based on version of opencv
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if int(major_ver) < 3:
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
else:
  fps = cap.get(cv2.CAP_PROP_FPS)

if (cap.isOpened() == False):
  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()

  # Check for end of desired minutes
  if current_set > len(minutes) - 1:
    break

  minute_tuple = minutes[current_set]

  start_frame = int((math.floor(minute_tuple[0]) * 60 + ((minute_tuple[0] - math.floor(minute_tuple[0])) * 100)) * fps)
  end_frame = int((math.floor(minute_tuple[1]) * 60 + ((minute_tuple[1] - math.floor(minute_tuple[1])) * 100)) * fps)

  if ping_counter == 30:
    print("--------------------")
    print("Start frame: " + str(start_frame))
    print("End frame: " + str(end_frame))
    print("Current frame: " + str(frame_id))
    print("--------------------")
    ping_counter = 0
  else:
    ping_counter += 1

  if start_frame > end_frame:
    print("Start frame is greater than end frame. Moving to next set.")
    current_set += 1
    continue

  if ret == True:
    if frame_id > start_frame and frame_id < end_frame:
      if frame_id % skip_frame == 0:
        # Display the resulting frame
        if resize:
          frame = cv2.resize(frame, resize_size)
        cv2.imshow('Frame', frame)
        cv2.imwrite(result_folder + "/" + img_names + str(frame_id)+'.jpg', frame)

    frame_id +=1

    if frame_id > end_frame:
      current_set += 1

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
