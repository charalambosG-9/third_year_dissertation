# Imports
import cv2
from pathlib import Path
import math
import argparse
import ast

# Check validity of time frame
def check_valid(start_frame, end_fame, total_frames):
  
  if start_frame > end_frame:
        print("Start frame is greater than end frame. Moving to next set.")
        return False
  
  if start_frame > total_frames:
        print("Start frame is greater than total frames. Moving to next set.")
        return False
  
  return True

parser = argparse.ArgumentParser()
parser.add_argument('--video', dest='video', type=str, help='Add video name without extension', default='')
parser.add_argument('--result_folder', dest='result_folder', type=str, help='Add directory to save images', default='allFrames')
parser.add_argument('--skip_frame', dest='skip_frame', type=int, help='Add frame interval', default=15)
parser.add_argument('--resize', dest='resize', type=bool, help='Resize images', default=True)
parser.add_argument('--minutes', dest='minutes', type=str, help='Add time frames', default='')
args = parser.parse_args()
  
video_name = args.video

video_filename = video_name + ".mp4"
result_folder = args.result_folder + "/" + video_name.replace(" ", "+") + "/"
img_names = "_frame_"

skip_frame = args.skip_frame
frame_id = 0

resize = args.resize
resize_size = (1920, 1080)

# List of tuples with start and end minute
minutes = ast.literal_eval(args.minutes)

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

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(len(minutes)):

    minute_tuple = minutes[i]
    start_frame = int((math.floor(minute_tuple[0]) * 60 + ((minute_tuple[0] - math.floor(minute_tuple[0])) * 100)) * fps)
    end_frame = int((math.floor(minute_tuple[1]) * 60 + ((minute_tuple[1] - math.floor(minute_tuple[1])) * 100)) * fps)

    if not check_valid(start_frame, end_frame, total_frames):
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_id = start_frame

    while(cap.isOpened() and frame_id < end_frame):
        ret, frame = cap.read()

        if ret == True:
            if frame_id % skip_frame == 0:
                # Display the resulting frame
                if resize:
                    frame = cv2.resize(frame, resize_size)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
                cv2.imwrite(result_folder + "/" + video_name + img_names + str(frame_id)+'.jpg', frame)

        frame_id += 1

# When everything is done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()