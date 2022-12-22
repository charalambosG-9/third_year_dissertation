import cv2
from pathlib import Path

video_name = "DJI_20220202131005_0005_W"

video_filename = "./Videos_EAC/" + video_name + ".MP4"
result_folder = "./results/" + video_name + "/"
skip_frame = 15
frame_id = 0
img_names = "pylon_"
resize = True
resize_size = (1920,1080)

#create folder if not exists
Path(result_folder).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(video_filename)

if (cap.isOpened() == False):
  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    if frame_id % skip_frame == 0:
      # Display the resulting frame
      if resize:
        frame = cv2.resize(frame, resize_size )
      cv2.imshow('Frame', frame)
      cv2.imwrite(result_folder+"/"+img_names+str(frame_id)+'.jpg',frame)

    frame_id +=1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

    # Break the loop
  else:
    break

    # When everything done, release the video capture object
cap.release()

  # Closes all the frames
cv2.destroyAllWindows()
