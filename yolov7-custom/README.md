## Install dependencies

pip install -r requirements.txt
pip install -r requirements_gpu.txt

## Get images from video

python get_images_from_video.py --video <video_name> --result_folder <result_folder> --skip_frame 15 --resize True --minutes <time_frames i.e. [(0.05,0.12),(0.11,0.20)]>

## Train face classification model

python face_classification_trainer.py --recognizer <recognizer> --detector <detector> --folder <input_folder>

recognizer is eigen, fisher or lbph. Detector is haar or mtcnn

## Create training dataset

python face_detector.py --detector <detector> --recognizer <recognizer>  --folder <input_folder> --conf <confidence>

## Train face detection model

python train.py --workers 16 --device 0 --batch-size 32 --epochs 300  --img <image-size> <image-size> --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml  --cfg cfg/training/yolov7-custom.yaml --name yolov7  --weights weights/yolov7_training.pt

Note: This command uses parameters suitable for a HPC. For a less powerful machine, the number of workers, epochs and batch-size should be reduced.

## Detect faces in video

python detect.py --weights <weight> --conf 0.1 --img-size <img-size> --source <source> --view-img --no-trace

The weight file can be found in the runs/detect folder. The weights with the highest accuracy are named best.pt. The source is the image or video file. The img-size is the size of the image and depends on the model used.

## Test face detection model

python test.py --data data/custom_data.yaml --img <img-size> --batch 16 --conf 0.001 --iou 0.65 --device 0 --weights <weights> --name yolov7_test


