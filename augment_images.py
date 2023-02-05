import random
import os
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import time
import shutil

import albumentations as A
import sys

def drawProgressBar(percent, barLen = 50):
    # percent float from 0 to 1. 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}% ".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

def check_bbox(bbox):
    # Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums
    bbox = list(bbox)
    for i in range(4):
      if (bbox[i] < 0) :
        bbox[i] = 0
      elif (bbox[i] > 1) :
        bbox[i] = 1
    bbox = tuple(bbox)
    return bbox

# Input directory
images_path = "labels/" 

images = [] # to store paths of images from folder
augmented_path = "augmented_images/"
#os.makedirs(augmented_path)
print("Augmenting:", augmented_path, images_path)
# Append  images to list
for im in os.listdir(images_path):  
    if im.endswith('.jpg') or im.endswith('.JPG') or im.endswith('.png') or im.endswith('.PNG') or im.endswith('.JPEG') or im.endswith('.jpeg'):
        images.append(os.path.join(images_path, im))
images_to_generate = len(images)  #you can change this value according to your requirement
print("Images loaded:", len(images))
print("Images to generate:", images_to_generate)
print("Augmentating Images:")
# with open(augmented_path+"classes.txt", "w") as f:
#     f.write("tbar")
# Copy classes.txt file to augmented folder
shutil.copyfile(images_path + "/classes.txt", augmented_path + "/classes.txt")
i = 1
while i <= images_to_generate:
    
    drawProgressBar(i / images_to_generate)
    image_name = random.choice(images)
    original_img = cv2.imread(image_name)
    image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    images.remove(image_name)
    
    # p is the probability of applying the augmentation in the image
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit = 0.08, scale_limit = 1.4, rotate_limit = 45, interpolation = 1, border_mode = 0, always_apply = True),
        A.OneOf([
            A.MultiplicativeNoise(p = 0.5),
            A.GaussNoise(p = 0.5),
        ], p = 0.2), # Performs one of these two with a probability of 20%
        A.OneOf([
            A.MotionBlur(p = 0.3),
            A.MedianBlur(blur_limit = 3, p = 0.2),
            A.Blur(blur_limit = 3, p = 0.2),
        ], p = 0.2),
        A.OneOf([
            A.Sharpen(),
            A.Emboss(),
        ], p = 0.5),
        A.CLAHE(clip_limit = 2),
        A.RandomContrast(limit = 0.2),
        A.RandomRain(p = 0.1),
        A.RandomSunFlare(p=0.1)
        ],
        bbox_params = A.BboxParams(format = 'yolo', label_fields = ['category_ids']))
    # random.seed(42)
    filename = image_name.split('.')[-2] + ".txt"
    bboxes = []
    category_ids = []
    # get the actual ground truth
    with open(filename) as f:
        print(filename)
        lines = f.readlines()
        for line in lines:
            category_ids.append(int(line.split(' ')[0]))
            bboxes_str = line.strip('\n').split(' ')[1:]
            bboxes.append ([float(i) for i in bboxes_str])
            # print(out[1:])
    # apply the transformations for images, boxes and categories
    
    for i_bbox in range(len(bboxes)) : 
        bboxes[i_bbox] = check_bbox(bboxes[i_bbox])
    # print("Bbox checked: ", bboxes)
    transformed = transform(image = image, bboxes = bboxes, category_ids = category_ids)
    augmented_image = transformed['image']
    boxes = transformed['bboxes']
    categories = transformed['category_ids']
    cv2.imwrite(augmented_path + '/' + Path(image_name).stem + '.jpg',cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
    # save the changed ground truth file
    augm_file = open(augmented_path + '/' + Path(image_name).stem + '.txt', 'w')
    for n in range(0,len(boxes)):
        augm_file.write(str(categories[n]) + ' ' + str(float(format(boxes[n][0], '.5f'))) + ' ' + str(float(format(boxes[n][1], '.5f'))) + ' ' + str(float(format(boxes[n][2], '.5f'))) + ' ' + str(float(format(boxes[n][3], '.5f'))) + '\n')
    augm_file.close()
    # display the images
    # cv2.imshow('original', cv2.resize(original_img, (1280, 720)))
    # cv2.imshow('augm', cv2.resize(cv2.cvtColor(augmented_image,cv2.COLOR_RGB2BGR), (1280, 720)))
    # cv2.waitKey(0)
    i += 1

print("\nAugmentation Finished")