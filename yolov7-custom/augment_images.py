import random
import os
import cv2
from pathlib import Path
import albumentations as A
import sys

# Draw progress bar
def drawProgressBar(percent, barLen = 50): 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}% ".format("=" * int(barLen * percent), barLen, percent * 100))
    sys.stdout.flush()

# Check if bbox boundaries are in range 0, 1 and minimums are lesser then maximums
def check_bbox(bbox):
    bbox = list(bbox)
    for i in range(4):
      if (bbox[i] < 0):
        bbox[i] = 0
      elif (bbox[i] > 1):
        bbox[i] = 1
    bbox = tuple(bbox)
    return bbox

def augment_images(images_path, labels_path, images_to_generate):

    # Store images in list
    images = [] 

    # Append  images to list
    for image in os.listdir(images_path):  
        if image.endswith(('.jpg', '.png', 'jpeg', '.JPG', '.PNG', 'JPEG', 'WEBP', 'webp')):
            images.append(image)

    print("Images loaded:", len(images))
    print("Images to generate:", images_to_generate)
    print("Augmentating Images:")

    i = 1
    while i <= images_to_generate:

        drawProgressBar(i / images_to_generate)

        try:
            temp_image_name = random.choice(images)
        except:
            break

        image_name = os.path.join(images_path, temp_image_name)
        original_img = cv2.imread(image_name)
        image = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        images.remove(temp_image_name)

        # p is the probability of applying the augmentation in the image
        transform = A.Compose([
            A.ShiftScaleRotate(shift_limit = 0.08, scale_limit = 1.4, rotate_limit = 45, interpolation = 1, border_mode = 0, p = 0.4),
            A.OneOf([
                A.MultiplicativeNoise(p = 0.5),
                A.GaussNoise(p = 0.5),
            ], p = 0.2),
            A.OneOf([
                A.MotionBlur(p = 0.4),
                A.MedianBlur(blur_limit = 3, p = 0.2),
                A.Blur(blur_limit = 3, p = 0.2),
            ], p = 0.3),
            A.OneOf([
                A.Sharpen(),
                A.Emboss(),
            ], p = 0.5),
            A.CLAHE(clip_limit = 2),
            A.RandomContrast(limit = 0.2),
            A.OneOf([
                A.RandomRain(p = 0.2),
                A.RandomSunFlare(p=0.2),
                A.RandomFog(p=0.2),
                A.RandomSnow(p=0.2),
                A.RandomShadow(p=0.2),
                A.RandomBrightness(p=0.2),
            ], p = 0.5),
            ],
            bbox_params = A.BboxParams(format = 'yolo', label_fields = ['category_ids']))

        # random.seed(42)
        filename = os.path.join(labels_path, temp_image_name.split('.')[-2])  + ".txt"
        bboxes = []
        category_ids = []

        # get the actual ground truth
        print(filename)
        with open(filename) as f:
            lines = f.readlines()
            for line in lines:
                category_ids.append(int(line.split(' ')[0]))
                bboxes_str = line.strip('\n').split(' ')[1:]
                bboxes.append ([float(i) for i in bboxes_str])

        # apply the transformations for images, boxes and categories

        for i_bbox in range(len(bboxes)) : 
            bboxes[i_bbox] = check_bbox(bboxes[i_bbox])

        try:
            transformed = transform(image = image, bboxes = bboxes, category_ids = category_ids)
        except:
            continue

        augmented_image = transformed['image']
        boxes = transformed['bboxes']
        categories = transformed['category_ids']

        cv2.imwrite(images_path + '/' + Path(image_name).stem + '_augm_' + str(i) +'.jpg', cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
        # save the changed ground truth file
        augm_file = open(labels_path + '/' + Path(image_name).stem + '_augm_' + str(i) + '.txt', 'w')
        for n in range(0, len(boxes)):
            augm_file.write(str(categories[n]) + ' ' + str(float(format(boxes[n][0], '.5f'))) + ' ' + str(float(format(boxes[n][1], '.5f'))) + ' ' + str(float(format(boxes[n][2], '.5f'))) + ' ' + str(float(format(boxes[n][3], '.5f'))) + '\n')
        augm_file.close()

        i += 1

    print("\nAugmentation Finished")