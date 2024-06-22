import os

import numpy as np

from PIL import Image
import cv2
from yolov_model import load

from rembg import remove

def remove_bg(image):
    return remove(image, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_structure_size=6, alpha_matting_base_size=1000, alpha_matting_base_color=(255, 255, 255))



def parse_image(image_path, item):
    yolov_model = load()

    result = yolov_model(image_path,verbose =False)
    objects = result[0].boxes.data.cpu().numpy()
    labels = result[0].names

    detected_objects = ()
    for i in range(len(objects)):
        x1, y1, x2, y2, p, it = objects[i]
        label = labels[it]
        if(label == item):
            detected_objects=(label, (x1, y1, x2, y2))

    # Load the image using PIL
    image = Image.open(image_path)

    # Crop the image to the bounding box
    if detected_objects:
        _, (x1, y1, x2, y2) = detected_objects
        cropped_image = image.crop((x1, y1, x2, y2))
        # cropped_image = remove_bg(cropped_image)
        cropped_image.save('cropped.png')

    print(detected_objects)
    return detected_objects

def compare(item:str, image_path:str):
    files = os.listdir('/home/rohan/hackonama/datasets/amazon/rmbg_'+item)

    sift = cv2.SIFT_create()

    img_desc = sift.detectAndCompute(cv2.imread(image_path), None)[1]

    desc = []
    for file in files:
        image = cv2.imread('/home/rohan/hackonama/datasets/amazon/rmbg_'+item+'/'+file)
        desc.append(sift.detectAndCompute(image, None)[1])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = []
    for d in desc:
        raw_matches = bf.match(img_desc, d)
        if len(raw_matches) > 4:  # We need at least 4 matches to compute a homography
            src_pts = np.float32([img_desc[m.queryIdx] for m in raw_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([d[m.trainIdx] for m in raw_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches.append(np.sum(mask))  # The number of inlier matches
        else:
            matches.append(0)
    print(matches)
    matches = np.array(matches)
    order = np.argsort(matches)[::-1]
    print([files[i] for i in order[:5]])
    
    

