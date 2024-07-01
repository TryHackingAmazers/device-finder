import os

import numpy as np

from PIL import Image
import cv2
from recommendation.yolov_model import load

from rembg import remove

def remove_bg(image):
    return remove(image, alpha_matting=True, alpha_matting_foreground_threshold=240, alpha_matting_background_threshold=10, alpha_matting_erode_structure_size=6, alpha_matting_base_size=1000, alpha_matting_base_color=(255, 255, 255))



def parse_image(image_path, id):
    yolov_model = load()

    result = yolov_model(image_path,verbose =False)
    objects = result[0].boxes.data.cpu().numpy()
    labels = result[0].names

    x1, y1, x2, y2, p, it = objects[id]
    label = labels[it]
    detected_objects=(label, (x1, y1, x2, y2))

    # Load the image using PIL
    image = Image.open(image_path)

    _, (x1, y1, x2, y2) = detected_objects
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image,label

def compare(image_path:str,id:int):
    cropped_image, item = parse_image(image_path,id)
    files = os.listdir('./datasets/amazon/'+item.lower()+"_rmbg")

    sift = cv2.SIFT_create()

    cropped_image = np.array(cropped_image)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)

    img_desc = sift.detectAndCompute(cropped_image, None)[1]

    desc = []
    for file in files:
        image = cv2.imread('./datasets/amazon/'+item.lower()+"_rmbg"+'/'+file)
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
    matches = np.array(matches)
    order = np.argsort(matches)[::-1]
    return (['amazon/'+item.lower()+'/'+files[i][:-3]+"jpg" for i in order[:]])