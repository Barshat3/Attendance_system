import os
import cv2
import numpy as np

# Set dataset path and image dimensions
DATASET_PATH = '//home/barshat/Desktop/Attendence System/dataset/lfw-deepfunneled' 
IMG_SIZE = (160, 160)

def load_images(dataset_path):
    data = {}
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):  # Ensure it's a directory
            images = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)  # Resize image
                    images.append(img)
            if len(images) >= 2:  # Only include people with at least two images
                data[person_name] = images
    return data

# Load dataset
data = load_images(DATASET_PATH)