"""
Created by Daniel-Iosif Trubacs on 2 June 2023 for the MAChellenge. The aim
of this module is to process the images for feeding them into a CNN. The images are
have to already be cleaned with the image_cleaning module
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import cv2 as cv

# load the cleaned images and labels with pickle
with open('data/images', 'rb') as handle:
    images = pickle.load(handle)

# resize the images to size (200x200) to reduce the data usage and divide by 255
# to have all the pixels in the range 0-1
width = 200
height = 200
resized_images = np.zeros(shape=(images.shape[0], width, height, images.shape[-1]))
for i in range(images.shape[0]):
    resized_images[i] = cv.resize(images[i], (width, height), interpolation=cv.INTER_AREA)/255

# save the processed images with pickle
with open('processed_data/images', 'wb') as handle:
    pickle.dump(resized_images, handle)
    print("The data has been saved.")
