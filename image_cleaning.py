"""
Created by Daniel-Iosif Trubacs on 30 May 2023 for the MAChellenge. The aim
of this module is to clean the dataset and label each image in the dataset provided
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle

# use this backend to avoid Turning the interactive backend on
matplotlib.use('TkAgg')

# directory containing all the images
path_folder = 'robotics_images'

# a numpy aray that contains all images
images = None

# the labels that each image  can contain
labels = ['red plastic buoy', 'green plastic buoy', 'white plastic buoy', 'duck',
          'milk plastic bottle', 'water plastic bottle', 'red metallic buoy']

# saved labels for each image (saved_labels[i] should be the label for images images[i])
saved_labels = []

# read all the images in the directory
for path in os.listdir(path_folder):
    # load every image and convert to a numpy array
    img = plt.imread(os.path.join(path_folder, path), format='jpeg')

    # save each image in the numpy array
    reshaped_img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
    if images is None:
        images = reshaped_img

    else:
        images = np.concatenate((images, reshaped_img))

    # print(images.shape[0], images.shape, images.nbytes*1e-9)
    # save a label for each image
    running = True
    while running:
        plt.title("No label selected yet")
        plt.imshow(img)
        plt.show()

        x = input("Please insert the label (0-6): ")
        plt.title(labels[int(x)])
        plt.imshow(img)
        plt.show()
        y = input("Press x if the label is correct: ")
        if y == 'x':
            running = False
            saved_labels.append(int(x))

# convert the labels to numpy array
saved_labels = np.array(saved_labels)

# save the images with pickle
with open('data/images', 'wb') as handle:
    pickle.dump(images, handle)


# save the labels with pickle
with open('data/labels', 'wb') as handle:
    pickle.dump(saved_labels, handle)

