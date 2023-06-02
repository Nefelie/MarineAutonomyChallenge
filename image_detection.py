"""
Created by Daniel-Iosif Trubacs on 30 May 2023 for MAChellenge. The aim
of this module is to train an CNN for object detection. The trained CNN
should be used to assist with the object detection task.
"""
import keras
import numpy as np
import pickle
from keras.layers import Dropout, Conv2D, Dense, MaxPool2D, InputLayer, BatchNormalization, Flatten
from matplotlib import pyplot as plt
import cv2 as cv

def one_hot(a: np.ndarray, n_classes: float) -> np.ndarray:
    """ One-hot encoding for labels.

    Args:
        a: numpy array (n_labels, )
        n_classes: number of classes contained in the labels array

    Returns:
        one_hot_a: numpy array, (n_labels, n_classes)
    """

    one_hot_a = np.zeros((a.shape[0], n_classes))
    for i in range(a.shape[0]):
        one_hot_a[i][a[i]] = 1
    # return the one hot encoded labels
    return one_hot_a

# the classes that each image  can contain
classes = ['red plastic buoy', 'green plastic buoy', 'white plastic buoy', 'duck',
          'milk plastic bottle', 'water plastic bottle', 'red metallic buoy']

# load the processed data (the images should already be shuffled
# labeled correspondingly)
# load the images and labels with pickle (divide by 255 to get all pixels in the 0-1 range)
with open('processed_data/images', 'rb') as handle:
    images = pickle.load(handle)
# load the labels with pickle and one_hot encode them (there are 7 classes)
with open('processed_data/labels', 'rb') as handle:
    labels = one_hot(pickle.load(handle), 7)

# split the data into train and test
train_images = images[:50]
train_labels = labels[:50]
test_images = images[50:]
test_labels = labels[50:]

print("The data has been loaded.")
print('The shape of the train data is: ', train_images.shape)
print('The shape of the test data is: ', test_images.shape)


# build a normal CNN
def build_model(img_shape: np.ndarray) -> keras.Model:
    """ Build a normal CNN for object detection.

    Args:
        img_shape: shape of the images in the training data (width, height, n_channels)

    Returns:
        Keras Models (see tf.keras.models for more documentation)
    """
    # input layer
    inputs = keras.Input(img_shape)

    # the main CNN architecture:
    x = Conv2D(filters=64, kernel_size=7, strides=1, activation='relu')(inputs)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=128, kernel_size=5, strides=1, activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=312, kernel_size=3, strides=1, activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # flatten the data before feed into into FCNet
    x = Flatten()(x)

    # main architecture for FCnet:
    x = Dense(units=256, activation='relu')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=128, activation='relu')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.1)(x)

    x = Dense(units=64, activation='relu')(x)
    x = Dropout(rate=0.1)(x)

    # output of the model (n_classes)
    outputs = Dense(units=train_labels.shape[1], activation='softmax')(x)

    # define the model
    model = keras.Model(inputs, outputs, name='Marine_CNN')

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

    # return the model
    return model


# create the model
marine_cnn = build_model(img_shape=train_images[0].shape)
marine_cnn.summary()

# train the model
marine_cnn.fit(x=train_images, y=train_labels, batch_size=4, epochs=2)

# test the model
predictions = marine_cnn.predict(test_images)
# Number of correct predictions
N_acc = 0
# number of all test samples
N_samples = 0
for i in range(len(predictions)):
    N_samples += 1
    # the largest value in the array correspond to the class predicted
    prediction = np.where(predictions[i] == max(predictions[i]))[0][0]
    true = np.where(test_labels[i] == max(test_labels[i]))[0][0]
    if prediction == true:
        N_acc += 1
    else:
        plt.title("Prediction:"+classes[prediction]+" Reality: "+classes[true])
        plt.imshow(test_images[i])
        plt.show()

print("The accuracy is:", N_acc/N_samples)