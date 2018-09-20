from __future__ import print_function

import sys

from keras import losses

sys.setrecursionlimit(10000)

import densenet_reg
import numpy as np
import sklearn.metrics as metrics

from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
import os
import time
import matplotlib.pyplot as plt

import sklearn.metrics as metrics

from keras.preprocessing.image import img_to_array
import numpy as np
import random
import cv2

batch_size = 32
nb_classes = 30
nb_epoch = 15

img_rows, img_cols = 64, 64
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 12
nb_filter = 12
bottleneck = True
reduction = 0.0
dropout_rate = 0.0  # 0.0 for data augmentation

train_file_path = r'../DatasetA_train_20180813/train/'
com_path = r'../data/com_train.txt'


def load_data(dir, path):
    print("[INFO] loading images...")
    data = []
    labels = []

    filelist = []
    # grab the image paths and randomly shuffle them
    with open(path, 'r') as f:
        for line in f:
            # print(line)
            filelist.append(line.split('\t'))
    # loop over the input images
    count = 0
    for file in filelist:
        # load the image, pre-process it, and store it in the data list
        if count<1000:

            imagePath = dir + file[0]
            # print(imagePath)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (img_rows, img_cols))
            image = img_to_array(image)
            # extract the class label from the image path and update the

            label = np.array(file[1:31])
            labels.append(label)

            data.append(image)
        count += 1
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    labels = np.array(labels)

    return data, labels


def test():
    model = densenet_reg.DenseNetImageNet264(input_shape=img_dim, classes=nb_classes)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training
    model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    testX, testY = load_data(train_file_path, com_path)

    print(testX.shape)
    print(testY.shape)

    testX = testX.astype('float32')

    testY = testY.astype('float32')

    weights_file = r'Zero_DenseNet_Reg20180919.h5'
    if os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

        yPred = model.predict(testX)
        yTrue = testY

        print(yPred.shape)
        print(yTrue.shape)

        loss = 0

        for i in range(len(yPred)):
            for j in range(len(yPred[i])):
                loss += np.abs(yPred[i][j]-yTrue[i][j])

        print("test loss:\t" + str(loss/len(yPred)/30))
    else:
        print("No model")


if __name__ == '__main__':
    test()
