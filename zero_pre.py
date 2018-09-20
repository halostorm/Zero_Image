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
com_path = r'../DatasetA_train_20180813/train.txt'


def load_data(dir, path):
    print("[INFO] loading images...")
    data = []
    filelist = []
    # grab the image paths and randomly shuffle them
    count = 0
    with open(path, 'r') as f:
        for line in f:
            if count<100:
                filelist.append(line.rstrip('\n').split('\t')[0])
            count += 1
    # loop over the input images
    for file in filelist:
        # load the image, pre-process it, and store it in the data list

        imagePath = dir + file

        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_rows, img_cols))
        image = img_to_array(image)

        data.append(image)
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0

    return data, filelist


def predict(labelPath, outputPath):
    label = {}

    with open(labelPath, 'r') as f:
        for line in f:
            line = line.split('\t')
            line = np.array(line)
            label[line[0]] = np.array(line[1:31])

    # model = densenet_reg.DenseNetImageNet264(input_shape=img_dim, classes=nb_classes)
    model = densenet_reg.DenseNet(img_dim, classes=nb_classes, depth=depth, nb_dense_block=nb_dense_block,
                                  growth_rate=growth_rate, nb_filter=nb_filter, dropout_rate=dropout_rate,
                                  bottleneck=bottleneck, reduction=reduction, weights=None)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-4)  # Using Adam instead of SGD to speed up training
    model.compile(loss=losses.mean_absolute_error, optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    testX, filelist = load_data(train_file_path, com_path)

    print(testX.shape)

    testX = testX.astype('float32')

    weights_file = r'../dataB/Zero_DenseNet_Reg.h5'
    if os.path.exists(weights_file):
        model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

        yPred = model.predict(testX)

        with open(outputPath, 'w+') as out:
            for i in range(len(yPred)):
                loss = 1000
                id = None
                lis = None
                for l in label.keys():
                    nLoss = Loss(yPred[i], label[l])
                    # print('nLoss:\t' + str(nLoss))
                    if loss > nLoss:
                        loss = nLoss
                        id = l
                        # lis = yPred
                # print('Loss-------------------\t' + str(loss))
                # print('Id-------------------\t' + str(id))
                out.write(filelist[i] + '\t' + id + '\n')


def Loss(yPre, yLabel):
    loss = 0

    yPre = yPre.astype('float32')

    yLabel = yLabel.astype('float32')

    for i in range(30):
        loss += np.abs(yPre[i] - yLabel[i])

    loss = loss / 30
    return loss


if __name__ == '__main__':
    predict(r'../DatasetB_20180919/attributes_per_class.txt', r'./outT.txt')
