from __future__ import print_function

import os.path

import numpy as np
import sklearn.metrics as metrics
from keras import losses

from keras.datasets import cifar10

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
import numpy as np
import random
import cv2
import os
import sys

import densenet

norm_size = 64
train_file_path = r'../DatasetA_train_20180813/train/'
com_path = r'../data/com.txt'


def load_data(dir, path):
    print("[INFO] loading images...")
    data = []
    labels = []

    data1 = []
    labels1 = []

    filelist = []
    # grab the image paths and randomly shuffle them
    with open(path, 'r') as f:
        for line in f:
            # print(line)
            filelist.append(line.split('\t'))

    # imagePaths = sorted(list(paths.list_images(path)))
    filelist = sorted(filelist)
    random.seed(42)
    random.shuffle(filelist)
    # loop over the input images
    count = 0
    for file in filelist:
        # load the image, pre-process it, and store it in the data list
        # print(file)

        imagePath = dir + file[0]
        # print(imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        # extract the class label from the image path and update the
        # labels list
        label = file[1:31]
        label = np.array(label)
        if count < 30000:
            data.append(image)
            labels.append(label)
        else:
            data1.append(image)
            labels1.append(label)
        count += 1

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    data1 = np.array(data1, dtype="float") / 255.0
    labels = np.array(labels)
    labels1 = np.array(labels1)

    # convert the labels from integers to vectors
    # labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels, data1, labels1


def train():
    batch_size = 20
    nb_classes = 10
    nb_epoch = 30

    img_rows, img_cols = 64, 64
    img_channels = 3

    img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (
        img_rows, img_cols, img_channels)
    depth = 40
    nb_dense_block = 3
    growth_rate = 12
    dropout_rate = 0.3  # 0.0 for data augmentation

    model = densenet.DenseNet(input_shape=img_dim, nb_classes=nb_classes, depth=depth, dense_blocks=nb_dense_block,
                              growth_rate=growth_rate, dropout_rate=dropout_rate)
    print("Model created")

    model.summary()
    optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
    model.compile(loss=losses.mean_squared_error, optimizer=optimizer, metrics=["accuracy"])
    print("Finished compiling")
    print("Building model...")

    # (trainX, trainY), (testX, testY) = cifar10.load_data()

    trainX, trainY, testX, testY = load_data(train_file_path, com_path)

    print(trainX.shape)
    print(trainY.shape)
    print(testX.shape)
    print(testY.shape)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')

    # trainX = densenet.preprocess_input(trainX)
    # testX = densenet.preprocess_input(testX)

    # Y_train = np_utils.to_categorical(trainY, nb_classes)
    # Y_test = np_utils.to_categorical(testY, nb_classes)

    generator = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5. / img_rows,
                                   height_shift_range=5. / img_cols,
                                   horizontal_flip=True)

    generator.fit(trainX, seed=0)

    # Load model
    weights_file = "weights/DenseNet-40-12-CIFAR10.h5"
    if os.path.exists(weights_file):
        # model.load_weights(weights_file, by_name=True)
        print("Model loaded.")

    out_dir = "weights/"

    lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1),
                                   cooldown=0, patience=5, min_lr=1e-5)
    model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                       save_weights_only=True, verbose=1)

    callbacks = [lr_reducer, model_checkpoint]

    model.fit_generator(generator.flow(trainX, None, batch_size=batch_size),
                        steps_per_epoch=len(trainX) // batch_size, epochs=nb_epoch,
                        callbacks=callbacks,
                        validation_data=(testX, None),
                        validation_steps=testX.shape[0] // batch_size, verbose=1)

    yPreds = model.predict(testX)
    yPred = np.argmax(yPreds, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)


if __name__ == '__main__':
    train()
