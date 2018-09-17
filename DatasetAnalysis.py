import os


# import numpy as np
# import time
# import pickle
# import random as RD
# import matplotlib.pyplot as plt
# import types
#
# from mpl_toolkits.mplot3d import Axes3D
#
# from sklearn import datasets
# from sklearn.externals import joblib
# from sklearn.cluster import KMeans
# from sklearn import svm
# from sklearn.datasets import load_svmlight_file
#
# from sklearn.ensemble import GradientBoostingClassifier
#
# from sklearn.metrics import label_ranking_average_precision_score as LRAP
# from sklearn.metrics import coverage_error
#
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import train_test_split
#
# import xgboost as xgb


class Handle:
    def __init__(self, image_data_path, train_path, attribute_path, word_path, label_path):
        self.image_data_path = image_data_path
        self.train_path = train_path
        self.attribute_path = attribute_path
        self.word_path = word_path
        self.label_path = label_path

        self.image_data = None
        self.train = None
        self.attribute = None
        self.word = None
        self.label = None

        self.label_feature = {}

        self.image_label = {}

    def inputImageFolder(self):
        print()
        for root, dirs, fileName in os.walk(self.image_data_path):
            for i in fileName:
                file = os.path.join(root, i)
                self.readImage(file)

    def readImage(self, file):
        print()

    def readFile1(self, file):

        for line in open(file, 'r'):
            line = line.strip().split('\t')

            self.image_label[line[0]] = line[1]

    def readFile2(self, file):

        for line in open(file, 'r'):
            line = line.strip().split('\t')

            self.label_feature[line[0]] = line[1:]

    def writeFile(self, dataFile, featureFile):
        data = []
        feature = []
        for i in self.image_label.keys():
            for j in self.label_feature.keys():

                if self.image_label[i] == j:
                    data.append(i)
                    feature.append(self.label_feature[j])

        with open(dataFile, 'w+', ) as dataF:
            i = 0
            for l in data:
                dataF.write(str(l) + "\t" + str(i) + "\t" + "\n")
                i += 1
        with open(featureFile, 'w+', ) as featureF:
            i = 0
            for l in feature:
                featureF.write(str(i) + "\t")
                for item in l:
                    featureF.write(str(item) + "\t")
                featureF.write("\n")
                i += 1

    def writeFileCombine(self, comFile):
        data = []
        feature = []
        for i in self.image_label.keys():
            for j in self.label_feature.keys():

                if self.image_label[i] == j:
                    data.append(i)
                    feature.append(self.label_feature[j])

        with open(comFile, 'w+', ) as dataF:
            for i in range(0, len(data)):
                dataF.write(data[i] + "\t")
                for j in feature[i]:
                    dataF.write(str(j) + "\t")
                dataF.write("\n")


if __name__ == '__main__':
    image_data_path = r'../DatasetA_train_20180813/train/'
    train_path = r'../DatasetA_train_20180813/train.txt'
    attribute_path = r'../DatasetA_train_20180813/attributes_per_class.txt'
    word_path = r'../DatasetA_train_20180813/class_wordembeddings.txt'
    label_path = r'../DatasetA_train_20180813/label_list.txt'
    data_path = r'../ZERO_IMAGE/data/data_train.txt'
    feature_path = r'../ZERO_IMAGE/data/feature_train.txt'
    com_train_path = r'../ZERO_IMAGE/data/com_train.txt'

    # image_data_path = r'../DatasetA_test_20180813/test/'
    # test_path = r'../DatasetA_test_20180813/test.txt'
    # attribute_path = r'../DatasetA_test_20180813/attributes_per_class.txt'
    # word_path = r'../DatasetA_test_20180813/class_wordembeddings.txt'
    # label_path = r'../DatasetA_test_20180813/label_list.txt'
    # data_path = r'../ZERO_IMAGE/data/data_test.txt'
    # feature_path = r'../ZERO_IMAGE/data/feature_test.txt'
    # com_test_path = r'../ZERO_IMAGE/data/com_test.txt'

    handle = Handle(image_data_path, train_path, attribute_path, word_path, label_path)

    handle.readFile1(handle.train_path)
    handle.readFile2(handle.attribute_path)

    # print(handle.label_feature)

    # print(handle.image_label)

    # handle.writeFile(data_path, feature_path)

    handle.writeFileCombine(com_test_path)
    # handle.readFile(handle.word_path)
    # handle.readFile(handle.label_path)
