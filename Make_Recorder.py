import os
import tensorflow as tf
from PIL import Image

# 原始图片的存储位置
img_folder = r'../DatasetA_train_20180813/train/'

TRAINING_COM = '../data/com.txt'


# 制作TFRecords数据
def create_record(img_folder):
    writer = tf.python_io.TFRecordWriter("../data/train.tfrecords")
    with open(TRAINING_COM,'r') as file:
        for line in file:
            line  = line.split('\t')
            img_path = img_folder + line[0]
            img = Image.open(img_path)
            img = img.resize((64, 64))  # 设置需要转换的图片大小
            img_raw = img.tobytes()  # 将图片转化为原生bytes

            labels = line[1:31]
            label = []
            for i in labels:
                label.append(float(i))

            # print(label)
            # print(img_raw)
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    create_record(img_folder)