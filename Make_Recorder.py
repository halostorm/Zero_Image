# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
import os
import scipy
import time
from PIL import Image

'''tfrecord 写入数据.
将图片数据写入 tfrecord 文件。以 MNIST png格式数据集为例。

首先将图片解压到 ../../MNIST_data/mnist_png/ 目录下。
解压以后会有 training 和 testing 两个数据集。在每个数据集下，有十个文件夹，分别存放了这10个类别的数据。
每个文件夹名为对应的类别编码。

现在网上关于打包图片的例子非常多，实现方式各式各样，效率也相差非常多。
选择合适的方式能够有效地节省时间和硬盘空间。
有几点需要注意：
1.打包 tfrecord 的时候，千万不要使用 Image.open() 或者 matplotlib.image.imread() 等方式读取。
 1张小于10kb的png图片，前者（Image.open) 打开后，生成的对象100+kb, 后者直接生成 numpy 数组，大概是原图片的几百倍大小。
 所以应该直接使用 tf.gfile.FastGFile() 方式读入图片。
2.从 tfrecord 中取数据的时候，再用 tf.image.decode_png() 对图片进行解码。
3.不要随便使用 tf.image.resize_image_with_crop_or_pad 等函数，可以直接使用 tf.reshape()。前者速度极慢。
4.如果有固态硬盘的话，图片数据一定要放在固态硬盘中进行读取，速度能高几十倍几十倍几十倍！生成的 tfrecord 文件就无所谓了，找个机械盘放着就行。
'''

# png 文件路径
TRAINING_DIR = '/home/halo/Workspace/ALi_ZhiJiang_Competation/DatasetA_train_20180813/train/'
TESTING_DIR = '/home/halo/Workspace/ALi_ZhiJiang_Competation/DatasetA_train_20180813/test/'

TRAINING_COM = '../data/com.txt'

# tfrecord 文件保存路径,这里只保存一个 tfrecord 文件
TRAINING_TFRECORD_NAME = 'training.tfrecord'
TESTING_TFRECORD_NAME = 'testing.tfrecord'


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def convert_tfrecord_dataset(dataset_dir, tfrecord_name, tfrecord_path='../data/'):
    """ convert samples to tfrecord dataset.
    Args:
        dataset_dir: 数据集的路径。
        tfrecord_name: 保存为 tfrecord 文件名
        tfrecord_path: 保存 tfrecord 文件的路径。
    """
    if not os.path.exists(dataset_dir):
        print(u'png文件路径错误')
        exit()
    if not os.path.exists(os.path.dirname(tfrecord_path)):
        os.makedirs(os.path.dirname(tfrecord_path))
    tfrecord_file = os.path.join(tfrecord_path, tfrecord_name)

    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for line in open(TRAINING_COM, 'r'):
            line = line.split("\t")
            imagePath = dataset_dir + line[0]
            # print(imagePath)
            img_raw = Image.open(imagePath)
            img_raw = img_raw.resize((64, 64))
            img = img_raw.tobytes()  # 将图片转化为二进制格式

            label_id = line[1:]

            label = np.zeros([30], dtype=np.float32)

            for i in range(0,len(label_id)):
                label[i] = (np.float32(label_id[i]))

            label = label.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': bytes_feature(img),
                        'label': bytes_feature(label)
                    }))
            serialized = example.SerializeToString()

            writer.write(serialized)
    print('\nFinished writing data to tfrecord files.')


    # class_names = os.listdir(dataset_dir)
    #
    # n_class = len(class_names)
    # print(u'一共有 %d 个类别' % n_class)
    # with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
    #     for class_name in class_names:  # 对于每个类别
    #         class_dir = os.path.join(dataset_dir, class_name)  # 获取类别对应的文件夹路径
    #         file_names = os.listdir(class_dir)  # 在该文件夹下，获取所有图片文件名
    #         label_id = DICT_LABEL_TO_ID.get(class_name)  # 获取类别 id
    #         print(u'\n正在处理类别 %d 的数据' % label_id)
    #         time0 = time.time()
    #         n_sample = len(file_names)
    #         for i in range(n_sample):
    #             file_name = file_names[i]
    #             sys.stdout.write('\r>> Converting image %d/%d , %g s' % (
    #                 i + 1, n_sample, time.time() - time0))
    #             png_path = os.path.join(class_dir, file_name)  # 获取每个图片的路径
    #             # CNN inputs using
    #             img = tf.gfile.FastGFile(png_path, 'rb').read()  # 读入图片
    #             example = tf.train.Example(
    #                 features=tf.train.Features(
    #                     feature={
    #                         'image': bytes_feature(img),
    #                         'label': int64_feature(label_id)
    #                     }))
    #             serialized = example.SerializeToString()
    #             writer.write(serialized)
    # print('\nFinished writing data to tfrecord files.')


if __name__ == '__main__':
    convert_tfrecord_dataset(TRAINING_DIR, TRAINING_TFRECORD_NAME)
    # convert_tfrecord_dataset(TESTING_DIR, TESTING_TFRECORD_NAME)
