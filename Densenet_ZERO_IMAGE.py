# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tflearn.layers.conv import global_avg_pool
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np

# Hyper parameter
growth_k = 12
nb_block = 2  # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8  # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 10
batch_size = 10

# Image params
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = 64, 64, 3

LABEL_SIZE = 30

total_epochs = 20


def read_example(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    _, serialized_example = reader.read(filename_queue)

    min_queue_examples = 10
    batch = tf.train.shuffle_batch([serialized_example], batch_size=batch_size,
                                   capacity=min_queue_examples + 100 * batch_size, min_after_dequeue=min_queue_examples,
                                   num_threads=2)

    parsed_example = tf.parse_example(batch, features={'image': tf.FixedLenFeature([12288], tf.string),
                                                       'label': tf.FixedLenFeature([30], tf.float32)})

    image_raw = tf.decode_raw(parsed_example['image'], tf.uint8)
    image = tf.cast(tf.reshape(image_raw, [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]), tf.float32)
    image = image / 255.0

    label_raw = parsed_example['label']
    label = tf.cast(tf.reshape(label_raw, [batch_size, LABEL_SIZE]), tf.float32)

    print("image:")
    print(image)
    print("label:")
    print(label)

    return image, label


def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'img': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

    label_raw = features['label']
    label = tf.cast(tf.reshape(label_raw, [LABEL_SIZE]), tf.float32)

    return img, label


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network


def dense_layer(input, output_size, layer_name="dense"):
    with tf.name_scope(layer_name):
        network = tf.layers.dense(inputs=input, units=output_size)
        return network


def Global_Average_Pooling(x, stride=1):
    # width = np.shape(x)[1]
    # height = np.shape(x)[2]
    # pool_size = [width, height]
    # return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    #
    # It is global average pooling without tflearn

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def Drop_out(x, rate, training):
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def Relu(x):
    return tf.nn.relu(x)


def Average_pooling(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3, 3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def Linear(x):
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3, 3], stride=2)

        for i in range(self.nb_blocks):
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_' + str(i))
            x = self.transition_layer(x, scope='trans_' + str(i))

        # x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        # x = self.transition_layer(x, scope='trans_1')
        #
        # x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        # x = self.transition_layer(x, scope='trans_2')
        #
        # x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        # x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)

        x = flatten(x)

        x = dense_layer(x, LABEL_SIZE)

        # x = flatten(x)
        # x = Linear(x)

        # x = tf.reshape(x, [-1, 10])
        return x


def train():
    image = tf.placeholder(shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH], dtype=tf.float32,
                           name='image_placeholder')
    label = tf.placeholder(shape=[None, LABEL_SIZE], dtype=tf.float32, name='label_palceholder')

    train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    # init = tf.initialize_all_variables()
    #
    # with tf.Session() as sess:
    #     sess.run(init)
    #     threads = tf.train.start_queue_runners(sess=sess)
    #     for i in range(3):
    #         val, l = sess.run([img_batch, label_batch])
    #         # 我们也可以根据需要对val， l进行处理
    #         # l = to_categorical(l, 12)
    #         print(val.shape, l)

    with tf.name_scope('batch'):
        imgs, labels = read_and_decode("../data/train.tfrecords")

        # 使用shuffle_batch可以随机打乱输入
        img_batch, label_batch = tf.train.shuffle_batch([imgs, labels],
                                                        batch_size=batch_size, capacity=2000,
                                                        min_after_dequeue=1000)

    with tf.variable_scope('net'):
        y = DenseNet(image, nb_block, growth_k, train_flag).model

    with tf.name_scope('loss'):
        print("label1:")
        print(label)
        print("y1:")
        print(y)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(label, y))

    opt = tf.train.AdamOptimizer(learning_rate=init_learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    saver = tf.train.Saver()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    writer = tf.summary.FileWriter("../logs", sess.graph)
    if False:
        checkpoint_dir = './model/'

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print ckpt
        if ckpt and ckpt.model_checkpoint_path:

            # ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_name = 'dense.ckpt'
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print('True')
        else:
            print('False')

    for i in range(total_epochs):
        image_data, label_data = sess.run([img_batch, label_batch])

        _, loss_data, data, summary_str = sess.run([train_step, loss, y],
                                                   feed_dict={train_flag: True, image: image_data,
                                                              label: label_data})
        # print summary_str
        writer.add_summary(summary_str, i)

        print('iter: %i, loss: %f' % (i, loss_data))

    saver.save(sess=sess, save_path='./model/dense.ckpt')


def read_iamge(Root_path, batch_size):
    imagepaths = []
    labels = []
    label = 0
    classes = sorted(os.walk(Root_path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(Root_path, c)
        walk = os.walk(c_dir).__next__()[2]
        for sample in walk:
            if sample.endswith('.jpg') and sample.endswith('.jpeg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label += 1

    # 将iamgepaths 和 labels 转换为tf可以处理的格式
    imagepaths = tf.convert_to_tensor(imagepaths, tf.string)
    labels = tf.convert_to_tensor(labels, tf.int32)

    # 建立 Queue
    imagepath, label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)

    # 读取图片，并进行解码
    image = tf.read_file(imagepath)
    image = tf.image.decode_jpeg(image, channels=IMAGE_DEPTH)

    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）
    image = tf.image.resize_images(image, size=[IMAGE_HEIGHT, IMAGE_WIDTH])
    image = image * 1.0 / 127.5 - 1.0

    # 创建 batch
    X, Y = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=batch_size * 8)
    return X, Y


if __name__ == '__main__':
    print()

    trainRecords = r'../data/training.tfrecord'
    # img, label = read_example(trainRecords)

    train()
