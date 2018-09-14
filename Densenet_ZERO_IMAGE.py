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
batch_size = 10

# Image params
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = 64, 64, 3

LABEL_SIZE = 30

total_epochs = 20


def read_and_decode(tf_folder):
    filepaths = [os.path.join(tf_folder,file) for file in os.listdir(tf_folder)]
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(filepaths)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example( serialized_example, features={
                        'image': tf.FixedLenFeature([], tf.string),
                        'label': tf.FixedLenFeature([], tf.string)
                        })
    image = tf.decode_raw( features['image'], tf.uint8 )
    label = tf.decode_raw( features['label'], tf.float64 )
    image = tf.reshape(image, [64, 64, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(tf.reshape(label, [30]), tf.float64)
    return image, label


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
    return tf.layers.dense(inputs=x, units=30, name='linear')


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
    label = tf.placeholder(shape=[None, LABEL_SIZE], dtype=tf.float64, name='label_palceholder')

    train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    with tf.name_scope('batch'):
        imgs, labels = read_and_decode('../records/')

        # 使用shuffle_batch可以随机打乱输入
        img_batch, label_batch = tf.train.shuffle_batch([imgs, labels],
                                                        batch_size=batch_size,
                                                        num_threads = 4,
                                                        capacity=1200,
                                                        min_after_dequeue=500)

    with tf.variable_scope('net'):
        y = DenseNet(image, nb_block, growth_k, train_flag).model


    with tf.name_scope('loss'):
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = y))
        loss = tf.reduce_mean(tf.square(label - y))


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
        print(str(i) + ":\t epoch")

        image_data, label_data = sess.run([img_batch, label_batch])

        print("image data")
        print(image_data.shape)
        print("label data")
        print(label_data.shape)

        _, loss_data, data, summary_str = sess.run([train_step, loss, y],
                                                   feed_dict={train_flag: True, image: image_data,
                                                              label: label_data})
        # print summary_str
        writer.add_summary(summary_str, i)

        print('iter: %i, loss: %f' % (i, loss_data))

    saver.save(sess=sess, save_path='./model/dense.ckpt')


if __name__ == '__main__':
    # trainRecords = r'../data/training.tfrecord'
    train()
