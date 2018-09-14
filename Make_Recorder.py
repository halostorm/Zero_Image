import os, math
import tensorflow as tf
from PIL import Image
import numpy as np

# 原始图片的存储位置
img_folder = r'../DatasetA_train_20180813/train/'

def readlist(filepath):
    image_name = []
    label_vect = []
    f = open(filepath)
    line = f.readline()
    while line:
        line = line.rstrip('\n').rstrip('\t').split('\t')
        image_name.append(line[0])
        label_vect.append(np.array([float(v) for v in line[1:]]))
        line = f.readline()
    f.close()
    return image_name, label_vect

# 制作TFRecords数据
def create_record(image_name, label_vect, save_dir, parts):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    per_num = math.ceil(len(image_name)/(parts+0.))
    for i in range(parts):
        cur_name = image_name[i*per_num:min(i*per_num+per_num,len(image_name))]
        cur_label= label_vect[i*per_num:min(i*per_num+per_num,len(image_name))]
        writer = tf.python_io.TFRecordWriter( os.path.join(save_dir,'%d.tfrecords'%(i+1)) )
        for j,name in enumerate(cur_name):
            image = Image.open( os.path.join(img_folder,cur_name[j]) )
            image = np.array(image.resize((64, 64)))
            label = cur_label[j]
            example = tf.train.Example( features=tf.train.Features( feature={
                'image': tf.train.Feature( bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature( bytes_list=tf.train.BytesList(value=[label.tobytes()]))
                }) )
            writer.write( example.SerializeToString() )
        writer.close()


if __name__ == '__main__':
    image_name, label_vect = readlist('../data/com.txt')
    create_record(image_name, label_vect, '../records/', 2)
