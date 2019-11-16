import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import shutil

# 批次大小
BATCH_SIZE = 50
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160


def read_image_tensor(fp='./number_image'):
    '''
    将 3 通道图片 转换为 1 通道 矩阵
    '''

    label0 = []
    label1 = []
    label2 = []
    label3 = []
    image_batch = []

    # 打乱列表顺序
    filenames = os.listdir(path=fp)
    random.shuffle(filenames)

    # 随机选取一个批次
    filenames = random.sample(filenames, BATCH_SIZE)
    for i, filename in enumerate(filenames):
        file_path = os.path.join(fp + '/', filename)

        # 使用 plt.imread() 读出的 narray
        # image = plt.imread(fname=file_path)

        # 处理图片
        image = Image.open(fp=file_path)
        image = image.resize((128, 128))
        image = np.array(image.convert('L'))
        image_data = np.reshape(image, (128, 128, 1))

        if i == 0:
            image_batch = image_data
        else:
            image_batch = np.append(image_batch, image_data)
            image_batch = np.reshape(image_batch, (-1, 128, 128, 1))

        # 处理标签
        labels = filename.split('.')[0]
        label_list = list(labels)

        label0.append(int(label_list[0]))
        label1.append(int(label_list[1]))
        label2.append(int(label_list[2]))
        label3.append(int(label_list[3]))

    return image_batch, label0, label1, label2, label3


if __name__ == '__main__':
    image_data = tf.placeholder(dtype=tf.float32)
    label0_data = tf.placeholder(dtype=tf.int32)
    label1_data = tf.placeholder(dtype=tf.int32)
    label2_data = tf.placeholder(dtype=tf.int32)
    label3_data = tf.placeholder(dtype=tf.int32)

    # image 数据预处理
    #
    # 将 tensor 数据类型转换为 tf.float32
    image = tf.cast(x=image_data, dtype=tf.float32) / 255.0
    # 将 image 数据归一化为 -1 到 1 之间
    image = tf.subtract(x=image, y=0.5)  # x-y
    image = tf.multiply(x=image, y=2.0)  # x*y

    # 获取标签
    label0 = tf.cast(x=label0_data, dtype=tf.int32)
    label1 = tf.cast(x=label1_data, dtype=tf.int32)
    label2 = tf.cast(x=label2_data, dtype=tf.int32)
    label3 = tf.cast(x=label3_data, dtype=tf.int32)

    with tf.Session() as sess:

        for i in range(3):
            image_data_1, lable_data0, lable_data1, label_data2, label_data3 = read_image_tensor()

            train_image, train_label0, train_label1, train_label2, train_label3 = sess.run(fetches=[image, label0,
                                                                                                    label1, label2, 
                                                                                                    label3
                                                                                                    ],
                                                                                           feed_dict={image_data: image_data_1,
                                                                                                      label0_data: lable_data0, 
                                                                                                      label1_data: lable_data1,
                                                                                                      label2_data: label_data2, 
                                                                                                      label3_data: label_data3
                                                                                                      }
                                                                                            )

            print(train_label0, train_label1, train_label2, train_label3)
