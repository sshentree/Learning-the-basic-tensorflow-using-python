import tensorflow as tf
import numpy as np
import os

# 导入自定义模块（只能转成 .py 格式文件）
from class_image_to_tensor import *

# 导入绘图自定义函数
from figure_scallar import *

# 不同字符数量
CHAR_SET_LEN = 10
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160

# 定义卷积神将网络必要函数
#
# 初始化权重


def weigth_variable(shape):
    initial = tf.truncated_normal(
        shape=shape, mean=0, stddev=1.0)  # 生成阶段正态分布数据
    return tf.Variable(initial_value=initial, dtype=tf.float32)

# 初始化偏置


def biase_variable(shape):
    initial = tf.constant(value=0.1, dtype=tf.float32)
    return tf.Variable(initial_value=initial, dtype=tf.float32)

# 卷积层


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 正则化


def regularizer(value):
    regularizer = tf.contrib.layers.l1_l2_regularizer()
    return tf.get_variable(initializer=value, regularizer=regularizer)


# 定义网络模型结构
#
# 第一层 输入层
#
# 初始化两个 placeholder 作为输入 images 、labels
x = tf.placeholder(dtype=tf.float32, shape=(None, 128, 128, 1))
# 多任务
y0 = tf.placeholder(dtype=tf.float32, shape=(None, 10))
y1 = tf.placeholder(dtype=tf.float32, shape=(None, 10))
y2 = tf.placeholder(dtype=tf.float32, shape=(None, 10))
y3 = tf.placeholder(dtype=tf.float32, shape=(None, 10))

# 隐层层
#
# 一
#
# 初始化卷积核权重
# shape=(height, width, channel, filter_num)
W_conv1 = weigth_variable(shape=(3, 3, 1, 32))
# 绘图
variable_scalars(W_conv1)
# 为每一个卷积核初始化一个偏置
# 为卷积后的 256 个特征图，每一个图加一个偏置 [[1, 2, 3], [4, 5, 6]] + 1 = [[2, 3, 4], [5, 6, 7]]
B_conv1 = biase_variable(shape=(1, 32))
variable_scalars(B_conv1)
# 激活函数 relu
h_conv1 = tf.nn.relu(features=(conv2d(x=x, W=W_conv1) + B_conv1))
# 池化
h_pool1 = max_pool_2x2(h_conv1)

# 二
#
# 初始化卷积核权重
W_conv2 = weigth_variable(shape=(3, 3, 32, 64))  # channel 是由上一层 filter_num 决定的
variable_scalars(W_conv2)
# 为每一个卷积核初始化一个偏置
B_conv2 = biase_variable(shape=(1, 64))
variable_scalars(B_conv2)
# 激活函数 relu
h_conv2 = tf.nn.relu(features=(conv2d(x=h_pool1, W=W_conv2) + B_conv2))
# 池化
h_pool2 = max_pool_2x2(h_conv2)

# 三
#
# 初始化卷积核权重
W_conv3 = weigth_variable(shape=(3, 3, 64, 128))
variable_scalars(W_conv3)
# 为每一个卷积核初始化一个偏置
B_conv3 = biase_variable(shape=(1, 128))
variable_scalars(W_conv3)
# 激活函数 relu
h_conv3 = tf.nn.relu(features=(conv2d(x=h_pool2, W=W_conv3) + B_conv3))
# 池化
h_pool3 = max_pool_2x2(h_conv3)

# 四
#
# 初始化卷积核权重
W_conv4 = weigth_variable(shape=(3, 3, 128, 256))
variable_scalars(W_conv4)
# 为每一个卷积核初始化一个偏置
B_conv4 = biase_variable(shape=(1, 256))
variable_scalars(B_conv4)
# 激活函数 relu
h_conv4 = tf.nn.relu(features=(conv2d(x=h_pool3, W=W_conv4) + B_conv4))
# 池化
h_pool4 = max_pool_2x2(h_conv4)

# 全连接层
#
# 扁平化处理
# 卷积不改变 image 尺寸，因为 padding='same'。每一池化 image 尺寸缩小 1 倍
h_flat = tf.reshape(tensor=h_pool4, shape=(-1, 8 * 8 * 256))

# 第一层全连接
#
# 初始化全连接权重
W_fc1 = weigth_variable(shape=(8 * 8 * 256, 1024))  # 设置全连接 1024 个神经元
variable_scalars(W_fc1)
# 偏置
B_fc1 = biase_variable(shape=(1, 1024))
variable_scalars(B_fc1)
# 激活函数
h_fc1 = tf.nn.relu(features=(tf.matmul(a=h_flat, b=W_fc1) + B_fc1))

# 第二层全连接（最后一层分类层，多任务模式）
W_fc2 = weigth_variable(shape=(1024, 10))
variable_scalars(W_fc2)
B_fc2 = biase_variable(shape=(1, 10))
variable_scalars(B_fc2)

# 激活函数（使用 softmax 作为分类器）
#
# 任务 0
predictions_0 = tf.nn.softmax(logits=(tf.matmul(a=h_fc1, b=W_fc2) + B_fc2))
# 任务 1
predictions_1 = tf.nn.softmax(logits=(tf.matmul(a=h_fc1, b=W_fc2) + B_fc2))
# 任务 2
predictions_2 = tf.nn.softmax(logits=(tf.matmul(a=h_fc1, b=W_fc2) + B_fc2))
# 任务 3
predictions_3 = tf.nn.softmax(logits=(tf.matmul(a=h_fc1, b=W_fc2) + B_fc2))


# 打印 y 、 predictions 形状
# print(y0.shape)
# print(predictions_0.shape)

# 定义损失函数（使用交叉熵损失函数）
#
# 任务 0
cross_entropy_0 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y0, logits=predictions_0))
tf.summary.scalar('loss0', cross_entropy_0)
# 任务 1
cross_entropy_1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y1, logits=predictions_1))
tf.summary.scalar('loss1', cross_entropy_1)
# 任务 2
cross_entropy_2 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y2, logits=predictions_2))
tf.summary.scalar('loss2', cross_entropy_2)
# 任务 3
cross_entropy_3 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y3, logits=predictions_3))
tf.summary.scalar('loss3', cross_entropy_3)

# 4 个任务总损失
total_loss = (cross_entropy_0 + cross_entropy_1 +
              cross_entropy_2 + cross_entropy_3) / 4.0

# 定义优化器（联合训练，使用一个优化器）
#
# 定义可变学习率
# trainable=False 防止被数据流图收集，在训练时尝试更新
global_step = tf.Variable(initial_value=0, dtype=tf.float32, trainable=False)

learning_rate = tf.train.exponential_decay(
    learning_rate=0.1, global_step=global_step, decay_steps=504, decay_rate=0.96, staircase=True)
train = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(
    loss=total_loss, global_step=global_step)
tf.summary.scalar('learning', learning_rate)

# 计算任务准确率（四个任务分开计算）
#
# 任务 0
correct_predictions_0 = tf.equal(x=tf.argmax(input=y0, axis=1), y=tf.argmax(
    input=predictions_0, axis=1))  # 函数 tf.argmax()，默认 axis=0 ，获取'第一维'最大值的索引
# 任务 1
correct_predictions_1 = tf.equal(x=tf.argmax(
    input=y1, axis=1), y=tf.argmax(input=predictions_1, axis=1))
# 任务 2
correct_predictions_2 = tf.equal(x=tf.argmax(
    input=y2, axis=1), y=tf.argmax(input=predictions_2, axis=1))
# 任务 3
correct_predictions_3 = tf.equal(x=tf.argmax(
    input=y3, axis=1), y=tf.argmax(input=predictions_3, axis=1))

accuracy_0 = tf.reduce_mean(input_tensor=tf.cast(x=correct_predictions_0, dtype=tf.float32))
tf.summary.scalar('acc0', accuracy_0)
accuracy_1 = tf.reduce_mean(input_tensor=tf.cast(x=correct_predictions_1, dtype=tf.float32))
tf.summary.scalar('acc1', accuracy_1)
accuracy_2 = tf.reduce_mean(input_tensor=tf.cast(x=correct_predictions_2, dtype=tf.float32))
tf.summary.scalar('acc2', accuracy_2)
accuracy_3 = tf.reduce_mean(input_tensor=tf.cast(x=correct_predictions_3, dtype=tf.float32))
tf.summary.scalar('acc3', accuracy_3)

# 获取样本数据、标签
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


# 将 labels 转换为 one-hot
one_hot_label0 = tf.one_hot(indices=label0, depth=CHAR_SET_LEN)
one_hot_label1 = tf.one_hot(indices=label1, depth=CHAR_SET_LEN)
one_hot_label2 = tf.one_hot(indices=label2, depth=CHAR_SET_LEN)
one_hot_label3 = tf.one_hot(indices=label3, depth=CHAR_SET_LEN)

# 合并所有 summary
merged = tf.summary.merge_all()
# 初始化变量
init_op = tf.global_variables_initializer()
# 迭代次数
epochs = 2
# 用于保存模型
saver = tf.train.Saver()


# 计算图
with tf.Session() as sess:
    # 创建 writer，写入日志文件
    writer = tf.summary.FileWriter('logs/', tf.get_default_graph())

    # 初始化变量
    sess.run(init_op)

    for epoch in range(epochs):
        for i in range(126):

            image_data_1, lable_data0, lable_data1, label_data2, label_data3 = read_image_tensor()

            train_image, train_label0, train_label1, train_label2, train_label3 = sess.run(fetches=[image,
                                                                                                    one_hot_label0,
                                                                                                    one_hot_label1,
                                                                                                    one_hot_label2,
                                                                                                    one_hot_label3,
                                                                                                    ],
                                                                                           feed_dict={image_data: image_data_1,
                                                                                                      label0_data: lable_data0, 
                                                                                                      label1_data: lable_data1,
                                                                                                      label2_data: label_data2, 
                                                                                                      label3_data: label_data3
                                                                                                      }
                                                                                            )


            # 训练、计算 loss
            _, loss, res_merge = sess.run(fetches=[train, total_loss, merged], feed_dict={x: train_image, y0: train_label0,
                                                                       y1: train_label1, y2: train_label2, y3: train_label3})
            # 将所有 summary 写入文件
            writer.add_summary(summary=res_merge, global_step=epoch)
            # 准确率
        acc0, acc1, acc2, acc3 = sess.run(fetches=[accuracy_0, accuracy_1, accuracy_2, accuracy_3],
                                          feed_dict={x: train_image, y0: train_label0, y1: train_label1,
                                                     y2: train_label2, y3: train_label3})

        lr = sess.run(learning_rate)
        acc_total = (acc0 + acc1 + acc2 + acc3) / 4.0
        print('Iter %d  Loss: %.4f  Accuracy: [%0.3f, %.3f, %.3f, %.3f, %.3f]  Learning_rate: %.4f' % (
            epoch, loss, acc_total, acc0, acc1, acc2, acc3, lr))

    writer.close()
    saver.save(sess=sess, save_path='model/crack_captcha.model')
