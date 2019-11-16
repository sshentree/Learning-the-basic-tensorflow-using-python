import tensorflow as tf
import numpy as np


# 定义批次大小
BATCH_SIZE = 20
# 标签 10 种类别
CHAR_SET_LEN = 10
TFRECORD_PATH = '../class_9_1/captcha_image/tfrecord/train.tfrecords'

# 读取 tfrecord 格式文件
def read_and_decode(fp=TFRECORD_PATH):
    '''
    解析 TFrecord 格式文件
    param
        fp:文件名
    '''
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer(string_tensor=[fp])
    # 返回文件名和文件
    _, serialized_example = tf.TFRecordReader().read(queue=filename_queue)
    
   # 给出构建形状及类型标签
    feature_description = {
    'image_data': tf.FixedLenFeature(shape=[], dtype=tf.string, default_value=''),
    'label0': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
    'label1': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
    'label2': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0),
    'label3': tf.FixedLenFeature(shape=[], dtype=tf.int64, default_value=0)
    }
    
    # 解析 TFrecord 格式文件
    # 获取了文件所有内容
    features = tf.parse_single_example(serialized_example, feature_description)
    
    return features


def image_labels_to_tensor(features):
    '''
    将 TFrecord 格式文件解析出的 features 数据提取，并将标签转换为 one-hot
    param
        features: TFrecord 构造格式
    '''
#     # 解析 TFrecord 文件函数
#     features = read_and_decode()

    # 将 bytes 字符串重新转换为 Tensor
    image = tf.decode_raw(bytes=features['image_data'], out_type=tf.uint8)
    # tensor 
    image = tf.reshape(tensor=image, shape=(224, 224, 1))

    # image 数据预处理
    #
    # 将 tensor 数据类型转换为 tf.float32
    image = tf.cast(x=image, dtype=tf.float32) / 255.0
    # 将 image 数据归一化为 -1 到 1 之间
    image = tf.subtract(x=image, y=0.5) # x-y
    image = tf.multiply(x=image, y=2.0) # x*y

    # 获取标签
    label0 = tf.cast(x=features['label0'], dtype=tf.int32)
    label1 = tf.cast(x=features['label1'], dtype=tf.int32)
    label2 = tf.cast(x=features['label2'], dtype=tf.int32)
    label3 = tf.cast(x=features['label3'], dtype=tf.int32)

    image_labels = [image, label0, label1, label2, label3]


    # 使用 shuffle_batch 可以随机打乱顺序
    image_batch, label0_batch, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(tensors=image_labels, 
                                                                                                 batch_size=BATCH_SIZE, 
                                                                                                 capacity=50000, 
                                                                                                 min_after_dequeue=10000, 
                                                                                                 num_threads=1
                                                                                                )


    # 将 labels 转换为 one-hot
    one_hot_label0 = tf.one_hot(indices=label0_batch, depth=CHAR_SET_LEN)
    one_hot_label1 = tf.one_hot(indices=label1_batch, depth=CHAR_SET_LEN)
    one_hot_label2 = tf.one_hot(indices=label2_batch, depth=CHAR_SET_LEN)
    one_hot_label3 = tf.one_hot(indices=label3_batch, depth=CHAR_SET_LEN)


    # print(image_batch.shape)
    # print(label0_batch.shape)
    # print(one_hot_label0.shape)


    # (50, 224, 224, 1) 
    # (50,)
    # (50, 10)
    
    return image_batch, one_hot_label0, one_hot_label1, one_hot_label2, one_hot_label3    