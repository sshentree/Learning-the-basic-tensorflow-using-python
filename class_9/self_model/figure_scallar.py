import tensorflow as tf


# 定义绘张量图函数
def variable_scalars(var):
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var) # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean))) # 均方差
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histigram', var)