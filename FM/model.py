import tensorflow as tf
import numpy as np


# FM模型
class FmModel(object):
    def __init__(self, i, x, y, feature_num, valid_num, hidden_num):
        self.i = i
        self.x = x
        self.y = y
        self.feature_num = feature_num  # 获取特征数，这个值要建Variable，所以不能动态获取
        self.valid_num = valid_num  # 获取有效特征数，这个值要建Variable，所以不能动态获取
        self.sample_num = tf.shape(x)[0]  # 获取样本数
        self.hidden_num = hidden_num  # 获取隐藏特征维度

        self.bias = tf.Variable([0.0])
        self.weight = tf.Variable(tf.random_normal([self.feature_num, 1], 0.0, 1.0))
        self.weight_mix = tf.Variable(tf.random_normal([self.feature_num, self.hidden_num], 0.0, 1.0))

        x_ = tf.reshape(self.x, [self.sample_num, self.valid_num, 1])  # SAMPLE_NUM*VALID_NUM*1
        w_ = tf.nn.embedding_lookup(self.weight, self.i)  # SAMPLE_NUM*VALID_NUM*1

        expressings = tf.multiply(x_, w_)  # SAMPLE_NUM*VALID_NUM*1
        expressings_reduce = tf.reshape(self.x, [self.sample_num, self.valid_num])  # SAMPLE_NUM*VALID_NUM

        x__ = tf.tile(x_, [1, 1, self.hidden_num])  # SAMPLE_NUM*VALID_NUM*HIDDEN_NUM
        w__ = tf.nn.embedding_lookup(self.weight_mix, self.i)  # SAMPLE_NUM*VALID_NUM*HIDDEN_NUM

        embeddings = tf.multiply(x__, w__)  # SAMPLE_NUM*VALID_NUM*HIDDEN_NUM
        embeddings_sum = tf.reduce_sum(embeddings, 1)  # SAMPLE_NUM*HIDDEN_NUM
        embeddings_sum_square = tf.square(embeddings_sum)  # SAMPLE_NUM*HIDDEN_NUM
        embeddings_square = tf.square(embeddings)  # SAMPLE_NUM*VALID_NUM*HIDDEN_NUM
        embeddings_square_sum = tf.reduce_sum(embeddings_square, 1)  # SAMPLE_NUM*HIDDEN_NUM

        z = self.bias + \
            tf.reduce_sum(expressings_reduce, 1, keepdims=True) + \
            1.0 / 2.0 * tf.reduce_sum(tf.subtract(embeddings_sum_square, embeddings_square_sum), 1, keepdims=True)
        z_ = tf.clip_by_value(z, -4.0, 4.0)

        self.hypothesis = tf.sigmoid(z_)

        self.y_expand = tf.expand_dims(self.y, axis=1)

        self.loss = tf.losses.log_loss(self.y_expand, self.hypothesis)


tf.reset_default_graph()  # 清空Graph

VALID_NUM = 8  # 有效特征数量
with tf.name_scope("input"):
    i = tf.placeholder(tf.int32, shape=[None, VALID_NUM])
    x = tf.placeholder(tf.float32, shape=[None, VALID_NUM])
    y = tf.placeholder(tf.int32, shape=[None])
    y_expand = tf.expand_dims(y, axis=1)

FEATURE_NUM = 20  # 特征数量
HIDDEN_NUM = 5  # 隐藏特征维度
with tf.name_scope("fm"):
    fm = FmModel(i, x, y, FEATURE_NUM, VALID_NUM, HIDDEN_NUM)

LEARNING_RATE = 0.02  # 学习速率
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    training_op = optimizer.minimize(fm.loss)

THRESHOLD = 0.5  # 判断门限
with tf.name_scope("eval"):
    predictions = tf.to_int32(fm.hypothesis - THRESHOLD)
    corrections = tf.equal(predictions, fm.y_expand)
    accuracy = tf.reduce_mean(tf.cast(corrections, tf.float32))

init = tf.global_variables_initializer()  # 初始化所有变量

EPOCH = 1000  # 迭代次数
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(EPOCH):
        _training_op, _loss = sess.run([training_op, fm.loss],
                                       feed_dict={i: np.array([np.random.choice(20, 8) for cnt in range(10)]),
                                                  x: np.random.rand(10, 8), y: np.random.randint(2, size=10)})
        _accuracy = sess.run([accuracy],
                             feed_dict={i: np.array([np.random.choice(20, 8) for cnt in range(5)]),
                                        x: np.random.rand(5, 8), y: np.random.randint(2, size=5)})
        print("epoch:", epoch, _loss, _accuracy)
