import tensorflow as tf
from DeepFM.build_data import load_data
import numpy as np

'''
    train_data['xi'] 
    train_data['xv'] 
    train_data['feat_dim'] 
'''

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.flags.DEFINE_integer('embedding_size', 8, 'Embedding_size')
tf.flags.DEFINE_integer('feature_sizes', 1, 'feature_sizes')
tf.flags.DEFINE_integer('field_size', 1, 'field_size')
tf.flags.DEFINE_list('deep_layers', [512, 256, 128], 'deep_layers')
tf.flags.DEFINE_list('dropout_deep', [0.5, 0.5, 0.5, 0.5], 'dropout_deep')
tf.flags.DEFINE_integer('numeric_feature_size', 1, 'numeric_feature_size')
tf.flags.DEFINE_integer('cross_layer_num', 3, 'cross_layer_num')
tf.flags.DEFINE_string('deep_layers_activation', 'relu', 'deep_layers_activation')
tf.flags.DEFINE_string('loss', 'logloss', 'loss')
tf.flags.DEFINE_integer('batch_size', 256, 'batch_size')
tf.flags.DEFINE_integer('epoch', 10, 'epoch')


class DCN():
    def __init__(self):
        self.weights = dict()
        self.X_sparse = tf.placeholder(tf.float32, shape=[None, None], name='X_sparse')
        self.X_sparse_index = tf.placeholder(tf.int32, shape=[None, None], name='X_sparse_index')
        self.X_dense = tf.placeholder(tf.float32, shape=[None, None], name='X_dense')
        self.label = tf.placeholder(tf.int32, shape=[None, 1], name='data_y')
        self.build_model()

    def build_model(self):

        self.weights['feature_weight'] = tf.Variable(
            tf.random_normal([FLAGS.feature_sizes, FLAGS.embedding_size], 0.0, 0.01),
            name='feature_weight')
        self.embedding_index = tf.nn.embedding_lookup(self.weights['feature_weight'],
                                                      self.X_sparse_index)  # Batch*F*K
        sparse_value = tf.reshape(self.X_sparse, shape=[-1, FLAGS.field_size, 1])
        self.embedding_part = tf.multiply(self.embedding_index, sparse_value)
        self.input = tf.concat(
            [self.X_dense, tf.reshape(self.embedding_part, shape=[-1, FLAGS.field_size * FLAGS.embedding_size])],
            axis=1)
        self.total_size = FLAGS.field_size * FLAGS.embedding_size + FLAGS.numeric_feature_size
        self.input = tf.reshape(self.input, [-1, self.total_size, 1])

        # cross part
        for i in range(FLAGS.cross_layer_num):
            self.weights['cross_layer_weight_{0}'.format(i)] = tf.Variable(
                tf.random_normal([self.total_size, 1], 0.0, 0.01), tf.float32)
            self.weights['cross_layer_bias_{0}'.format(i)] = tf.Variable(
                tf.random_normal([self.total_size, 1], 0.0, 0.01), tf.float32)

        x_now = self.input
        for i in range(FLAGS.cross_layer_num):
            x_now = tf.add(tf.add(tf.tensordot(tf.matmul(self.input, x_now, transpose_b=True),
                                               self.weights['cross_layer_weight_{0}'.format(i)], axes
                                               =1), self.weights['cross_layer_bias_{0}'.format(i)]), x_now)
        self.cross_network_out = tf.reshape(x_now, (-1, self.total_size))

        print(self.cross_network_out)

        # deep part
        deep_layer_num = len(FLAGS.deep_layers)
        for i in range(deep_layer_num):
            if (i == 0):
                self.weights['deep_layer_weight_{0}'.format(i)] = tf.Variable(
                    tf.random_normal([self.total_size, FLAGS.deep_layers[0]], 0.0, 0.01), tf.float32)
                self.weights['deep_layer_bias_{0}'.format(i)] = tf.Variable(
                    tf.random_normal([1, FLAGS.deep_layers[0]], 0.0, 0.01), tf.float32)
            else:
                self.weights['deep_layer_weight_{0}'.format(i)] = tf.Variable(
                    tf.random_normal([FLAGS.deep_layers[i - 1], FLAGS.deep_layers[i]], 0.0, 0.01), tf.float32)
                self.weights['deep_layer_bias_{0}'.format(i)] = tf.Variable(
                    tf.random_normal([1, FLAGS.deep_layers[i]], 0.0, 0.01), tf.float32)

        self.input = tf.reshape(self.input, [-1, self.total_size])
        deep_out = tf.nn.dropout(self.input, keep_prob=FLAGS.dropout_deep[0])
        for i in range(deep_layer_num):
            deep_out = tf.add(tf.matmul(deep_out, self.weights['deep_layer_weight_{0}'.format(i)]),
                              self.weights['deep_layer_bias_{0}'.format(i)])
            if FLAGS.deep_layers_activation == 'relu':
                deep_out = tf.nn.relu(deep_out)
            else:
                deep_out = tf.nn.sigmoid(deep_out)
            deep_out = tf.nn.dropout(deep_out, keep_prob=FLAGS.dropout_deep[i + 1])

        self.deep_out = deep_out
        print(self.deep_out)
        self.weights['concat_weight'] = tf.Variable(
            tf.random_normal([FLAGS.deep_layers[-1] + self.total_size, 1], 0.0, 0.01), dtype=tf.float32)
        self.weights['concat_bias'] = tf.Variable(tf.random_normal([1, 1]), dtype=tf.float32)
        self.out = tf.concat([self.cross_network_out, self.deep_out], axis=1)
        self.out = tf.add(tf.matmul(self.out, self.weights['concat_weight']), self.weights['concat_bias'])
        print(self.out)

        # loss
        if FLAGS.loss == 'logloss':
            self.out = tf.nn.sigmoid(self.out)
            self.loss = tf.losses.log_loss(self.label, self.out)
            correct_prediction = tf.equal(tf.to_int32(tf.round(self.out)), self.label)
            self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.acc)

        elif FLAGS.loss == 'mse':
            self.loss = tf.losses.mean_squared_error(labels=self.label, predictions=self.out)

        self.train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(
            self.loss)

    def train(self, sess, X_sparse, X_sparse_index, X_dense, label, index):
        _loss, _step, _acc, _result = sess.run([self.loss, self.train_op, self.acc, merged], feed_dict={
            self.X_sparse: X_sparse,
            self.X_sparse_index: X_sparse_index,
            self.X_dense: X_dense,
            self.label: label
        })
        writer.add_summary(_result, index)  # 将日志数据写入文件

        return _loss, _step, _acc

    def predict(self, sess, X_sparse, X_sparse_index, X_dense):
        result = sess.run([self.out], feed_dict={
            self.X_sparse: X_sparse,
            self.X_sparse_index: X_sparse_index,
            self.X_dense: X_dense,
        })
        return result

    def eval(self, sess, X_sparse, X_sparse_index, X_dense, y, index):
        val_out = self.predict(sess, X_sparse, X_sparse_index, X_dense)
        correct_prediction = np.equal(np.int32(np.round(val_out)), y)
        val_acc = np.mean(np.int32(correct_prediction))
        print('the times of training is %d ,and val acc = %s' % (index, val_acc))
        return val_acc


def get_batch(X_sparse, X_dense, X_sparse_index, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return X_sparse[start:end], X_dense[start:end], X_sparse_index[start:end], y[start:end]


data = load_data()

FLAGS.feature_sizes = data['feat_dim']
FLAGS.field_size = 26
FLAGS.numeric_feature_size = 13

X_dense = np.array(data['xv'])[:, :13]
X_sparse = np.array(data['xv'])[:, 13:]
X_sparse_index = np.array(data['xi'])[:, 13:]
y = np.array(data['y_train'])

X_dense_train = X_dense[2000:, :]
X_dense_val = X_dense[:2000, :]
X_sparse_train = X_sparse[2000:, :]
X_sparse_val = X_sparse[:2000, :]
X_sparse_index_train = X_sparse_index[2000:, :]
X_sparse_index_val = X_sparse_index[:2000, :]
y_train = y[2000:, :]
y_val = y[:2000, :]

with tf.Session() as sess:
    Model = DCN()
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter('logs/', sess.graph)  # 将训练日志写入到logs文件夹下

    cnt = int(len(data['y_train']) / FLAGS.batch_size)
    print('cnt all:%s' % cnt)
    val_acc_list = []
    for i in range(FLAGS.epoch):
        print('epoch %s:' % i)
        for j in range(0, cnt):
            cnt_X_sparse, cnt_X_dense, cnt_X_sparse_index, cnt_y = get_batch(X_sparse_train, X_dense_train,
                                                                             X_sparse_index_train, y_train,
                                                                             FLAGS.batch_size, j)
            loss, step, acc = Model.train(sess, cnt_X_sparse, cnt_X_sparse_index, cnt_X_dense, cnt_y, i * cnt + j)
            # 0.94左右
            if j % 5 == 0:
                _tmp_val = Model.eval(sess, X_sparse_val, X_sparse_index_val, X_dense_val, y_val, j)
                val_acc_list.append(_tmp_val)

    from matplotlib import pyplot as plt

    plt.plot(val_acc_list)
    plt.show()
