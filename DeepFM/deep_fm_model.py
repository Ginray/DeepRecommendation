import os
import sys
import numpy as np
import tensorflow as tf

from DeepFM.build_data import load_data


class Args():
    feature_sizes = 1
    field_size = 1
    embedding_size = 8
    deep_layers = [512, 256, 128]
    epoch = 3
    batch_size = 1024

    # 1e-2 1e-3 1e-4
    learning_rate = 1e-3

    # 防止过拟合
    l2_reg_rate = 0.01
    BASE_PATH = os.path.dirname(os.path.dirname(__file__))
    checkpoint_dir = os.path.join(BASE_PATH, 'data/saver2/ckpt')
    is_training = True


class model():
    def __init__(self, args):
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
        self.label = tf.placeholder(tf.float32, shape=[None, None], name='label')

        self.feature_sizes = args.feature_sizes
        self.field_size = args.field_size
        self.embedding_size = args.embedding_size
        self.deep_layers = args.deep_layers
        self.l2_reg_rate = args.l2_reg_rate

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.deep_activation = tf.nn.relu
        self.weight = dict()
        self.checkpoint_dir = args.checkpoint_dir
        self.build_model()

    def build_model(self):
        # One-hot编码后的输入层与Dense embeddings层的权值定义，即DNN的输入embedding。注：Dense embeddings层的神经元个数由field_size和决定
        self.weight['feature_weight'] = tf.Variable(
            tf.random_normal([self.feature_sizes, self.embedding_size], 0.0, 0.01),
            name='feature_weight')

        # FM部分中一次项的权值定义
        # shape (61,1)
        self.weight['feature_first'] = tf.Variable(
            tf.random_normal([self.feature_sizes, 1], 0.0, 1.0),
            name='feature_first')

        # deep网络部分的weight
        num_layer = len(self.deep_layers)
        # deep网络初始输入维度：input_size = 39x256 = 9984 (field_size(原始特征个数)*embedding个神经元)
        input_size = self.field_size * self.embedding_size
        # 标准差
        init_method = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        # shape (9984,512)
        self.weight['layer_0'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(input_size, self.deep_layers[0])), dtype=np.float32
        )
        # shape(1, 512)
        self.weight['bias_0'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(1, self.deep_layers[0])), dtype=np.float32
        )

        # 生成deep network里面每层的weight 和 bias
        if num_layer != 1:
            for i in range(1, num_layer):
                init_method = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))

                # shape  (512,256)  (256,128)
                self.weight['layer_' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                    dtype=np.float32)

                # shape (1,256)  (1,128)
                self.weight['bias_' + str(i)] = tf.Variable(
                    np.random.normal(loc=0, scale=init_method, size=(1, self.deep_layers[i])),
                    dtype=np.float32)

        # deep部分output_size + 一次项output_size + 二次项output_size 423
        last_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size
        init_method = np.sqrt(np.sqrt(2.0 / (last_layer_size + 1)))
        # 生成最后一层的结果
        self.weight['last_layer'] = tf.Variable(
            np.random.normal(loc=0, scale=init_method, size=(last_layer_size, 1)), dtype=np.float32)
        self.weight['last_bias'] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # embedding_part
        # shape (?,?,256)
        self.embedding_index = tf.nn.embedding_lookup(self.weight['feature_weight'],
                                                      self.feat_index)  # Batch*F*K

        # shape (?,39,256)
        self.embedding_part = tf.multiply(self.embedding_index,
                                          tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # [Batch*F*1] * [Batch*F*K] = [Batch*F*K],用到了broadcast的属性
        print('embedding_part:', self.embedding_part)

        """
        网络传递结构
        """
        # FM部分
        # 一阶特征
        # shape (?,39,1)
        self.embedding_first = tf.nn.embedding_lookup(self.weight['feature_first'],
                                                      self.feat_index)  # bacth*F*1
        self.embedding_first = tf.multiply(self.embedding_first, tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # shape （？,39）
        self.first_order = tf.reduce_sum(self.embedding_first, 2)
        print('first_order:', self.first_order)

        # 二阶特征
        self.sum_second_order = tf.reduce_sum(self.embedding_part, 1)
        self.sum_second_order_square = tf.square(self.sum_second_order)
        print('sum_square_second_order:', self.sum_second_order_square)

        self.square_second_order = tf.square(self.embedding_part)
        self.square_second_order_sum = tf.reduce_sum(self.square_second_order, 1)
        print('square_sum_second_order:', self.square_second_order_sum)

        # 1/2*((a+b)^2 - a^2 - b^2)=ab
        self.second_order = 0.5 * tf.subtract(self.sum_second_order_square, self.square_second_order_sum)

        # FM部分的输出(39+256)
        self.fm_part = tf.concat([self.first_order, self.second_order], axis=1)
        print('fm_part:', self.fm_part)

        # DNN部分
        # shape (?,9984)
        self.deep_embedding = tf.reshape(self.embedding_part, [-1, self.field_size * self.embedding_size])
        print('deep_embedding:', self.deep_embedding)

        # 全连接部分
        for i in range(0, len(self.deep_layers)):
            self.deep_embedding = tf.add(tf.matmul(self.deep_embedding, self.weight["layer_%d" % i]),
                                         self.weight["bias_%d" % i])
            self.deep_embedding = self.deep_activation(self.deep_embedding)

        # FM输出与DNN输出拼接
        din_all = tf.concat([self.fm_part, self.deep_embedding], axis=1)
        self.out = tf.add(tf.matmul(din_all, self.weight['last_layer']), self.weight['last_bias'])
        print('output:', self.out)

        # loss部分
        # self.out = tf.nn.sigmoid(self.out)
        self.out = tf.nn.sigmoid(self.out)
        # self.loss = -tf.reduce_mean(
        #     self.label * tf.log(self.out + 1e-24) + (1 - self.label) * tf.log(1 - self.out + 1e-24))

        correct_prediction = tf.equal(tf.round(self.out), self.label)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.acc = tf.Print(self.acc, [self.acc])
        tf.summary.scalar('acc', self.acc)
        self.loss = -1 * tf.reduce_mean(
            self.label * tf.log(tf.clip_by_value(self.out, 1e-10, 1.0 - (1e-10))) + (1 - self.label) * tf.log(
                (1 - tf.clip_by_value(self.out, 1e-10, 1.0 - (1e-10)))))
        tf.summary.scalar('loss', self.loss)

        # 正则：sum(w^2)/2*l2_reg_rate
        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["last_layer"])
        for i in range(len(self.deep_layers)):
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg_rate)(self.weight["layer_%d" % i])

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        grads, variables = zip(*opt.compute_gradients(self.loss))
        grads, global_norm = tf.clip_by_global_norm(grads, 5)
        train_op = opt.apply_gradients(zip(grads, variables))

        self.train_op = train_op

    def train(self, sess, feat_index, feat_value, label, index):
        loss, step, acc, result = sess.run([self.loss, self.train_op, self.acc, merged], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label
        })
        writer.add_summary(result, index)  # 将日志数据写入文件

        return loss, step, acc

    def predict(self, sess, feat_index, feat_value):
        result = sess.run([self.out], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value
        })
        return result

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], np.array(y[start:end])


if __name__ == '__main__':
    args = Args()

    data = load_data()

    args.feature_sizes = data['feat_dim']
    args.field_size = len(data['xi'][0])
    args.is_training = True

    with tf.Session() as sess:
        Model = model(args)
        # init variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
        writer = tf.summary.FileWriter('logs/', sess.graph)  # 将训练日志写入到logs文件夹下

        cnt = int(len(data['y_train']) / args.batch_size)
        print('time all:%s' % cnt)
        sys.stdout.flush()
        if args.is_training:
            for i in range(args.epoch):
                print('epoch %s:' % i)
                for j in range(0, cnt):
                    X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                    loss, step, acc = Model.train(sess, X_index, X_value, y, i * cnt + j)
                    if j % 10 == 0:
                        print('the times of training is %d, and the loss is %s ,and acc = %s' % (j, loss, acc))
                        Model.save(sess, args.checkpoint_dir)
                        # 准确率最高到0.940429688
        else:
            Model.restore(sess, args.checkpoint_dir)
            for j in range(0, cnt):
                X_index, X_value, y = get_batch(data['xi'], data['xv'], data['y_train'], args.batch_size, j)
                result = Model.predict(sess, X_index, X_value)
                print(result)
