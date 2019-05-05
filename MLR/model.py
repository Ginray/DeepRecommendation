import tensorflow as tf
from sklearn.metrics import accuracy_score
from data.get_data_criteo import get_data

X_train, X_test, y_train, y_test = get_data()

X = tf.placeholder(tf.float32, [None, None], name='X_data')
y = tf.placeholder(tf.float32, [None, ], name='y_data')

m = 2
learning_rate = 0.01

u = tf.Variable(tf.random_normal(shape=[39, m], mean=0, stddev=0.5), name='v')
w = tf.Variable(tf.random_normal(shape=[39, m], mean=0, stddev=0.5), name='w')

U = tf.matmul(X, u)
p1 = tf.nn.softmax(U)

W = tf.matmul(X, w)
p2 = tf.nn.sigmoid(W)

pred = tf.reduce_sum(tf.multiply(p1, p2), axis=1)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=pred, logits=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for each in range(1000):
        _, now_cost, out = sess.run([train_op, cost, pred], feed_dict={X: X_train, y: y_train})
        acc = accuracy_score(y_train.values, out.round())
        print('train epoch = {0}, acc = {1}'.format(each, acc))
        if (each % 10 == 0):
            out = sess.run(pred, feed_dict={X: X_test})
            acc = accuracy_score(y_test, out.round())
            print('------test epoch , acc = {0}'.format(acc))
