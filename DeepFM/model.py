import tensorflow as tf
import pandas as pd

data = pd.read_csv('../criteo_data/criteo_small.txt')
print(data.info())

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_data_path", "../criteo_data/criteo_small.txt", "training data dir")

X = tf.placeholder(tf.float32,shape=[None,None])
y = tf.placeholder(tf.int32,shape=[None,1])


X = tf.nn.embedding_lookup(X,)




print(FLAGS)