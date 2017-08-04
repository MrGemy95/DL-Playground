# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Tensorflow.layers import conv, max_pool, dense, flatten
from Tensorflow.basic_model import BaseModel


class AutoEncoderModel(BaseModel):
    def __init__(self,config):
        super(AutoEncoderModel,self).__init__(config)
        self.data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        # self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

        # network_architecture
        en1 = tf.layers.dense(self.x,256,activation=tf.nn.relu, kernel_initializer =tf.contrib.layers.xavier_initializer(), name='encoder1')
        en2 = tf.layers.dense(en1,128,activation=tf.nn.relu , kernel_initializer =tf.contrib.layers.xavier_initializer(), name='encoder2')
        de1 = tf.layers.dense(en2,256,activation=tf.nn.relu , kernel_initializer =tf.contrib.layers.xavier_initializer(), name='decoder1')
        self.output = tf.layers.dense(de1,784,activation=tf.nn.relu , kernel_initializer =tf.contrib.layers.xavier_initializer(), name='decoder2')

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x, predictions=self.output))
            self.train_step = tf.train.RMSPropOptimizer(self.config.lr).minimize(self.cross_entropy)



