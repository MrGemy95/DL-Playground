# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Tensorflow.layers import conv, max_pool, deconv, flatten
from Tensorflow.base_model import BaseModel


class AutoEncoderModel(BaseModel):
    def __init__(self,config):
        super(AutoEncoderModel,self).__init__(config)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth

        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        # self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

        # network_architecture
        en1=conv(self.x, num_filters=16, kernel_size=[3, 3], stride=[2, 2], name='conv1')
        en2=conv(en1, num_filters=16, kernel_size=[3, 3], stride=[2, 2], name='conv2')
        de1=deconv(en2, [32,14,14,1], kernel_size=[3, 3], stride=(2, 2), name="deconv1", activation=tf.nn.relu)
        self.output=deconv(de1,  [32,28,28,1], kernel_size=[3, 3], stride=(2, 2), name="deconv2", activation=tf.nn.relu)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x, predictions=self.output))
            self.train_step = tf.train.RMSPropOptimizer(self.config.lr).minimize(self.cross_entropy)



