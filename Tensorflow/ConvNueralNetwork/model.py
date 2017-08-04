# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Tensorflow.layers import conv, max_pool, dense, flatten
from Tensorflow.basic_model import BaseModel


class ConvModel(BaseModel):
    def __init__(self,config):
        super(ConvModel,self).__init__(config)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        with tf.name_scope("model"):

            self.is_training = tf.placeholder(tf.bool)

            self.x = tf.placeholder(tf.float32, shape=[None]+self.config.state_size)
            self.y = tf.placeholder(tf.float32, shape=[None, 10])

            # network_architecture
            h_conv1 = conv(self.x, num_filters=32, kernel_size=[3, 3], stride=[2, 2], name='conv1')
            h_pool1 = max_pool(h_conv1, kernel_size=[2, 2], stride=[2, 2])
            flat = flatten(h_pool1)
            d1 = dense(flat, num_units=512, activation_fn=tf.nn.relu, name="densee2")
            d2 = dense(d1, num_units=10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))








sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
