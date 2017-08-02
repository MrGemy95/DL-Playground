# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Tensorflow.layers import conv, max_pool, dense, flatten
from Tensorflow.base_model import BaseModel


class ConvModel(BaseModel):
    def __init__(self,config):
        super(ConvModel,self).__init__(config)
        self.data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

        # network_architecture
        h_conv1 = conv(self.x_reshaped, num_filters=32, kernel_size=[3, 3], stride=[2, 2], name='conv1')
        h_pool1 = max_pool(h_conv1, kernel_size=[2, 2], stride=[2, 2])
        flat = flatten(h_pool1)
        d1 = dense(flat, num_units=512, activation_fn=tf.nn.relu, name="densee2")
        d2 = dense(d1, num_units=10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self._config.lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))








sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
