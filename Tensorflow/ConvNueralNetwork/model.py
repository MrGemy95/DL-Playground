# From tensorflow official tutorial
import tensorflow as tf

from Tensorflow.base.base_model import BaseModel
from Tensorflow.layers import conv, flatten


class ConvModel(BaseModel):
    def __init__(self,config):
        super(ConvModel,self).__init__(config)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth

        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None]+self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network_architecture
        h_conv1 = conv(self.x, num_filters=32, kernel_size=[3, 3], stride=[2, 2], name='conv1')
        flat = flatten(h_conv1)
        d1 = tf.layers.dense(flat, 512, activation_fn=tf.nn.relu, name="densee2")
        d2 = tf.layers.dense(d1, 10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))








sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
