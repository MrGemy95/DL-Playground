# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from TF.layers import conv, max_pool, dense, flatten,lstm
from TF.base_model import BaseModel


class LstmModel(BaseModel):
    def __init__(self,config):
        super(LstmModel, self).__init__(config)
        self.data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 28,28])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network_architecture
        l1 = lstm(self.x,self._config.n_steps,self._config.state_size,name='lstm')
        d1 = tf.layers.dense(l1[-1], 256, name="dense1")
        d2 = dense(d1, num_units=10)

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self._config.adam_lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







