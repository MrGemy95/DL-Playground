# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from TF.layers import conv, max_pool, dense, flatten
from TF.base_model import BaseModel


class NeuralModel(BaseModel):
    def __init__(self,config):
        super(NeuralModel,self).__init__(config)
        self.data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        d1 = dense(self.x, num_units=512, name="dense2")
        d2 = dense(d1, num_units=512, name="dense2")
        d3 = dense(d2, num_units=256, name="dense3")
        d4 = dense(d3, num_units=128, name="dense4")
        d5 = dense(d4, num_units=10, name="dense5")

        # loss
        with tf.name_scope("loss"):

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d5))
            self.train_step = tf.train.AdamOptimizer(self._config.adam_lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d5, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







