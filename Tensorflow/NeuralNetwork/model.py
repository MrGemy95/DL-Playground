# From tensorflow official tutorial
import tensorflow as tf

from Tensorflow.base.base_model import BaseModel


class NeuralModel(BaseModel):
    def __init__(self,config):
        super(NeuralModel,self).__init__(config)
        # self.data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None]+self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])
        d1 = tf.layers.dense(self.x, 256, name="dense1")
        d2 = tf.layers.dense(d1, 512,activation=tf.nn.relu, name="dense2")
        d3 = tf.layers.dense(d2, 256,activation=tf.nn.relu, name="dense3")
        d4 = tf.layers.dense(d3, 128,activation=tf.nn.relu, name="dense4")
        d5 = tf.layers.dense(d4, 10, name="dense5")


        # loss
        with tf.name_scope("loss"):

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d5))
            self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.cross_entropy)
            correct_prediction = tf.equal(tf.argmax(d5, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))







