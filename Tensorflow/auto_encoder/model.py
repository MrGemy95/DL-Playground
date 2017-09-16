# From tensorflow official tutorial
import tensorflow as tf

from Tensorflow.base.base_model import BaseModel
from Tensorflow.layers import conv, deconv


class AutoEncoderModel(BaseModel):
    def __init__(self,config):
        super(AutoEncoderModel,self).__init__(config)
        self.build_model()
        self.summaries=None
    def build_model(self):
        # define the placeholder to pass the data and the ground_truth

        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None]+self.config.state_size)
        self.y = tf.placeholder(tf.int32, shape=[None]+self.config.labels_size)

        with tf.name_scope('encoder_1'):
            h1 = tf.layers.conv2d(self.x, 64, kernel_size=(8, 8), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn1 = tf.layers.batch_normalization(h1, training=self.is_training)
            drp1 = tf.layers.dropout(tf.nn.relu(bn1), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('encoder_2'):
            h2 = tf.layers.conv2d(drp1, 32, kernel_size=(6, 6), strides=(2, 2),
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn2 = tf.layers.batch_normalization(h2, training=self.is_training)
            drp2 = tf.layers.dropout(tf.nn.relu(bn2), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')


        with tf.name_scope('decoder_1'):
            h6 = tf.layers.conv2d_transpose(drp2, 32, kernel_size=(4, 4), strides=(2, 2), padding='SAME',
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn6 = tf.layers.batch_normalization(h6, training=self.is_training)
            drp6 = tf.layers.dropout(tf.nn.relu(bn6), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')

        with tf.name_scope('decoder_2'):
            h7 = tf.layers.conv2d_transpose(drp6, 32, kernel_size=(6, 6), strides=(2, 2),
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            bn7 = tf.layers.batch_normalization(h7, training=self.is_training)
            drp7 = tf.layers.dropout(tf.nn.relu(bn7), rate=self.config.dropout_rate, training=self.is_training,
                                     name='dropout')



        with tf.name_scope('decoder_5'):
            self.output = tf.layers.conv2d(drp7, 2, kernel_size=(3, 3), strides=(1, 1),
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='SAME')
            self.output_softmax = tf.nn.softmax(self.output)

        # # network_architecture
        # en1=conv(self.x, num_filters=16, kernel_size=[3, 3], stride=[2, 2], name='conv1')
        # en2=conv(en1, num_filters=16, kernel_size=[3, 3], stride=[2, 2], name='conv2')
        # de1=tf.layers.conv2d_transpose(en2, 32, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        # self.output=tf.layers.conv2d_transpose(de1, 2, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
        #                            kernel_initializer=tf.contrib.layers.xavier_initializer())
        #
        # self.output_softmax = tf.nn.softmax(self.output)
        #
        #

        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.output)
            self.train_step = tf.train.RMSPropOptimizer(self.config.lr).minimize(self.cross_entropy)



