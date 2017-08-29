import tensorflow as tf
from Tensorflow.ConvNueralNetwork.model import ConvModel
from Tensorflow.config import ConvConfig
from Tensorflow.utils import create_dirs
from Tensorflow.ConvNueralNetwork.train import ConvTrain
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('exp_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    config = ConvConfig()
    create_dirs([config.summary_dir])
    model = ConvModel(config)

    sess = tf.Session()
    data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)

    trainer = ConvTrain(sess, model, data, config, FLAGS)

    if FLAGS.is_train:
        trainer.train()
    else :
        trainer.test()

if __name__ == '__main__':
    tf.app.run()