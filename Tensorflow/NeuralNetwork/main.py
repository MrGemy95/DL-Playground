import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from Tensorflow.NeuralNetwork.model import NeuralModel
from Tensorflow.NeuralNetwork.train import NeuralTrain
from Tensorflow.utils.utils import create_dirs
from config import NeuralConfig

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('exp_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', False , """ whether to Load the Model and Continue Training or not """)


def main(_):
    config = NeuralConfig()
    create_dirs([config.summary_dir])
    model = NeuralModel(config)
    data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
    sess = tf.Session()

    trainer = NeuralTrain(sess, model, data, FLAGS, config)
    if FLAGS.is_train:
        trainer.train()
    else :
        trainer.test()

if __name__ == '__main__':
    tf.app.run()