import tensorflow as tf
from Tensorflow.auto_encoder.model import AutoEncoderModel
from Tensorflow.config import AutoEncoderConfig
from Tensorflow.utils import create_dirs
from Tensorflow.auto_encoder.trainer import AutoEncoderTrainer
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('exp_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', True, """ whether to Load the Model and Continue Training or not """)
tf.app.flags.DEFINE_boolean('train_n_test', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    config = AutoEncoderConfig()
    create_dirs([config.summary_dir])
    model = AutoEncoderModel(config)
    data=input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)

    sess = tf.Session()

    trainer = AutoEncoderTrainer(sess, model,data,config, FLAGS)
    if FLAGS.is_train:
        trainer.train()


if __name__ == '__main__':
    tf.app.run()