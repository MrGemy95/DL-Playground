import tensorflow as tf

from Tensorflow.auto_encoder.data_generator import GenerateData
from Tensorflow.auto_encoder.model import AutoEncoderModel
from Tensorflow.auto_encoder.train import AutoEncoderTrain
from Tensorflow.utils.utils import create_dirs
from config import AutoEncoderConfig

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('exp_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', False, """ whether to Load the Model and Continue Training or not """)
tf.app.flags.DEFINE_boolean('train_n_test', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    config = AutoEncoderConfig()
    create_dirs([config.summary_dir])
    model = AutoEncoderModel(config)
    data=GenerateData(config)

    sess = tf.Session()

    trainer = AutoEncoderTrain(sess, model, data, config, FLAGS)
    if FLAGS.is_train:
        trainer.train()


if __name__ == '__main__':
    tf.app.run()