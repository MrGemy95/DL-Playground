import tensorflow as tf
from TF.NeuralNetwork.model import NeuralModel
from TF.config import NeuralConfig
from TF.utils import create_dirs
from TF.ConvNueralNetwork.trainer import ConvTrainer

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('exp_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', True, """ whether to Load the Model and Continue Training or not """)
tf.app.flags.DEFINE_boolean('train_n_test', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    config = NeuralConfig()
    create_dirs([config.summary_dir])
    model = NeuralModel(config)

    sess = tf.Session()

    trainer = ConvTrainer(sess, model,config, FLAGS)

    trainer.train()


if __name__ == '__main__':
    tf.app.run()