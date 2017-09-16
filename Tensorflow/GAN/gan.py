





class Gan():
    def main(_):
        config = NeuralConfig()
        create_dirs([config.summary_dir])
        model = NeuralModel(config)
        data = input_data.read_data_sets("../Data/MNIST_data/", one_hot=True)
        sess = tf.Session()

        trainer = NeuralTrain(sess, model, data, FLAGS, config)
        if FLAGS.is_train:
            trainer.train()
        else:
            trainer.test()

    if __name__ == '__main__':
        tf.app.run()