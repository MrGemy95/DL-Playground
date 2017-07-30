import numpy as np
import tensorflow as tf
import TF.basic_train.Trainer as Trainer


class ConvTrainer(Trainer):
    def __init__(self, model, config,sess):
        super(ConvTrainer).init(model, config,sess)


    def train(self):

        # training
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self._config.n_epochs + 1, 1):

            for it in range(self._config.nit_epoch):
                batch_x, batch_y = self.model.data.train.next_batch(self._config.batch_size)
                _,loss=self.sess.run([self.model.train_step,self.model.cross_entropy],
                                     feed_dict={self.model.x: batch_x, self.model.y: batch_y})

        self.global_step_assign_op.eval(session=self.sess, feed_dict={
            self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})

    def test(self):
        print("Test Acc : ", self.sess.run(self.model.accuracy, feed_dict={self.x: self.model.test.images, self.y: self.data.test.labels}),
              "% \nExpected to get around 94-98% Acc")