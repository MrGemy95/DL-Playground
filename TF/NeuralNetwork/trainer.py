import numpy as np
import tensorflow as tf
from TF.base_train import Trainer
from tqdm import tqdm


class NeuralTrainer(Trainer):
    def __init__(self, sess, model, config, FLAGS):
        super(NeuralTrainer, self).__init__(sess, model, config, FLAGS)

    def train(self):
        # i init the epoch as a tensor to be saved in the graph so i can restore it and continue traning

        # training
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            losses = []
            loop = tqdm(range(self.config.nit_epoch))

            for it in loop:
                batch_x, batch_y = self.model.data.train.next_batch(self.config.batch_size)
                feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
                _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                             feed_dict=feed_dict)

                # increment global step by 1
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
                losses.append(loss)
            loop.close()

            loss = np.mean(losses)
            #getting the current global step to add summary
            cur_it = self.global_step_tensor.eval(self.sess)
            summaries_dict = {}
            summaries_dict['loss'] = loss
            summaries_dict['acc'] = acc
            self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)

            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))
            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            self.save()

    def test(self):
        feed_dict = {self.model.x: self.data.test.images, self.model.y: self.data.test.labels,
                     self.model.is_training: False}

        print("Test Acc : ", self.sess.run(self.model.accuracy,
                                           feed_dict={self.x: self.data.test.images, self.y: self.data.test.labels}),
              "% \nExpected to get around 94-98% Acc")
