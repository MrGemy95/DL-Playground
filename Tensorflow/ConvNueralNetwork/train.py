import numpy as np
import tensorflow as tf
from Tensorflow.base_train import BaseTrain
from tqdm import tqdm

class ConvTrain(BaseTrain):
    def __init__(self,sess, model, data,config,FLAGS):
        super(ConvTrain, self).__init__(sess, model, data, config, FLAGS)


    def train(self):
        # i init the epoch as a tensor to be saved in the graph so i can restore it and continue traning

        # training
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            losses=[]
            loop=tqdm(range(self.config.nit_epoch))
            for it in loop:
                batch_x, batch_y = self.data.train.next_batch(self.config.batch_size)
                batch_x=batch_x.reshape(([-1]+self.config.state_size))  #reshape to [-1,28,28]
                feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
                _,loss,acc=self.sess.run([self.model.train_step,self.model.cross_entropy,self.model.accuracy],
                                     feed_dict=feed_dict)

                #increment global step by 1
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
                losses.append(loss)
            loop.close()
            cur_it = self.global_step_tensor.eval(self.sess)
            loss = np.mean(losses)

            summaries_dict = {}
            summaries_dict['loss'] = loss
            summaries_dict['acc'] = acc
            self.add_image_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries,scope='train')

            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))
            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            self.save()
    def test(self):
        feed_dict = {self.model.x: self.data.test.images, self.model.y: self.data.test.labels, self.model.is_training: False}

        print("Test Acc : ", self.sess.run(self.model.accuracy, feed_dict=feed_dict),
              "% \nExpected to get around 94-98% Acc")