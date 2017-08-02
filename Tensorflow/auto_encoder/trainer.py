import numpy as np
import tensorflow as tf
from Tensorflow.base_train import Trainer
from tqdm import tqdm
import matplotlib.pyplot as plt

class AutoEncoderTrainer(Trainer):
    def __init__(self,sess, model, config,FLAGS):
        super(AutoEncoderTrainer,self).__init__(sess,model, config,FLAGS)


    def train(self):
        # i init the epoch as a tensor to be saved in the graph so i can restore it and continue traning

        # training
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            losses=[]
            loop=tqdm(range(self.config.nit_epoch))
            for it in loop:
                batch_x, _ = self.model.data.train.next_batch(self.config.batch_size)

                feed_dict = {self.model.x: batch_x, self.model.is_training: True}
                _,loss=self.sess.run([self.model.train_step,self.model.cross_entropy],
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
            feed_dict = {self.model.x: self.model.data.test.images[:self.config.num_test]
                        , self.model.is_training: False}
            encode_decode = self.sess.run(
                self.model.output, feed_dict=feed_dict)
            #concatinate ground truth with network output
            concatenated_image=(encode_decode.reshape([-1]+ self.config.state_size),
             self.model.data.test.images[:self.config.num_test].reshape([-1]+ self.config.state_size))
            summaries_dict['test_images']=np.concatenate(concatenated_image,axis=2)
            self.add_summary(cur_it, summaries_dict=summaries_dict, summaries_merged=self.model.summaries)

            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))
            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            self.save()
    def test(self):
        feed_dict = {self.model.x: self.data.test.images[:self.config.num_test], self.model.y: self.data.test.labels, self.model.is_training: False}
        encode_decode = self.sess.run(
            self.model.de2, feed_dict=feed_dict)
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(self.config.num_test):
            a[0][i].imshow(np.reshape(self.data.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()
