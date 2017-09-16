import numpy as np
from tqdm import tqdm

from Tensorflow.base.base_train import BaseTrain


class AutoEncoderTrain(BaseTrain):
    def __init__(self, sess, model, data, config, FLAGS):
        super(AutoEncoderTrain, self).__init__(sess, model, data, config, FLAGS)

    def train(self):
        # i init the epoch as a tensor to be saved in the graph so i can restore it and continue traning

        # training
        for cur_epoch in range(self.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            losses = []
            loop = tqdm(self.data.next_batch(), total=self.config.nit_epoch, desc="epoch-" + str(cur_epoch) + "-")
            cur_iter=0
            for batch_x, batch_y in loop:
                feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
                _, loss = self.sess.run([self.model.train_step, self.model.cross_entropy],
                                        feed_dict=feed_dict)

                # increment global step by 1
                self.global_step_assign_op.eval(session=self.sess, feed_dict={
                    self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
                losses.append(loss)
                if cur_iter > self.config.nit_epoch:
                    break
                cur_iter+=1
            loop.close()
            cur_it = self.global_step_tensor.eval(self.sess)
            loss = np.mean(losses)

            summaries_scalers_dict = {}
            summaries_images_dict = {}

            summaries_scalers_dict['loss'] = loss

            # adding test summaries
            batch,_ = next(self.data.next_batch())
            feed_dict = {self.model.x: batch
                , self.model.is_training: False}
            encode_decode = self.sess.run(
                self.model.output_softmax, feed_dict=feed_dict)
            # concatinate ground truth with network output to visualize
            concatenated_image = np.concatenate((encode_decode, batch), axis=2)[..., 0, None]
            summaries_images_dict['test_images'] = concatenated_image
            self.add_scaler_summary(cur_it, summaries_dict=summaries_scalers_dict,
                                    summaries_merged=self.model.summaries, scope='train')
            self.add_image_summary(cur_it, summaries_dict=summaries_images_dict, scope='test')

            print("epoch-" + str(cur_epoch) + "-" + "loss-" + str(loss))
            # increment_epoch
            self.cur_epoch_assign_op.eval(session=self.sess,
                                          feed_dict={self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            # Save the current checkpoint
            self.save()
