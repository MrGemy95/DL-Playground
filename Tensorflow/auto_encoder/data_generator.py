import numpy as np
import cv2
import matplotlib.pyplot as plt


class GenerateData:
    def __init__(self, config):
        """
        it just take the config file which contain all paths the generator needs
        :param config: configuration
        """

        self.config = config
        np.random.seed(2)
        x = np.load(config.states_path)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        self.y = x.copy()

        self.x = self.prepare_states(x)
        self.y = self.prepare_labels(self.y)

        # shuffles
        self.x = self.x[idx]
        self.y = self.y[idx]

        train_idx=int(self.config.train_ratio*self.x.shape[0])
        self.x = self.x[:train_idx]
        self.y = self.y[:train_idx]
        self.xtest = self.x[train_idx:]
        self.ytest = self.y[train_idx:]

    def next_batch(self):
        """
        :return: a tuple of all batches

        """
        while True:
            idx = np.random.choice(self.x.shape[0], self.config.batch_size)
            batch_x = self.x[idx]
            batch_y = self.y[idx]


            yield batch_x, batch_y

    def sample(self):
        idx = np.random.choice(self.config.num_episodes_train, self.config.batch_size)
        return self.x[idx], self.actions[idx]


    def prepare_states(self, x):

        new_x = np.zeros((x.shape[0], x.shape[1], 96, 96, 1))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                retval2, threshold = cv2.threshold(x[i, j, :, :, 0].astype('uint8'), 89, 255, cv2.THRESH_BINARY)
                threshold = threshold.astype('uint8') // 255
                new_x[i, j, :, :, 0] = cv2.resize(threshold, (96, 96))
        new_x[:,:,:15,:,:]=0

        new_x=new_x.reshape((-1,96,96,1))

        #creating 2 channels
        new_x = (np.arange(2) == new_x).astype(int)

        return new_x


    def prepare_labels(self, x):

        new_x = np.zeros((x.shape[0], x.shape[1], 96, 96, 1))

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                retval2, threshold = cv2.threshold(x[i, j, :, :, 0].astype('uint8'), 89, 255, cv2.THRESH_BINARY)
                threshold = threshold.astype('uint8') // 255
                new_x[i, j, :, :, 0] = cv2.resize(threshold, (96, 96))
        new_x[:,:,:15,:,:]=0
        new_x=new_x.reshape((-1,96,96))


        return new_x



