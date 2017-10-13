import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from logger import Logger

class Trainer():
    def __init__(self, config, net, trainloader,testloader):
        self.config = config
        self.trainloader = trainloader
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(net.parameters(), lr=0.00001)
        self.logger=Logger("./summ")
    def train(self):
        cnt=0
        for epoch in range(self.config.n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs
                cnt+=1
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.logger.scalar_summary("loss", loss.data[0], i)
                # print statistics
                running_loss += loss.data[0]
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    self.logger.scalar_summary("loss",running_loss,i)
                    running_loss = 0.0

        print('Finished Training')
