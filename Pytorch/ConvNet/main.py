import argparse
from config import NeuralConfig
from Pytorch.ConvNet.model import Net
from Pytorch.ConvNet.train import Trainer
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('exp_dir',  nargs='*',type=str,  default= "/tmp/multilayer-perceptron/",
                    help=""" Experiment dir to store ckpt & summaries """)
parser.add_argument('is_train', type=bool,nargs='*',  default=True,
                    help=""" Whether it is a training or testing""")

FLAGS = parser.parse_args()


def create_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../../Data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../../Data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader,testloader
def main():
    config = NeuralConfig()
    net = Net().cuda()
    trainloader, testloader=create_data()

    trainer = Trainer(config, net, trainloader,testloader)
    if FLAGS.is_train:
        trainer.train()
    else :
        trainer.test()

if __name__ == '__main__':
    main()