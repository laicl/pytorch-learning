
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import visdom

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--wn', type=str, default="window name", metavar='WN',
                    help='name this processing')
args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

##the output of torchvision is PILImage, the values is [0,1]
##we use transform to normalize the PILImage to [-1,1]
transforms = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_dataset = datasets.CIFAR10('../data', train=True, download=False, transform=transforms)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, 
                                           shuffle=True, **kwargs)

test_dataset = datasets.CIFAR10('../data', train=False, transform=transforms)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, 
                                          shuffle=True, **kwargs)

def norm_op(num):
    return nn.BatchNorm2d(num)

def act_op():
    return nn.ReLU()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con_layer1 = nn.Sequential(
           nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
           norm_op(32),
           act_op(),  
           nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
           norm_op(32),
           act_op(),
           nn.MaxPool2d(2)
        ) #(32,32) -> (16,16)
        self.con_layer2 = nn.Sequential(
           nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
           norm_op(64),
           act_op(),
           nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
           norm_op(64),
           act_op(),
           nn.MaxPool2d(2)
        ) #(16,16) -> (8,8)
        self.con_layer3 = nn.Sequential(
           nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
           norm_op(128),
           act_op(),
           nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
           norm_op(128),
           act_op(),
           nn.MaxPool2d(2)
        ) #(8,8) -> (4,4)
        self.fc_layer = nn.Sequential(
           nn.Linear(4*4*128, 500),
           norm_op(500),
           act_op(),
           nn.Linear(500,10)
        )

    def forward(self, x):
        x = self.con_layer1(x)
        x = self.con_layer2(x)
        x = self.con_layer3(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layer(x)
        return x #F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

loss_f = nn.CrossEntropyLoss() #F.nll_loss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
vis = visdom.Visdom()

line = vis.line(Y=np.array([0]),win="train_vgg_"+args.wn)
def train(epoch):
    cur_train_len = (epoch-1)*len(train_loader)
    model.train()
    loss_sum = 0.
    i = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
		
        loss = loss_f(output, target)
        loss_sum += loss
        i += 1
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            print('batch_idx:{}, i: {}, ave_loss:{}'.format(batch_idx,i,loss_sum/i))
            vis.line(X=np.array([cur_train_len+batch_idx]), 
			         Y=np.array([loss.data[0]]), 
			         win=line, 
                     opts=dict(legend=["tr_loss"],title=args.wn),
                     update="append")


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += loss_f(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
