import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as torchmodels

import os
import argparse

from utils import progress_bar
from utils import get_config
import moe
import resnet, mobilenet
from PIL import Image

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
plt.rcParams.update({'font.size': 18, 'figure.figsize': (8,8), 'axes.axisbelow':True})

torch.cuda.set_device(3)

config = get_config()
EXPERT_NUM = config['experts']
CLUSTER_NUM = config['clusters']
strategy = config['strategy']

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_rotate_train = transforms.Compose([
    torchvision.transforms.RandomRotation((30,30)),
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=(3,7), sigma=(1.1,2.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_rotate_test = transforms.Compose([
    torchvision.transforms.RandomRotation((30,30)),
    transforms.CenterCrop(24),
    transforms.Resize(size=32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Create trainset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

# Create cluster and targets
trainset.targets = torch.tensor(trainset.targets)
trainset.cluster = trainset.targets
trainset.targets = torch.zeros_like(trainset.targets)

# trainset negative examples
trainset_flip = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_rotate_train)
# Cluster and targets
trainset_flip.targets = torch.tensor(trainset_flip.targets)
trainset_flip.cluster = trainset_flip.targets
trainset_flip.targets = torch.ones_like(trainset_flip.targets)

trainset = torch.utils.data.ConcatDataset([trainset,trainset_flip])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                         shuffle=True, num_workers=2)

# Testset cluster and targets
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testset.targets = torch.tensor(testset.targets)
testset.cluster = testset.targets
testset.targets = torch.zeros_like(testset.targets)

# Testset negative
testset_flip = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_rotate_test)
testset_flip.targets = torch.tensor(testset_flip.targets)
testset_flip.cluster = testset_flip.targets
testset_flip.targets = torch.ones_like(testset_flip.targets)

testset = torch.utils.data.ConcatDataset([testset,testset_flip])
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)

# Model
print('==> Building model..')
net = moe.NonlinearMixtureRes(EXPERT_NUM, strategy=strategy).cuda()

checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
criterion = nn.CrossEntropyLoss()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    embeddings = []
    classes = []
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, clusters) in enumerate(testloader):
            classes += clusters
            labels += targets
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _, embeds = net(inputs)  
            embeddings.append(embeds)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return torch.cat(embeddings,dim=0), np.array(classes), np.array(labels)

embeddings, classes, labels = test(0)
print(embeddings.shape)
print(classes.shape)

tsne = TSNE(n_components=2)
visual_data = tsne.fit_transform(embeddings.cpu())
visual_data = visual_data/np.max(np.abs(visual_data))
print(visual_data.shape)

class_6_neg = [a and b for a, b in zip(classes==6,labels==1)]
class_8_neg = [a and b for a, b in zip(classes==8,labels==1)]

plt.scatter(visual_data[class_6_neg][:,0],visual_data[class_6_neg][:,1],marker='.',color='#d7191c')
plt.scatter(visual_data[class_8_neg][:,0],visual_data[class_8_neg][:,1],marker='.',color='#2b83ba')

plt.savefig('cifar10_rotate_fix_class.pdf')
plt.savefig('cifar10_rotate_fix_class.png')
plt.clf()

plt.scatter(visual_data[labels==1][:,0],visual_data[labels==1][:,1],marker='.')
plt.savefig('cifar10_rotate_fix.pdf')
plt.savefig('cifar10_rotate_fix.png')

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')