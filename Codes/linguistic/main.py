import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils import AvgMeter, get_lr

import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.optim.optimizer import Optimizer, required
from torch.optim import _functional

expert_num = 4

class NormalizedGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NormalizedGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NormalizedGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            per_expert_num = int(len(group['params'])/expert_num)
            per_expert_norm = [0 for i in range(expert_num)]
            for i in range(expert_num):
                for j in range(i*per_expert_num,(i+1)*per_expert_num):
                    p = group['params'][j]
                    if p.grad is not None:
                        per_expert_norm[i] += p.grad.norm()
            # expert_num 
            # print(per_expert_norm)
            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    # Normalizing
                    # if p.grad.norm() != 0:
                    if per_expert_norm[idx // per_expert_num] != 0:
                        p.grad /= per_expert_norm[idx // per_expert_num]

                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            _functional.sgd(params_with_grad,
                            d_p_list,
                            momentum_buffer_list,
                            weight_decay=weight_decay,
                            momentum=momentum,
                            lr=lr,
                            dampening=dampening,
                            nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        # print(per_expert_norm)
        return loss


class MultilingualDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {}
        item['text'] = self.texts[idx,:]
        item['label'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class NonlinearClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NonlinearClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim) 
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight * 0.001)
        self.fc1.bias = torch.nn.Parameter(self.fc1.bias * 0.001)

    def forward(self, x):
        output = self.fc1(x) ** 3
        return output


# top 1 hard routing
def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]


def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]


class Router(nn.Module):
    def __init__(self, input_dim, out_dim): 
        super(Router, self).__init__()
        self.linear = nn.Conv1d(1, out_dim, 32, 32, bias=False) #nn.Linear(input_dim, out_dim)
        self.out_dim = out_dim
        # zero initialization
        self.reset_parameters()

    def reset_parameters(self):
        self.linear.weight = torch.nn.Parameter(self.linear.weight * 0)

    def forward(self, x):
        x = self.linear(x.unsqueeze(1)) 
        x = torch.sum(x,2)
        if self.training:
            x = x + torch.rand(x.shape[0],self.out_dim).cuda()
        return x


class NonlinearMixture(nn.Module):
    def __init__(self, input_dim, out_dim, expert_num):
        super(NonlinearMixture, self).__init__()
        self.router = Router(input_dim, expert_num)
        self.models = nn.ModuleList()
        for i in range(expert_num):
            self.models.append(NonlinearClassifier(input_dim, out_dim)) 
        self.expert_num = expert_num

    def forward(self, x):

        select = self.router(x) 
        select = F.softmax(select, dim=1)

        gate, index = top1(select)

        mask = F.one_hot(index, self.expert_num).float()

        density = mask.mean(dim=-2)
        density_proxy = select.mean(dim=-2)
        loss = (density_proxy * density).mean() * float(self.expert_num ** 2)

        mask_count = mask.sum(dim=-2, keepdim=True)
        mask_flat = mask.sum(dim=-1)

        combine_tensor = (gate[..., None, None] * mask_flat[..., None, None]
                          * F.one_hot(index, self.expert_num)[..., None])
                          
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)
        select0 = dispatch_tensor.squeeze(-1)

        expert_inputs = torch.einsum('bd,ben->ebd', x, dispatch_tensor)

        output = []
        embed = []
        for i in range(self.expert_num):
            output.append(self.models[i](expert_inputs[i]))

        output = torch.stack(output)
        output = torch.einsum('ijk,jil->il', combine_tensor, output)

        output = F.softmax(output, dim=1)

        return output, select0, loss


def train_epoch(model, train_loader, optimizers, lr_scheduler, criterion):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        
        batch_cuda = batch
        batch_cuda['text'] = batch['text'].cuda()
        batch_cuda['label'] = batch['label'].cuda()
        batch = batch_cuda

        output, select, loss = model(batch['text'])
        loss = criterion(output, batch['label']) + 0.1*loss

        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()
        lr_scheduler.step()

        count = batch["text"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader, criterion):
    loss_meter = AvgMeter()
    total, correct = 0, 0
    select_total = torch.tensor([0.0,0.0,0.0,0.0]).cuda()
    select_part = torch.tensor([0.0,0.0,0.0,0.0]).cuda()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    i=0
    for batch in tqdm_object:

        batch_cuda = batch
        batch_cuda['text'] = batch['text'].cuda()
        batch_cuda['label'] = batch['label'].cuda()
        batch = batch_cuda

        output, select, loss = model(batch['text'])
        select_total += torch.sum(select,dim=0)
        select_part += torch.sum(select,dim=0)

        total += len(batch['label'])
        correct += (torch.argmax(output,dim=1) == batch['label']).sum().item()

        loss = criterion(output, batch['label']) 

        count = batch["text"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        i += 1

        if i % 80 == 0:
            print(f'Dispatch {select_part.cpu()}')
            select_part = torch.tensor([0.0,0.0,0.0,0.0]).cuda()

    print(f'Dispatch {select_total.cpu()}')
    print(f'Accuracy {correct/total}')
    return loss_meter


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_emb = torch.load('train_emb.pt')
test_emb = torch.load('test_emb.pt')

train_target = list(train['target'])
test_target = list(test['target'])

train_set = MultilingualDataset(train_emb, train_target)
test_set = MultilingualDataset(test_emb, test_target)

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=512,
    num_workers=2,
    shuffle=True,
)
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=500,
    num_workers=2,
    shuffle=False,
)

model = NonlinearMixture(input_dim=768, out_dim=2, expert_num=4).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = NormalizedGD(model.models.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(model.router.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

best_loss = float('inf')

for epoch in range(300):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, trainloader, [optimizer, optimizer2], lr_scheduler, criterion)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, testloader, criterion)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), f"best_moe_1.pt")
            print("Saved Best Model!")