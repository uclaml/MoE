import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import utils
import torchvision.models as torchmodels
import resnet, mobilenet

from torch.optim.optimizer import Optimizer, required
from torch.optim import _functional

config = utils.get_config()
classes = config["classes"]
expert_num = config["experts"]

class NormalizedGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, maximize=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
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
            maximize = group['maximize']

            per_expert_num = int(len(group['params'])/expert_num)
            per_expert_norm = [0 for i in range(expert_num)]
            for i in range(expert_num):
                for j in range(i*per_expert_num,(i+1)*per_expert_num):
                    p = group['params'][j]
                    if p.grad is not None:
                        per_expert_norm[i] += p.grad.norm()

            for idx, p in enumerate(group['params']):
                if p.grad is not None:
                    # Normalizing
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
                            nesterov=nesterov,
                            maximize=maximize)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


# top 1 hard routing
def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


# hard routing according to probability
def choose1(t):
    index = t.multinomial(num_samples=1)
    values = torch.gather(t, 1, index)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index


def choose2(t):
    index = t.multinomial(num_samples=2)
    values = torch.gather(t, 1, index)
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
    def __init__(self, input_dim, out_dim, strategy='top1', patch_size=4, rtype='conv2d', stride=4): #4,4
        super(Router, self).__init__()
        if rtype == 'conv2d':
            self.conv1 = nn.Conv2d(3, out_dim, patch_size, stride)
        elif rtype == 'linear':
            self.conv1 = nn.Linear(1728, out_dim)
        self.out_dim = out_dim
        self.strategy = strategy
        self.rtype = rtype
        # zero initialization
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight = torch.nn.Parameter(self.conv1.weight * 0)
        self.conv1.bias = torch.nn.Parameter(self.conv1.bias * 0)

    def forward(self, x):
        x = self.conv1(x) 

        if self.rtype == 'conv2d':
            x = torch.sum(x, 2)
            x = torch.sum(x, 2)
        
        elif self.rtype == 'linear':
            x = torch.sum(x, 1)

        if self.training:
            x = x + torch.rand(x.shape[0],self.out_dim).cuda()
        return x


class NonlinearMixtureMobile(nn.Module):
    def __init__(self, expert_num, strategy='top1'):
        super(NonlinearMixtureMobile, self).__init__()
        self.router = Router(3, expert_num, strategy=strategy)
        self.models = nn.ModuleList()
        for i in range(expert_num):
            self.models.append(mobilenet.MobileNetV2()) 
        self.strategy = strategy
        self.expert_num = expert_num

    def forward(self, x):
        select = self.router(x)
        select = F.softmax(select, dim=1)

        # top 1 or choose 1 according to probability
        if self.strategy == 'top1':
            gate, index = top1(select)
        else:
            gate, index = choose1(select)

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

        expert_inputs = torch.einsum('bjkd,ben->ebjkd', x, dispatch_tensor)

        output = []
        for i in range(self.expert_num):
            output.append(self.models[i](expert_inputs[i]))

        output = torch.stack(output)
        output = torch.einsum('ijk,jil->il', combine_tensor, output)
        output = F.softmax(output, dim=1)

        return output, select0, loss, 0


class NonlinearMixtureRes(nn.Module):
    def __init__(self, expert_num, strategy='top1'):
        super(NonlinearMixtureRes, self).__init__()
        self.router = Router(3, expert_num, strategy=strategy)
        self.models = nn.ModuleList()
        for i in range(expert_num):
            self.models.append(resnet.ResNet18()) 
        self.strategy = strategy
        self.expert_num = expert_num

    def forward(self, x):
        select = self.router(x)
        select = F.softmax(select, dim=1)

        # top 1 or choose 1 according to probability 
        if self.strategy == 'top1':
            gate, index = top1(select)
        else:
            gate, index = choose1(select)

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

        expert_inputs = torch.einsum('bjkd,ben->ebjkd', x, dispatch_tensor)

        output = []
        embed = []
        for i in range(self.expert_num):
            output.append(self.models[i](expert_inputs[i])[0])
            embed.append(self.models[i](expert_inputs[i])[1])

        output = torch.stack(output)
        output = torch.einsum('ijk,jil->il', combine_tensor, output)

        embed = torch.stack(embed)
        embed = torch.einsum('ijk,jil->il', combine_tensor, embed)

        output = F.softmax(output, dim=1)

        return output, select0, loss, embed

    def return_select(self, x, soft=False):
        select = self.router(x)

        if self.strategy == 'top1':
            _, index = top1(select)
        else:
            _, index = choose1(select)

        return index


