import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_act
from torch_geometric.utils import scatter
from functools import partial




@register_act('instance_rms_norm')
class InstanceRMSNorm(nn.Module):
    '''
    Instance RMSnorm for graphs
    '''
    def __init__(self, dim, affine=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)

    def forward(self, x, batch_index):

        rms = scatter(x ** 2, batch_index, dim=0, reduce='mean').sqrt()

        return x / (rms[batch_index] + self.eps) * self.gamma



@register_act('graph_group_norm')
class GraphGroupNorm(nn.Module):
    '''
        Group norm for graphs
        - by default layernorm for graphs (not for token/node)
    '''
    def __init__(self, dim, num_groups=1, affine=True, bias=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.bias = bias
        self.eps = eps
        self.num_groups = num_groups
        assert dim % num_groups == 0, f'dim={dim} shall be dividable by num_groups={num_groups}'

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)


    def forward(self, x, batch_index):
        sizes = [i for i in x.size()]

        x = x.view(*sizes[:-1], self.num_groups, -1)

        mean = scatter(x, batch_index, dim=0, reduce='mean').mean(dim=-1, keepdim=True)
        centered_x = x - mean[batch_index]
        std = scatter(centered_x**2, batch_index, dim=0, reduce='mean').mean(dim=-1, keepdim=True).sqrt()

        return (centered_x / (std[batch_index] + self.eps)).view(*sizes[:-1], -1) * self.gamma + self.beta

register_act('graph_group_norm_8', partial(GraphGroupNorm, num_groups=8))


@register_act('graph_group_rms_norm')
class GraphGroupRMSNorm(nn.Module):
    '''
        Group norm for graphs
        - by default layernorm for graphs (not for token/node)
    '''
    def __init__(self, dim, num_groups=1, affine=True, bias=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.bias = bias
        self.eps = eps
        self.num_groups = num_groups
        assert dim % num_groups == 0, f'dim={dim} shall be dividable by num_groups={num_groups}'

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)


    def forward(self, x, batch_index):
        sizes = [i for i in x.size()]

        x = x.view(*sizes[:-1], self.num_groups, -1)

        rms = scatter(x**2, batch_index, dim=0, reduce='mean').mean(dim=-1, keepdim=True).sqrt()

        return (x / (rms[batch_index] + self.eps)).view(*sizes[:-1], -1) * self.gamma + self.beta


register_act('graph_group_rms_norm_8', partial(GraphGroupRMSNorm, num_groups=8))





@register_act('rms_graph_norm')
class RMSGraphNorm(nn.Module):
    '''
        RMS Graph Norm
        - RMSnorm
        - Computer Vision like LayerNorm
    '''
    def __init__(self, dim, num_groups=1, affine=True, bias=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.bias = bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)


    def forward(self, x, batch_index):
        shapes = [1] * (x.dim()-1) + [-1]
        gamma, beta = self.gamma.view(*shapes), self.beta.view(*shapes)

        rms_inv = torch.rsqrt(scatter((x ** 2).mean(dim=-1, keepdim=True), batch_index, dim=0, reduce='mean') + self.eps)

        return x * rms_inv[batch_index] * gamma + beta
