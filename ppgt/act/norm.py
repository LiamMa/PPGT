


import torch
from torch import nn

from ..layer.utils.scatter import scatter

from torch import Tensor
from torch_geometric.typing import OptTensor
from typing import Optional

'''
    Cache Normalization layer as special activation
'''

from torch_geometric.graphgym.register import register_act

from functools import partial


from .norms import *


register_act('batch_norm', nn.BatchNorm1d)
register_act('layer_norm', nn.LayerNorm)
register_act('batch_norm_2d', nn.BatchNorm2d)



@register_act('rms_layer_norm')
class RMSLayerNorm(nn.Module):
    def __init__(self, dim=64, affine=True, bias=True, eps=1e-6):
        super().__init__()

        self.affine = affine
        self.bias = bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)

    def forward(self, x):
        # as verified at torch=2.0: torch.sqrt() now support zero-vectors on forward and backward now
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps
        return x/ std * self.gamma + self.beta



class MaxScaleNorm(nn.Module):
    '''
        To scale the max to 1
    '''
    def __init__(self, dim=64):
        super().__init__()

        self.register_buffer('max_values', torch.ones(1, dim) + 1e-8)


    def forward(self, x):
        if self.training:
            max_values = torch.max(torch.abs(x), dim=0, keepdim=True).values
            self.max_values = torch.where(self.max_values >= max_values, self.max_values, max_values)


        return x / self.max_values



@register_act('rms_kernel_norm')
class RMSKernelNorm(nn.Module):
    def __init__(self, dim=64, affine=True, bias=True, eps=1e-6, stat_per_channel=True):
        super().__init__()
        '''
            dist_recover: + mu and times std to enable distribution recover
        '''

        self.affine = affine
        self.bias = bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)

        self.stat_per_channel = stat_per_channel # compute statu

    def forward(self, x, index):
        # as verified at torch=2.0: torch.sqrt() now support zero-vectors on forward and backward now
        if self.stat_per_channel:
            scale = torch.sqrt(scatter(x**2, index, dim=0, reduce='mean'))
        else:
            scale = torch.sqrt(scatter(torch.mean(x**2, dim=-1, keepdim=True),
                                       index, dim=0, reduce='mean'))

        return x / (scale[index] + self.eps) * self.gamma + self.beta




@register_act('L2_kernel_norm')
class L2KernelNorm(nn.Module):
    def __init__(self, dim=64, affine=True, bias=True, eps=1e-6):
        super().__init__()
        '''
            dist_recover: + mu and times std to enable distribution recover
        '''

        self.affine = affine
        self.bias = bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)


    def forward(self, x, index):
        # as verified at torch=2.0: torch.sqrt() now support zero-vectors on forward and backward now
        scale = torch.sqrt(scatter(x**2, index, dim=0, reduce='sum'))

        return x / (scale[index] + self.eps) * self.gamma + self.beta


register_act('L2_graph_norm', L2KernelNorm)





class PartialObserveRMSKernelNorm(nn.Module):
    '''
        Assume the kernel is partial observed --> the other elements are all-zeros
    '''
    def __init__(self, dim=64, affine=True, bias=True, eps=1e-6, stat_per_channel=False):
        super().__init__()

        self.affine = affine
        self.bias = bias
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, dim), requires_grad=self.affine)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=self.bias)

        self.stat_per_channel = stat_per_channel # compute statu

    def forward(self, x, index, kernel_size):
        # as verified at torch=2.0: torch.sqrt() now support zero-vectors on forward and backward now
        if self.stat_per_channel:
            scale = torch.sqrt(scatter(x**2, index, dim=0, reduce='sum') / kernel_size.view(-1, 1))
        else:
            scale = torch.sqrt(scatter(torch.mean(x**2, dim=-1, keepdim=True),
                                       index, dim=0, reduce='sum') / kernel_size.view(-1, 1))

        return x / (scale[index] + self.eps) * self.gamma + self.beta




class GraphGroupNorm(nn.Module):
    def __init__(self, num_channels: int, num_groups:int=1, eps: float = 1e-5):
        super().__init__()

        self.num_groups =  num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)



    def forward(self, x: Tensor, batch: OptTensor=None,
                batch_size: Optional[int]=None) -> Tensor:

        size = [i for i in x.size()]
        size = size[:-1]
        w_size = [1] * size.__len__()

        x = x.view(*size, self.num_groups, self.num_channels // self.num_groups)
        if batch is None: # degenerate to group-nomr per node like LayerNorm
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True) + self.eps

            out = ((x - mean) / std).view(*size, self.num_channels)

        else:
            mean = scatter(x.mean(dim=-1, keepdim=True),
                           batch,
                           dim=0,
                           dim_size=batch_size,
                           reduce='mean'
                           )[batch]

            out = x - mean
            std = scatter((out ** 2).mean(dim=-1, keepdim=True),
                          batch,
                          dim=0,
                          dim_size=batch_size,
                          reduce='mean'
                          ).sqrt()[batch]

            out = (out / (std + self.eps)).view(*size, self.num_channels)

        return self.weight.view(*w_size, -1) * out + self.bias.view(*w_size, -1)


register_act('rms_norm_eps0', nn.RMSNorm)
register_act('rms_norm', partial(nn.RMSNorm, eps=1e-6)) # following llama
register_act('rms_norm_eps1', partial(nn.RMSNorm, eps=1))






#
# @register_act('batch_norm_plus')
# class BatchNormPlus1d(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#
#         self.bn = nn.BatchNorm1d(*args, **kwargs)
#
#     def forward(self, x):
#         # as verified at torch=2.0: torch.sqrt() now support zero-vectors on forward and backward now
#         x = torch.relu(x).sqrt() - torch.relu(-x).sqrt()
#
#         return self.bn(x)
