import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.graphgym.register import register_act

from functools import partial


from einops import rearrange



@register_act('rms_renorm')
class RMSReNorm(nn.Module):
    '''
    '''
    def __init__(self, dim, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 gamma:bool=True,
                 beta:bool=True,
                 num_groups=1,
                 id_init=False,
                 unif_init=False,
                 trunc_init=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.gamma = gamma
        self.beta = beta
        # self.group = group
        self.num_groups = num_groups


        self.gamma = nn.Parameter(torch.zeros(num_groups), requires_grad=gamma)
        self.beta = nn.Parameter(torch.ones(num_groups), requires_grad=beta)

        self.weight = nn.Parameter(torch.ones(dim), requires_grad=elementwise_affine)


    def forward(self, x):

        shapes = [1] * (x.dim()-1) + [-1]
        gamma, beta = self.gamma.view(*shapes).unsqueeze(-1), self.beta.view(*shapes).unsqueeze(-1)
        weight = self.weight.view(*shapes)

        x = rearrange(x, 'n (g d) -> n g d', g=self.num_groups)
        c = x * gamma + beta

        rms = (c ** 2).mean(dim=-1, keepdim=True).sqrt()

        return (F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps) * rms).flatten(1) * weight


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, affine={self.weight.requires_grad},'
                f' eps={self.eps}, gamma={self.gamma.requires_grad},'
                f' beta={self.beta.requires_grad})')

register_act('rms_renorm_x8', partial(RMSReNorm, num_groups=8))
register_act('rms_renorm_x16', partial(RMSReNorm, num_groups=16))
