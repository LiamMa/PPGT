import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.graphgym.register import register_act

from functools import partial

from einops import rearrange


def GroupNorm(dim, num_groups, eps=1e-05, affine=True, device=None, dtype=None):
    return nn.GroupNorm(num_groups, dim, eps, affine, device, dtype)

register_act('group_norm_x8', partial(GroupNorm, num_groups=8))
register_act('group_norm_x16', partial(GroupNorm, num_groups=16))






@register_act('group_norm++')
class GroupNormPP(nn.Module):
    '''
        With pre-scaler
    '''
    def __init__(self, dim, eps: float = 1,
                 affine: bool = True,
                 num_groups=1,
                 alpha:bool=True,
                 init_alpha:float=1.,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps

        if alpha:
            self.alpha = nn.Parameter(torch.ones(dim) * init_alpha)
        else:
            self.register_buffer('alpha', torch.ones(dim))

        self.gn = nn.GroupNorm(num_groups, dim, eps=eps, affine=affine)


    def forward(self, x):

        shapes = [1] * (x.dim()-1) + [-1]

        alpha = self.alpha.view(*shapes)

        return self.gn(alpha * x)


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'alpha={self.alpha.requires_grad}, '
                f'{self.gn.__repr__()}'
                f')')



register_act('group_norm++_x8', partial(GroupNormPP, num_groups=8))
register_act('group_norm++_x16', partial(GroupNormPP, num_groups=16))


























@register_act('group_rms_norm')
class GroupRMSNorm(nn.Module):
    '''
        Conditional RMS normalization V3
        > only support RMSNorm over the last dimension for now
    '''
    def __init__(self, dim, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 group=1,
                 alpha=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.group = group

        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=elementwise_affine)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=elementwise_affine)

        # no need to be dim

    def forward(self, x):

        shapes = [1] * (x.dim()-1) + [-1]

        gamma = self.gamma.view(*shapes)
        beta = self.beta.view(*shapes)
        # weight = self.weight.view(*shapes)

        # c = x * gamma + beta
        # rms = torch.norm(c, p=2, dim=-1, keepdim=True) / np.sqrt(c.size(-1))
        x = rearrange(x, 'n (g d) -> n g d', g=self.group)

        return F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps).flatten(1) * gamma + beta


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, eps={self.eps}, gamma={self.gamma.requires_grad}, beta={self.beta.requires_grad}, group={self.group})')


register_act('group_rms_norm_x8', partial(GroupRMSNorm, group=8))
register_act('group_rms_norm_x16', partial(GroupRMSNorm, group=16))

register_act('group_rms_norm_alpha_8', partial(GroupRMSNorm, group=8, alpha=True))






@register_act('layer_norm++')
class LayerNormPP(nn.Module):
    '''
        With pre-scaler
    '''
    def __init__(self, dim, eps: float = 1,
                 affine: bool = True,
                 num_groups=1,
                 alpha:bool=True,
                 init_alpha:float=1.,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.num_groups = num_groups

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        else:
            self.register_buffer('alpha', torch.ones(dim))

        # self.gn = nn.GroupNorm(num_groups, dim, eps=eps, affine=affine)

        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('gamma', torch.ones(dim))
            self.register_buffer('zeros', torch.zeros(dim))


    def forward(self, x):

        shapes = [1] * (x.dim()-1) + [-1]
        alpha = self.alpha.view(*shapes)
        gamma = self.gamma.view(*shapes)
        beta = self.beta.view(*shapes)

        return F.layer_norm(alpha * x, (x.size(-1), ), weight=None, eps=self.eps) * gamma + beta


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'affine={self.gamma.requires_grad}, '
                f'alpha={self.alpha.requires_grad}, '
                f')')
