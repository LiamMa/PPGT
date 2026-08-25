import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_act

from functools import partial




@register_act('affine')
class AffineTransform(nn.Module):
    '''
        LayerScale Layer
    '''
    def __init__(self, dim,
                 affine: bool = True,
                 bias: bool = True,
                 init_alpha:float=0.1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.affine = affine
        self.bias = bias
        self.init_alpha=init_alpha


        self.gamma = nn.Parameter(torch.ones(dim) * init_alpha, requires_grad=affine)
        # nn.init.trunc_normal_(self.gamma, mean=0., std=0.02)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=bias)



    def forward(self, x):

        shapes = [1] * (x.dim()-1) + [-1]
        gamma, beta = self.gamma.view(*shapes), self.beta.view(*shapes)

        return x * gamma + beta


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, '
                f'affine={self.gamma.requires_grad}, '
                f'bias={self.beta.requires_grad}), '
                f'init_alpha={self.init_alpha}')



register_act('affine_1e-1', partial(AffineTransform, init_alpha=1e-1))
register_act('affine_1e-3', partial(AffineTransform, init_alpha=1e-3))
