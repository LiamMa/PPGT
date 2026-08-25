import torch.nn as nn

from torch_geometric.graphgym.register import register_act

from functools import partial

from ..layer.utils.rbf import RBFLayer






@register_act('rbf_layer_norm')
class RBFLayerNorm(nn.Module):
    '''
        LayerNorm plus RBF mapping the Norm to a vector encoding
    '''
    def __init__(self, dim, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias:bool=True,
                 rbf_dim=16,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps

        self.fc_x = nn.Linear(dim, dim, bias=True)
        self.fc_rbf = nn.Linear(rbf_dim, dim, bias=False)
        self.rbf = RBFLayer(1, rbf_dim)


    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)
        centered_x = x - mean
        std = (centered_x**2).mean(dim=-1, keepdim=True).sqrt()
        x_ = centered_x / (std + self.eps)
        rbf_enc = self.rbf(std)

        return self.fc_x(x_) + self.fc_rbf(rbf_enc)


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine},'
                f'rbf_dim={self.rbf_dim},'
                f')')


register_act('rbf_layer_norm_16', partial(RBFLayerNorm, rbf_dim=16))

