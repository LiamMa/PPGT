import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.graphgym.register import register_act


@register_act('ada_rms_norm')
class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm.

    Applies RMSNorm to ``x``, then rescales by the RMS of an affine
    transform ``alpha * x + beta``.
    """

    def __init__(self, dim, eps: float = 1e-6,
                 beta: bool = True,
                 alpha: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim = dim
        self.eps = eps

        if beta:
            self.beta = nn.Parameter(torch.ones(dim, **factory_kwargs))
        else:
            self.register_buffer('beta', torch.ones(1, **factory_kwargs))

        if alpha:
            self.alpha = nn.Parameter(torch.zeros(dim, **factory_kwargs))
        else:
            self.register_buffer('alpha', torch.zeros(1, **factory_kwargs))

    def forward(self, x):
        orig_shapes = x.size()
        shapes = [1] * (x.dim() - 1) + [-1]
        alpha = self.alpha.view(*shapes)
        beta = self.beta.view(*shapes)

        c = x * alpha + beta
        rescale = ((c ** 2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        normed_x = F.rms_norm(x, (x.size(-1),), weight=None, eps=self.eps)
        return (normed_x * rescale).view(orig_shapes)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'dim={self.dim}, eps={self.eps}, '
                f'beta={self.beta.requires_grad}, '
                f'alpha={self.alpha.requires_grad})')
