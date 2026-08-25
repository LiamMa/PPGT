import torch
import torch.nn as nn
import numpy as np

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act

from functools import partial



class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)



register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, negative_slope=0.3, inplace=cfg.mem.inplace))
register_act('lrelu_02', partial(nn.LeakyReLU, negative_slope=0.2, inplace=cfg.mem.inplace))
# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
register_act('tanh', nn.Tanh)
register_act('sigmoid', nn.Sigmoid)
register_act('silu', nn.SiLU)

# register_act('elu', partial(nn.ELU, inplace=cfg.mem.inplace))
register_act('none', nn.Identity)
register_act('null', nn.Identity)


# class GaussianActivation(torch.nn.Module):
#     def __init__(self, sigma=0.05, trainable=False, minus_mean=False):
#         super().__init__()
#         self.sigma = sigma
#         self.minus_mean = minus_mean
#
#         if trainable:
#             # Todo: To consider different channel with different sigma
#             self.sigma = torch.nn.Parameter(torch.ones(1) * sigma, requires_grad=True)
#
#         self.trainable = trainable
#
#     def forward(self, input):
#         """
#         Args:
#             opt
#             x (torch.Tensor [B,num_rays,])
#         """
#         if self.minus_mean:
#             mean = input.mean(dim=-1, keepdim=True)
#         else:
#             mean = 0
#         if self.trainable:
#             shapes = [1] * input.dim()
#             shapes[-1] = -1
#             sigma = self.sigma.view(*shapes)
#         else:
#             sigma = self.sigma
#
#         k1 = (-0.5*(input - mean)**2/sigma**2).exp()
#         return k1
#
# register_act('gaussian', GaussianActivation)
# register_act('gaussianTrainable', partial(GaussianActivation, trainable=True))


class RadialBasisFunc(torch.nn.Module):
    '''
        Simple Radial Basis Activation
    '''
    def __init__(self):
        super().__init__()
    def forward(self, input):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        return torch.exp(-(input)**2)


register_act('rbf', RadialBasisFunc)

@register_act('signedsqrt')
class SignedSqrt(torch.nn.Module):
    '''
        Sign-preserving square root: sign(x) * sqrt(|x|)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor [N,D])
        """
        return torch.relu(x).sqrt() - torch.relu(-x).sqrt()

from torch.nn import functional as F





@register_act('soft_abs')
class SoftAbs(torch.nn.Module):
    r'''
        Smooth absolute value, built from two Softplus branches.

        .. math::
            \text{SoftAbs}(x) = \text{Softplus}(x) + \text{Softplus}(-x)
                                - \frac{2}{\beta}\log 2

        The constant makes :math:`\text{SoftAbs}(0) = 0`, and the function
        approaches :math:`|x|` as :math:`\beta \to \infty`.

        Args:
            beta: the :math:`\beta` value of the underlying Softplus. Default: 1
            threshold: Softplus reverts to a linear function above this. Default: 20

        Shape:
            - Input: :math:`(*)`, any number of dimensions.
            - Output: same shape as the input.
    '''
    def __init__(self, beta:float=1., threshold:float=20.0):
        super().__init__()

        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        """
        Args:
            x (torch.Tensor [N,D])
        """
        return (F.softplus(x, self.beta, self.threshold) + F.softplus(-x, self.beta, self.threshold)
                - 2 / self.beta * np.log(2))

    def extra_repr(self) -> str:
        return f"beta={self.beta}, threshold={self.threshold}"