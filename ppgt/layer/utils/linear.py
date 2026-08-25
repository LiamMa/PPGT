

import torch
from torch import nn

from einops import einsum

# from torch_scatter import scatter
# from torch_scatter import scatter





class AffineTransformLayer(nn.Module):
    '''
        Affine Transform Layer;  ResMLP (https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9888004)
    '''
    def __init__(self, dim, decay_factor=1.):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, dim) * decay_factor, requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        self.init_decay = decay_factor

    def forward(self, x):
        return x * self.gamma + self.beta

    def __repr__(self):
        return f'{super().__repr__()}(dim={self.gamma.size(1)}, init_decay={self.init_decay})'


class BiasOnlyLayer(nn.Module):
    '''
        Only Add Bias, For Depthwise Convolution
    '''
    def __init__(self, dim, decay_factor=1.):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(1, dim), requires_grad=True)

    def forward(self, x):
        return x + self.beta






class LinearCondBias(nn.Module):
    '''
        Add a conditional bias --> for scale equivariant
    '''
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.cond_bias = nn.Parameter(torch.zeros(1, out_dim), requires_grad=True)
        nn.init.normal_(self.bias, 0, 0.02)

    def forward(self, x, cond_bias):

        return self.fc(x) + self.cond_bias * cond_bias.view(-1, 1)



class GLU(nn.Module):
    '''
        Add a conditional bias --> for scale equivariant
    '''
    def __init__(self, dim, bias=True):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim, bias=bias)
        self.dim = dim

    def forward(self, x):
        x = self.fc(x)

        return  x[..., :self.dim] * torch.sigmoid(x[..., self.dim:])



class GroupedLinear(nn.Linear): # to inherit the class of nn.Linear
    '''
           (N, D) x (D, C, H) --> (N, H, C)
     or  (N, H, D) x (D, C, H) --> (N, H, C)
    '''


    def __init__(self, in_features: int, out_features: int, num_group:int =1, bias: bool = True,
                 device=None, dtype=None) -> None:
        nn.Module.__init__(self) # initilize as nn.Module instead of nn.Linear

        self.in_dim = in_features
        self.out_dim = out_features
        self.num_group = num_group

        self.weight = nn.Parameter(torch.zeros(in_features, out_features, num_group))
        nn.init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, num_group, out_features), requires_grad=bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        if x.dim() == 2:
            y = einsum(x, self.weight, 'n d, d c h -> n h c')
        else:
            y = einsum(x, self.weight, 'n h d, d c h -> n h c')

        if self.bias is None:
            return y

        return y + self.bias

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_group={}, bias={})]'.format(
            self.__class__.__name__,
            self.in_dim,
            self.out_dim,
            self.num_group,
            self.bias,
        )







