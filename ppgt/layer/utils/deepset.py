
import torch
from torch import nn

from .scatter import scatter

from .mlp import MLP






class DeepSetLinear(nn.Module):
    '''
        Simple Deepset Linear Layer
    '''
    def __init__(self, in_dim, out_dim, op='sum', bias=True):
        super().__init__()

        self.op = op
        self.gamma_fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.beta_fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, index, dim_size=None):

        gamma = self.gamma_fc(x)
        beta = scatter(self.beta_fc(x), index, dim=0, dim_size=dim_size, reduce=self.op)[index]

        return gamma - beta

class SetFFN(nn.Module):
    '''
        Set FFN
    '''
    def __init__(self, in_dim, out_dim, hid_dim=None, op='sum', act=nn.ReLU, bias=True):
        super().__init__()

        if hid_dim is None:
            hid_dim = max(in_dim, out_dim)

        self.op = op
        self.gamma_ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            act(),
            nn.Linear(hid_dim, out_dim, bias=bias)
        )

        self.beta_ffn = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=bias),
            act(),
            nn.Linear(hid_dim, out_dim, bias=bias)
        )

    def forward(self, x, index, dim_size=None):

        gamma = self.gamma_ffn(x)
        beta = self.beta_ffn(scatter(x, index, dim=0, dim_size=dim_size, reduce=self.op))[index]

        return gamma - beta




class DWDeepSetLinear(nn.Module):
    '''
        DeepSet Layer in Depthwise Fashion
    '''
    def __init__(self, dim, op='sum', bias=True):
        super().__init__()

        self.op = op
        self.Gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.Lambda = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim), requires_grad=bias)

    def forward(self, x, index, dim_size=None):
        shapes = [1 for i in range(x.dim())]
        shapes[-1] = -1

        gamma = x * self.Gamma.view(*shapes)
        beta = scatter(x, index, dim=0, dim_size=dim_size, reduce=self.op)[index] * self.Lambda.view(*shapes)

        return gamma - beta + self.bias.view(*shapes)



class UnivEquiSetLinear(nn.Module):
    '''
        Universal Equivariant Set Linear
        XA+ 11^TXB + 1c
    '''
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()

        self.gamma_fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.beta_fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, index, set_sizes=None, dim_size=None):
        gamma = self.gamma_fc(x)
        if set_sizes is None:
            beta = scatter(x, index, dim=0, dim_size=dim_size, reduce='mean')
        else:
            beta = scatter(x, index, dim=0, dim_size=dim_size, reduce='mean')
            shapes = [-1] + [1] * (beta.dim() - 1)
            beta = beta / set_sizes.view(*shapes)

        return gamma + self.beta_fc(beta)[index]



class AdaPoolSetPool(nn.Module):
    '''
        y = FFN_1(x) + AdaPool(FFN_2(x))
        AdaPool = [AvgPool(x) + log(set-size)]
    '''
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.reindex = kwargs.get('reindex', True) # recover the original sizes by re-indexing

        self.mlp = MLP(in_dim, out_dim-1, **kwargs)

    def forward(self, x, index, dim_size=None):

        x = self.gamma_fc(x)
        y = scatter(x, index, dim=0, dim_size=dim_size, reduce='mean')
        count = scatter(torch.ones_like(x[:, :1]), index, dim=0, dim_size=dim_size, reduce='sum')
        y = torch.cat([y, torch.log(1+count)], dim=-1)

        return y