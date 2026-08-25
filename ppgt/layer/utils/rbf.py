



import torch
import torch.nn as nn









class RBFLayer(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_dim, out_dim, affine=False, bias=False):
        super().__init__()
        self.in_dim =in_dim
        self.out_dim =out_dim
        self.mus = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.sigmas = nn.Parameter(torch.ones(out_dim))
        self.reset_parameters()

        if affine:
            self.gamma = nn.Parameter(torch.ones(out_dim))
        else:
            self.register_buffer('gamma', torch.ones(out_dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_buffer('beta', torch.zeros(out_dim))


    def reset_parameters(self):
        nn.init.uniform_(self.mus, -1, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, x):
        # N x D
        c_size = [1] * (x.dim()-1) + [x.size(-1), -1]
        s_size = [1] * (x.dim()-1) + [-1]
        c = self.mus.view(*c_size)
        s = self.sigmas.view(*s_size)
        # N x D -> N x D x C -> N x C
        dist = (x.unsqueeze(-1) - c).pow(2).sum(dim=-2)
        gauss = torch.exp(-1 * dist * s**2)

        gamma, beta = self.gamma.view(*s_size), self.beta.view(*s_size)


        return gauss * gamma + beta






class RBFLayerCenter(nn.Module):
    '''Transforms incoming data using a given radial basis function.
        - Input: (1, N, in_features) where N is an arbitrary batch size
        - Output: (1, N, out_features) where N is an arbitrary batch size'''

    def __init__(self, in_dim, out_dim, affine=False, bias=False):
        super().__init__()
        self.in_dim =in_dim
        self.out_dim =out_dim
        self.mus = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.mus_center = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.sigmas = nn.Parameter(torch.ones(out_dim))
        self.reset_parameters()

        if affine:
            self.gamma = nn.Parameter(torch.ones(out_dim))
        else:
            self.register_buffer('gamma', torch.ones(out_dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_buffer('beta', torch.zeros(out_dim))


    def reset_parameters(self):
        nn.init.uniform_(self.mus, -1, 1)
        nn.init.uniform_(self.mus_center, -1, 1)

        nn.init.constant_(self.sigmas, 10)

    def forward(self, x, x_center):
        # N x D
        c_size = [1] * (x.dim()-1) + [x.size(-1), -1]
        s_size = [1] * (x.dim()-1) + [-1]
        c = (self.mus.view(*c_size) +
             x_center.unsqueeze(-1) * self.mus_center.view(*c_size))
        s = self.sigmas.view(*s_size)
        # N x D -> N x D x C -> N x C
        dist = (x.unsqueeze(-1) - c).pow(2).sum(dim=-2)
        gauss = torch.exp(-1 * dist * s**2)

        gamma, beta = self.gamma.view(*s_size), self.beta.view(*s_size)

        return gauss * gamma + beta