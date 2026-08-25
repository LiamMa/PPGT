

import torch
from torch import nn





class ResidualLayer(nn.Module):
    '''
        Residual Layer;
        - support rezero and layer_scale
    '''
    def __init__(self, rezero=False, layer_scale=False, alpha=None, dim=1, ada_res=False, rescale=False):
        super().__init__()
        self.rezero = rezero
        self.layer_scale = layer_scale

        if rezero and layer_scale:
            assert False, 'cannot apply [ReZero] and [Layer Scale]'


        if rezero:
            self.init_alpha = float(alpha) if alpha is not None else 0
            self.alpha = nn.Parameter(torch.ones(1) * self.init_alpha, requires_grad=True)
            # self.alpha = nn.Parameter(torch.ones(1) * 1e-6, requires_grad=True)
            # rezero is set to zero be default
        elif layer_scale:
            self.init_alpha= float(alpha) if alpha is not None else 0.1
            self.alpha = nn.Parameter(torch.ones(dim) * self.init_alpha, requires_grad=True)
        else:
            # self.init_alpha= alpha if alpha is not None else 1
            self.init_alpha= 1.
            self.register_buffer('alpha', torch.ones(1) * self.init_alpha)
            # to allow scale residual with a fixed scalar

        self.dim = self.alpha.size(-1)

        # # ----------- Adaptive Residual Branch---------
        # if self.ada_res:
        #     self.res_alpha = nn.Parameter(torch.ones(1, dim), requires_grad=True)

        # ----------- residual rescale ---------
        r'''
            Assume that res and path are independent r.v. with std as $\sigma$; 
            the sum of them has std $\sqrt{2}\sigma$ per channel.
            > We can apply a rescaling term $1/\sqrt{2}$ to further regularize it
        '''
        # self.rescale = rescale
        # self.ada_res = ada_res # if True, assign a learnable affine Transform for residual


    def forward(self, x, x_res):

        shapes = [1] * (x.dim() - 1) + [-1]

        return (x * self.alpha.view(*shapes) + x_res)

    def __repr__(self):
        return f'{super().__repr__()}(rezero={self.rezero}, layer_scale={self.layer_scale}, ' \
               f'init_alpha={self.init_alpha}, ' \
               f'alpha_shape={self.alpha.shape}) '
               # f'ada_res={self.ada_res})'




class JumpingKnowledgeLayer(nn.Module):
    '''
        Jumping Knowledge Layer;
        - support rezero and layer_scale
    '''
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, x_res):
        if not self.rezero and not self.layer_scale:
            return x + x_res

        return x * self.alpha + x_res

    def __repr__(self):
        return f'{super().__repr__()}(rezero={self.rezero}, layer_scale={self.layer_scale}, ' \
               f'layer_scale_init={self.layer_scale_init}, ' \
               f'alpha_shape={self.alpha.shape})'
