import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.graphgym.register import  act_dict

from typing import List, Union


from .drop_path import GraphDropPath
from ..utils.residual import ResidualLayer


from ..utils.norm import NormalizationLayer

#
# def add_weight_norm(m, add_weight_norm, add_spectral_norm):
#     add_norm = add_spectral_norm or add_weight_norm
#
#     if add_spectral_norm and add_weight_norm:
#         assert False, '`add_spectral_norm` and `add_weight_norm` cannot be executed simultaneously'
#
#     if not isinstance(m, nn.Linear) or not add_norm:
#         return m
#
#     if add_spectral_norm:
#         return spectral_norm(m)
#
#     if add_weight_norm:
#         return weight_norm(m)
#     else:
#         return m
#

class MLP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hid_dim:int=None,
                 residual:bool=True,
                 num_layers:int=1,
                 ffn_ratio:int=1,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.spectral_norm = kwargs.get('add_spectral_norm', False)
        self.weight_norm = kwargs.get('add_weight_norm', False)
        # self.sphere_norm = kwargs.get('shpere_norm', False)

        # pre_norm
        self.pre_mlp_norm = NormalizationLayer(kwargs.get('pre_mlp_norm', 'none'), in_dim)


        hid_dim = [int(max(in_dim, out_dim) * ffn_ratio)] * num_layers if hid_dim is None else hid_dim
        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        hid_dim.append(out_dim)


        self.ffn_drop = ffn_drop = kwargs.get('dropout', 0.)
        proj_drop = kwargs.get('proj_drop', 0.)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()


        act = kwargs.get('act', 'gelu')
        if act is None: act = 'none'
        if act.lower() in ['swiglu']:
            warnings.warn('SwiGLU only support FFN like MLP')
            self.mlp = FFNSwiGLU(in_dim, out_dim, ffn_ratio=ffn_ratio * 2 / 3)
        else:
            act_fn = act_dict[kwargs.get('act', 'gelu')]
            # fc = add_weight_norm(nn.Linear(in_dim, hid_dim[0]), add_weight_norm=self.weight_norm, add_spectral_norm=self.spectral_norm)
            # fc = add_weight_norm(nn.Linear(in_dim, hid_dim[0]), add_weight_norm=self.weight_norm, add_spectral_norm=self.spectral_norm)
            mlp = [nn.Linear(in_dim, hid_dim[0])]
            for i in range(len(hid_dim)-1):
                mlp.append(act_fn())
                mlp.append(nn.Dropout(ffn_drop))
                mlp.append(nn.Linear(hid_dim[i], hid_dim[i+1]))
                # fc = add_weight_norm(nn.Linear(hid_dim[i], hid_dim[i+1]), add_weight_norm=self.weight_norm, add_spectral_norm=self.spectral_norm)
                # mlp.append(fc)

            self.mlp = nn.Sequential(*mlp)

        # if isinstance(self.pre_mlp_norm.norm_layer, nn.Identity) and isinstance(self.post_mlp_norm.norm_layer, nn.Identity) and isinstance(self.pre_res_post_mlp_norm.norm_layer, nn.Identity):
        #     self.mlp.apply(xavier_normal_init_)
        #     # xavier if no-normalization
        # else:
        #     # self.mlp.apply(trunc_init_)
        #     self.mlp.apply(xavier_normal_init_)

        # self.spherical = kwargs.get('spherical', False)

        # res_post_norm
        self.pre_res_post_mlp_norm = NormalizationLayer(kwargs.get('pre_res_post_mlp_norm', 'none'), out_dim)

        drop_path = kwargs.get('drop_path', 0.)
        drop_path_scale = kwargs.get('drop_path_scale', True)
        self.drop_path = GraphDropPath(drop_path, scale_by_keep=drop_path_scale)


        # post_norm
        self.post_mlp_norm = NormalizationLayer(kwargs.get('post_mlp_norm', 'none'), out_dim)


        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
            if self.residual:
                warnings.warn(f'Do not support residual connection with unequal in_dim={in_dim} and out_dim={out_dim} for now.')

        layer_scale = kwargs.get('layer_scale', False)
        rezero = kwargs.get('rezero', False)
        alpha = kwargs.get('alpha', 0.1)
        self.res_layer = ResidualLayer(rezero=rezero, layer_scale=layer_scale, alpha=alpha, dim=out_dim)


    def forward(self, x, batch_index=None, batch_tensor=None):

        res = x
        x = self.pre_mlp_norm(x, batch_index)
        # mlp = self.mlp if not self.spherical else self._spherical_mlp
        # x = mlp(x)

        x = self.mlp(x)
        x = self.proj_drop(x)

        x = self.pre_res_post_mlp_norm(x, batch_index)

        # if self.drop_path is not None and batch_index is not None:
        x = self.drop_path(x, batch_index, batch_tensor)
        x = self.res_layer(x, res)

        x = self.post_mlp_norm(x, batch_index)

        return x

    def _spherical_mlp(self, x):
        for m in self.mlp:
            x = m(x)
            if isinstance(m, nn.Linear):
                shapes = [1] * (x.dim()-1) + [-1]
                l2 = torch.norm(m.weight, p=2, dim=-1, keepdim=False)
                x = x / l2.view(*shapes)

        return x

import numpy as np


class SirenMLP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hid_dim:Union[int, List[int]]=None,
                 omega=30.,
                 act='siren',
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs

        # norm_fn = act_dict[kwargs.get('norm', 'null')]
        assert act in ['siren', 'spder']
        act_fn = act_dict[act]

        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]

        hid_dim.append(out_dim)

        self.fc = nn.Linear(in_dim, hid_dim[0])
        mlp = []
        for i in range(len(hid_dim)-1):
            mlp.append(act_fn(omega))
            fc = nn.Linear(hid_dim[i], hid_dim[i+1])
            scale = np.sqrt(6/hid_dim[i]) / omega
            nn.init.uniform_(fc.weight, -scale, scale)
            mlp.append(fc)

        self.mlp = nn.Sequential(*mlp)

        self.init_weights()


    def forward(self, x):

        return self.mlp(self.fc(x))

    def init_weights(self):
        nn.init.uniform_(self.fc.weight, -1/self.fc.weight.size(1), 1/self.fc.weight.size(1))
        self.mlp.apply(siren_init_)


def siren_init_(m):
    if isinstance(m, nn.Linear):

        in_dim  = m.weight.size(1)
        nn.init.uniform_(m.weight, -np.sqrt(6/in_dim), np.sqrt(6/in_dim))
        if m.bias is not None:
            nn.init.zeros_(m.bias)







class FFNSwiGLU(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hid_dim:int=None,
                 residual:bool=True,
                 num_layers:int=1,
                 ffn_ratio:int=1,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs


        hid_dim = int(max(in_dim, out_dim) * ffn_ratio) if hid_dim is None else hid_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(in_dim, hid_dim)
        self.fc3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):

        y = self.fc3(F.silu(self.fc1(x)) * self.fc2(x))

        return y
