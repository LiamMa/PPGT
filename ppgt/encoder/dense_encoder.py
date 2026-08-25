import torch
import numpy as np
import warnings
from timm.layers import DropPath
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder, act_dict

from ..layer.utils.init import (trunc_init_, uniform_init_, xavier_normal_init_,
                                kaiming_uniform_init_,  kaiming_normal_init_,
                                kaiming_uniform_linear_init_, kaiming_normal_linear_init_,
                                default_init_, lecun_normal_init_
                                )
from ..network.initialization import init_weights_vit_timm
from ..network.utils import copy_and_pop








from ..layer.utils.residual import ResidualLayer





@register_edge_encoder('SparseToDenseEncoder')
class SparseToDenseEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 x_name='x',
                 e_name='edge',
                 pe_name='rrwp',
                 add_invN:bool=False,
                 sin_L:int=0, # if > 0; sin enhancement
                 pe_to_x:bool=True,
                 add_logD_to_x=False,
                 add_logN_to_x=False,
                 add_loc:bool=False,
                 loc_name:str='pos',
                 loc_dim:int=2,
                 **kwargs):
        super().__init__()

        if kwargs.get('e_dim', None) is not None:
            out_dim = kwargs.e_dim

        self.x_name = x_name
        self.e_name = e_name
        self.pe_name = pe_name
        self.add_invN = add_invN

        self.add_loc = add_loc
        self.loc_name = loc_name
        self.loc_dim = loc_dim

        self.sin_L = sin_L
        if self.sin_L > 0:
            eps = (2 ** torch.arange(self.sin_L)) * np.pi
            self.register_buffer('eps', eps)


        pe_dim = in_dim
        if self.add_invN:
            pe_dim += 1
        if self.add_loc:
            pe_dim += self.loc_dim

        pe_dim = pe_dim + sin_L * 2 * pe_dim

        self.pe_fc = nn.Linear(pe_dim, out_dim)

        self.pe_to_x = pe_to_x
        if self.pe_to_x:
            self.pe_fc_to_x = nn.Linear(pe_dim, out_dim)

        self.add_logD_to_x = add_logD_to_x
        self.add_logN_to_x = add_logN_to_x

        if self.add_logD_to_x:
            self.logD_fc = nn.Linear(1, out_dim, bias=False)

        if self.add_logN_to_x:
            self.logN_fc = nn.Linear(1, out_dim, bias=False)

        # init -
        self.default_init = kwargs.get('default_init', False)
        self.trunc_init = kwargs.get('trunc_init', False)
        self.kaiming_linear_init = kwargs.get('kaiming_linear_init', False)
        self.uniform_init = kwargs.get('uniform_init', False)
        self.kaiming_init = kwargs.get('kaiming_init', False)
        self.kaiming_uniform_init = kwargs.get('kaiming_uniform_init', False)
        self.kaiming_uniform_linear_init = kwargs.get('kaiming_uniform_linear_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        self.init_weights()

    def forward(self, batch):

        max_num_nodes = torch.max(batch.ptr[1:] - batch.ptr[:-1])
        x = batch.pop(self.x_name)
        X, x_mask = pyg.utils.to_dense_batch(x, batch.batch, fill_value=0., max_num_nodes=max_num_nodes)

        batch.X = X
        batch.X_mask = x_mask.type(torch.float).unsqueeze(-1)
        batch.X_mask_bool = x_mask
        e_mask = x_mask.unsqueeze(-1) & x_mask.unsqueeze(1)
        batch.E_mask = e_mask.type(torch.float).unsqueeze(-1)
        batch.E_mask_bool = e_mask


        e = batch.pop(f'{self.e_name}_attr')
        e_index = batch.pop(f'{self.e_name}_index')
        E = pyg.utils.to_dense_adj(e_index, batch.batch,
                                   edge_attr=e,
                                   max_num_nodes=max_num_nodes,
                                   )


        pe = batch.pop(f'{self.pe_name}_attr')
        pe_index = batch.pop(f'{self.pe_name}_index')
        PE = pyg.utils.to_dense_adj(pe_index, batch.batch,
                                    edge_attr=pe,
                                    max_num_nodes=max_num_nodes
                                    ).transpose(1, 2) # to be compatible with original GRIT, which is right-mulmat

        if self.add_invN:
            invN = (1/x_mask.sum(dim=1)).view(PE.size(0), 1, 1, 1) * torch.ones_like(PE[..., :1])
            PE = torch.cat([PE, invN], dim=-1)


        if self.add_loc:
            loc, loc_mask = pyg.utils.to_dense_batch(batch[self.loc_name], batch.batch, fill_value=0., max_num_nodes=max_num_nodes)
            assert (loc_mask == x_mask).all()
            loc = loc.unsqueeze(2) - loc.unsqueeze(1) # b n d, b m d --> b n m d
            PE = torch.cat([PE, loc], dim=-1) * batch.E_mask


        if self.sin_L > 0:
            shapes = [1] * PE.ndim + [-1]
            eps = self.eps.view(*shapes)
            pe = PE.unsqueeze(-1) * eps
            sin_pe = torch.sin(pe)
            cos_pe = torch.cos(pe)
            PE = torch.cat([sin_pe.flatten(PE.ndim - 1), cos_pe.flatten(PE.ndim-1), PE], dim=-1)


        E[batch.E_mask_bool] = self.pe_fc(PE[batch.E_mask_bool]) + E[batch.E_mask_bool]
        if self.pe_to_x:
            PE_x = torch.diagonal(PE, dim1=1, dim2=2).transpose(-1, -2)
            X = self.pe_fc_to_x(PE_x) + X



        X = X * batch.X_mask
        E = E * batch.E_mask

        batch.X = X
        batch.E = E



        return batch



    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'



    def init_weights(self):
        if self.xavier_init:
            self.apply(xavier_normal_init_)

        elif self.trunc_init:
            self.apply(trunc_init_)

        elif self.kaiming_linear_init:
            self.apply(kaiming_normal_linear_init_)

        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)

        elif self.kaiming_uniform_init:
            self.apply(kaiming_uniform_init_)

        elif self.kaiming_uniform_linear_init:
            self.apply(kaiming_uniform_linear_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        else:
            self.apply(default_init_)

        # self.apply(trunc_init_)






@register_edge_encoder('DenseFFN')
class DenseFFN(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name='E',
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')


        self.pe_name = pe_name if pe_name != '' else 'E'
        assert in_dim == out_dim

        self.pre_backbone_norm = act_dict[kwargs.get('pre_backbone_norm', 'none')](in_dim)



        act = kwargs.get('act', 'gelu')
        self.act = act
        self.num_blocks = kwargs.get('num_blocks', 0)
        # ---- DropPath sccaling -----
        drop_path_rate = kwargs.get('drop_path', 0.)
        kwargs = copy_and_pop(kwargs, ['drop_path'])
        # safe operation, prevent from changing the cfg.
        layerwise_drop_path_rate = kwargs.get('layerwise_drop_path_rate', False)
        # by default use uniform droprate for droppath (from CaiT)
        num_layers =  self.num_blocks
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)] if layerwise_drop_path_rate else [drop_path_rate] * num_layers
        # ---------------------------

        mlp =  ([(nn.Identity(), 'x -> x')] +
                [(DenseMLP(out_dim, out_dim, drop_path=drop_path_rate[i],
                      **kwargs), 'x -> x')
                 for i in range(self.num_blocks)])
        self.mlp = pyg.nn.Sequential('x', mlp)


        self.post_backbone_norm = act_dict[kwargs.get('post_backbone_norm', 'none')](out_dim)
        # self.post_backbone_norm = NormalizationLayer(kwargs.get('post_backbone_norm', 'none'), out_dim)


        self.lecun_init = kwargs.get('lecun_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)

        self.init_weights()

    def forward(self, batch):

        E = batch.get(self.pe_name)
        mask = batch.get(f'{self.pe_name}_mask', 1.)

        E = self.pre_backbone_norm(E)
        E = self.mlp(E)
        E = self.post_backbone_norm(E)

        E = E * mask
        batch[self.pe_name] = E

        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'

    def init_weights(self):
        if 'gaussian' in self.act:
            self.mlp.apply(uniform_init_)
        else:
            init_weights_vit_timm(self.mlp)

        if self.lecun_init:
            self.mlp.apply(lecun_normal_init_)

        if self.xavier_init:
            self.mlp.apply(xavier_normal_init_)




class DenseMLP(torch.nn.Module):
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

        # pre_norm
        self.pre_mlp_norm  = act_dict[kwargs.get('pre_mlp_norm', 'none')](in_dim)


        hid_dim = [int(max(in_dim, out_dim) * ffn_ratio)] * num_layers if hid_dim is None else hid_dim
        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        hid_dim.append(out_dim)


        self.ffn_drop = ffn_drop = kwargs.get('dropout', 0.)
        proj_drop = kwargs.get('proj_drop', 0.)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()


        act = kwargs.get('act', 'gelu')
        if act is None:
            act = 'none'
        act_fn = act_dict[kwargs.get('act', 'gelu')]
        # fc = add_weight_norm(nn.Linear(in_dim, hid_dim[0]), add_weight_norm=self.weight_norm, add_spectral_norm=self.spectral_norm)
        # fc = add_weight_norm(nn.Linear(in_dim, hid_dim[0]), add_weight_norm=self.weight_norm, add_spectral_norm=self.spectral_norm)
        mlp = [nn.Linear(in_dim, hid_dim[0])]
        for i in range(len(hid_dim)-1):
            mlp.append(act_fn())
            mlp.append(nn.Dropout(ffn_drop))
            mlp.append(nn.Linear(hid_dim[i], hid_dim[i+1]))

        self.mlp = nn.Sequential(*mlp)


        # res_post_norm
        # self.pre_res_post_mlp_norm = NormalizationLayer(kwargs.get('pre_res_post_mlp_norm', 'none'), out_dim)
        # self.pre_res_post_mlp_norm = NormalizationLayer(kwargs.get('pre_res_post_mlp_norm', 'none'), out_dim)

        drop_path = kwargs.get('drop_path', 0.)
        drop_path_scale = kwargs.get('drop_path_scale', True)
        self.drop_path = DropPath(drop_path, scale_by_keep=drop_path_scale)

        # post_norm
        self.post_mlp_norm   = act_dict[kwargs.get('post_mlp_norm', 'none')](out_dim)


        self.residual = residual
        if in_dim != out_dim:
            self.residual = False
            if self.residual:
                warnings.warn(f'Do not support residual connection with unequal in_dim={in_dim} and out_dim={out_dim} for now.')

        layer_scale = kwargs.get('layer_scale', False)
        rezero = kwargs.get('rezero', False)
        alpha = kwargs.get('alpha', 0.1)
        self.res_layer = ResidualLayer(rezero=rezero, layer_scale=layer_scale, alpha=alpha, dim=out_dim)


    def forward(self, x):

        res = x
        x = self.pre_mlp_norm(x)
        # mlp = self.mlp if not self.spherical else self._spherical_mlp
        # x = mlp(x)
        x = self.mlp(x)
        x = self.proj_drop(x)


        # if self.drop_path is not None and batch_index is not None:
        x = self.drop_path(x)
        x = self.res_layer(x, res)

        x = self.post_mlp_norm(x)

        return x