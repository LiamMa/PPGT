from functools import partial

import torch
import torch.nn as nn
# from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head, act_dict

from ppgt.layer.utils.init import xavier_normal_init_, trunc_init_, kaiming_normal_init_, default_init_, lecun_normal_init_


'''
    Output Head for Dense Format GTs
'''


@register_head('dense_sum_graph')
class DenseHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2, pooling='sum'):
        super().__init__()

        self.pooling = pooling

        self.pre_pooling_norm = act_dict[cfg.gnn.get('pre_pooling_norm', 'none')](dim_in)

        self.post_pooling_norm = act_dict[cfg.gnn.get('post_pooling_norm', 'none')](dim_in)

        dropout = cfg.gnn.get('dropout', 0.)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        out_bn = cfg.gnn.get('out_bn', False)
        if out_bn:
            self.out_bn = nn.BatchNorm1d(dim_out)
        else:
            self.out_bn = nn.Identity()


        # if layer_norm:
        #     self.norm = nn.LayerNorm(dim_in)
        #
        # if batch_norm:
        #     self.norm = nn.BatchNorm1d(dim_in)

        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, max(dim_in // 2 ** (l + 1), dim_out), bias=True)
            for l in range(L-1)]

        list_FC_layers.append(
            nn.Linear(max(dim_in // 2 ** (L-1), dim_out), dim_out, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

        self.trunc_init = cfg.gnn.get('trunc_init', False)
        self.kaiming_init = cfg.gnn.get('kaiming_init', False)
        self.lecun_init = cfg.gnn.get('lecun_init', False)
        self.xavier_init = cfg.gnn.get('xavier_init', False)

        self.init_weights() # will also be called by init_vit_timm()


    def forward(self, batch):

        X = batch.X * batch.get('X_mask', 1.)
        X_mask = batch.X_mask if 'X_mask' in batch else torch.ones_like(X[:, :, 0])
        X = self.pre_pooling_norm(X) * X_mask

        if self.pooling == 'sum':
            O = torch.sum(X * X_mask, dim=1)
        elif self.pooling == 'mean':
            O = torch.sum(X * X_mask, dim=1)
            O = O / torch.sum(X_mask, dim=1)
        elif self.pooling == 'none':
            if 'X_mask' in batch:
                X_mask = batch.X_mask.type(torch.bool).squeeze(-1)
                O = X[X_mask]
            else:
                O = X
        else:
            raise NotImplementedError(f'not support the current pooling method [{self.pooling}]')

        O = self.post_pooling_norm(O)
        O = self.dropout(O)
        for l in range(self.L-1):
            O = self.FC_layers[l](O)
            O = self.activation(O)

        O = self.FC_layers[-1](O)
        pred, label = O, batch.y

        return self.out_bn(pred), label


    def init_weights(self):
        if self.trunc_init:
            self.apply(trunc_init_)
        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)
        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        elif self.xavier_init:
            self.apply(xavier_normal_init_)
        else:
            self.apply(default_init_)


        # self.apply(xavier_normal_init_)
    #     self.apply(trunc_init_)
        # use xavier initialization for better regression prediction


register_head('dense_mean_graph', partial(DenseHead, pooling='mean'))
register_head('dense_node', partial(DenseHead, pooling='none'))
