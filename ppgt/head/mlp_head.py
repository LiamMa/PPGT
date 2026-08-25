from functools import partial

import torch
import torch.nn as nn
# from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head, act_dict

from ppgt.layer.utils.init import xavier_normal_init_, trunc_init_, kaiming_normal_init_, default_init_, lecun_normal_init_, kaiming_uniform_init_

import ppgt.head.pooling.cls as cls_pooling  # noqa: F401  (registers the cls poolings)



@register_head('san_graph')
class SANGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        # batch_norm = cfg.gnn.get('batch_norm', False)
        # layer_norm = cfg.gnn.get('layer_norm', False)

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
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(max(dim_out, dim_in // 2 ** L), dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_emb = self.post_pooling_norm(graph_emb)

        graph_emb = self.dropout(graph_emb)
        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return self.out_bn(pred), label
        # out_bn is for BREC only; disable for others






@register_head('ada_graph')
class AdaGraphHead(nn.Module):
    """
        Adaptively switch between mean and scaled-sum
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False
        self.pooling_fun = register.pooling_dict['add']

        # self.affine = nn.Parameter(torch.zeros(1, dim_in))
        # nn.init.trunc_normal_(self.affine, std=0.02)
        self.affine = nn.Linear(1, dim_in, bias=False)

        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True)
            for l in range(L)]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)
        graph_order = self.pooling_fun(torch.ones_like(batch.x[:, :1]), batch.batch)
        # graph_emb = (graph_emb / graph_order) * (graph_order * self.affine.exp() + 1)
        graph_emb = (graph_emb / graph_order) * (self.affine(graph_order) + 1)

        for l in range(self.L):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)
        graph_emb = self.FC_layers[self.L](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label



@register_head('linear_graph')
class LinearHead(nn.Module):
    """Linear/MLP prediction head.

    Pools node representations (unless ``no_pooling``) and applies an
    ``L``-layer MLP. With ``decay_dim`` the hidden width is halved at every
    layer; otherwise it stays constant.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of layers in the MLP.
        no_pooling (bool): Predict per node instead of per graph.
        decay_dim (bool): Halve the hidden width at every layer.
    """

    def __init__(self, dim_in, dim_out, L=2, no_pooling=False, decay_dim=False):
        super().__init__()
        self.deg_scaler = False
        self.fwl = False

        self.pooling_index = "batch"
        if cfg.model.graph_pooling == 'cls_mask':
            self.pooling_index = "cls_mask"
        elif cfg.model.graph_pooling == 'cls_token':
            self.pooling_index = "cls_token"

        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        self.no_pooling = no_pooling


        # batch_norm = cfg.gnn.get('batch_norm', False)
        # layer_norm = cfg.gnn.get('layer_norm', False)

        self.post_pooling_norm = act_dict[cfg.gnn.get('post_pooling_norm', 'none')](dim_in)

        dropout = cfg.gnn.get('dropout', 0.)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()


        out_dim = cfg.gnn.get('out_dim', None) # for brec dataset
        if out_dim is not None:
            dim_out = out_dim

        out_bn = cfg.gnn.get('out_bn', False)
        if out_bn:
            self.out_bn = nn.BatchNorm1d(dim_out)
        else:
            self.out_bn = nn.Identity()

        self.decay_dim = decay_dim

        if decay_dim:
            min_dim = min(dim_in, 2 * dim_out)
            act_fn= register.act_dict[cfg.gnn.act]
            layers = []
            for l in range(L-1):
                layers += [
                    nn.Linear(dim_in // 2 ** l, max(dim_in // 2 ** (l + 1), min_dim), bias=True),
                    act_fn(),
                    nn.Dropout(dropout)
                ]
            layers.append(nn.Linear(max(dim_in // 2 ** (L-1), min_dim), dim_out, bias=True))
        else:
            hid_dim = max(dim_in, dim_out)
            act_fn= register.act_dict[cfg.gnn.act]
            layers = []
            for l in range(L-1):
                layers += [
                    nn.Linear(dim_in if l ==0 else hid_dim, hid_dim, bias=True),
                    act_fn(),
                    nn.Dropout(dropout)
                ]
            layers.append(nn.Linear(dim_in if L==1 else hid_dim, dim_out, bias=True))


        self.mlp = nn.Sequential(*layers)


        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version
        self.trunc_init = cfg.gnn.get('trunc_init', False)
        self.kaiming_init = cfg.gnn.get('kaiming_init', False)
        self.kaiming_uniform_init = cfg.gnn.get('kaiming_uniform_init', False)
        self.lecun_init = cfg.gnn.get('lecun_init', False)
        self.xavier_init = cfg.gnn.get('xavier_init', False)

        self.init_weights() # will also be called by init_vit_timm()

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch[self.pooling_index]) if not self.no_pooling else batch.x

        graph_emb = self.post_pooling_norm(graph_emb)

        graph_emb = self.dropout(graph_emb)
        graph_emb = self.mlp(graph_emb)

        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return self.out_bn(pred), label

    def init_weights(self):
        if self.trunc_init:
            self.apply(trunc_init_)
        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)
        elif self.kaiming_uniform_init:
            self.apply(kaiming_uniform_init_)
        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        elif self.xavier_init:
            self.apply(xavier_normal_init_)
        else:
            self.apply(default_init_)



register_head('linear_node', partial(LinearHead, no_pooling=True))
register_head('linear_graph_nodecay', partial(LinearHead, decay_dim=False))
register_head('linear_node_nodecay', partial(LinearHead, no_pooling=True, decay_dim=False))


