import torch.nn as nn
# from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head



@register_head('sg_graph')
class SgGraphHead(nn.Module):
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

        self.norm = nn.Identity()
        batch_norm = cfg.gnn.get('batch_norm', False)
        layer_norm = cfg.gnn.get('layer_norm', False)

        out_bn = cfg.gnn.get('out_bn', False)
        if out_bn:
            self.out_bn = nn.BatchNorm1d(dim_out)
        else:
            self.out_bn = nn.Identity()


        norm_fn = nn.Identity
        if layer_norm:
            norm_fn = nn.LayerNorm

        if batch_norm:
            norm_fn = nn.BatchNorm1d

        self.mlp1 = nn.Sequential(
            norm_fn(dim_in),
            nn.Linear(dim_in, dim_in),
            register.act_dict[cfg.gnn.act](),
            nn.Linear(dim_in, dim_in),
        )

        list_FC_layers = [norm_fn(dim_in)]
        for l in range(L):
            list_FC_layers += [
                nn.Linear(dim_in // 2 ** l, dim_in // 2 ** (l + 1), bias=True),
                register.act_dict[cfg.gnn.act]()
            ]
        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** L, dim_out, bias=True))

        self.mlp2 = nn.Sequential(*list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        subgraph_emb = self.pooling_fun(batch.x, batch.sg_n_root_index[0])
        subgraph_emb = self.mlp1(subgraph_emb)

        graph_emb = self.pooling_fun(subgraph_emb, batch.batch)
        graph_emb = self.mlp2(graph_emb)
        batch.graph_feature = graph_emb

        pred, label = self._apply_index(batch)

        return self.out_bn(pred), label
        # out_bn is for BREC only; disable for others




