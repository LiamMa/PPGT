import torch.nn as nn
# from torch_scatter import scatter

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head







@register_head('cls_graph')
class CLSGraphHead(nn.Module):
    """
    SAN prediction head for graph prediction tasks.
    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=2):
        super().__init__()
        # batch_norm = cfg.gnn.get('batch_norm', False)
        # layer_norm = cfg.gnn.get('layer_norm', False)

        dropout = cfg.gnn.get('dropout', 0.)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        out_bn = cfg.gnn.get('out_bn', False)
        if out_bn:
            self.out_bn = nn.BatchNorm1d(dim_out)
        else:
            self.out_bn = nn.Identity()


        list_FC_layers = [
            nn.Linear(dim_in // 2 ** l, max(dim_in // 2 ** (l + 1), dim_out), bias=True)
            for l in range(L-1)]

        list_FC_layers.append(
            nn.Linear(dim_in // 2 ** (L-1), dim_out, bias=True))

        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        self.activation = register.act_dict[cfg.gnn.act]()
        # note: modified to add () in the end from original code of 'GPS'
        #   potentially due to the change of PyG/GraphGym version

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):

        graph_emb = batch.cls

        graph_emb = self.dropout(graph_emb)
        for l in range(self.L-1):
            graph_emb = self.FC_layers[l](graph_emb)
            graph_emb = self.activation(graph_emb)

        graph_emb = self.FC_layers[-1](graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return self.out_bn(pred), label



