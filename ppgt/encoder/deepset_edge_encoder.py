import torch
from torch import nn

from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder, act_dict


import torch_geometric as pyg


@register_edge_encoder('DeepsetEdge')
class DeepsetEdgeEncoder(torch.nn.Module):
    '''
    For following LinearEdge or RRWPLinear
    '''
    def __init__(self, in_dim, out_dim, ffn_ratio=1., batchnorm=False):
        super().__init__()

        self.batchnorm = batchnorm
        act = cfg.gnn.act
        self.act = act_dict[act]()

        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

        # if self.batchnorm:
        #     self.batchnorm_e = nn.BatchNorm(emb_dim)
        # else:
        #     self.batchnorm_e = nn.Identity()

    def forward(self, batch):

        edge_attr = self.batchnorm_e(batch.edge_attr)
        if 'batch' not in batch:
            batch.batch = batch.e.new_zeros(batch.num_nodes)

        edge_batch = batch.batch[batch.edge_index[0]]
        pool_sum_attr = pyg.nn.global_add_pooling(edge_attr, edge_batch)
        unpool_sum_attr = pool_sum_attr[edge_batch]

        edge_attr = self.Gamma(edge_attr) - self.Lambda(unpool_sum_attr)
        batch.edge_attr = self.act(edge_attr)

        return batch



@register_edge_encoder('SetEquivFFNEdge')
class SetEquivFFNEdgeEncoder(torch.nn.Module):
    '''
    For following LinearEdge or RRWPLinear
    '''
    def __init__(self, in_dim, out_dim, ffn_ratio=1., num_block=1, batchnorm=False):
        super().__init__()

        self.batchnorm = batchnorm
        act = cfg.gnn.act
        self.act = act_dict[act]()

        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

        # if self.batchnorm:
        #     self.batchnorm_e = nn.BatchNorm(emb_dim)
        # else:
        #     self.batchnorm_e = nn.Identity()

    def forward(self, batch):

        edge_attr = self.batchnorm_e(batch.edge_attr)
        if 'batch' not in batch:
            batch.batch = batch.e.new_zeros(batch.num_nodes)

        edge_batch = batch.batch[batch.edge_index[0]]
        pool_sum_attr = pyg.nn.global_add_pooling(edge_attr, edge_batch)
        unpool_sum_attr = pool_sum_attr[edge_batch]

        edge_attr = self.Gamma(edge_attr) - self.Lambda(unpool_sum_attr)
        batch.edge_attr = self.act(edge_attr)

        return batch

class EquivSetLayer(torch.nn.Module):
    '''
    '''
    def __init__(self, in_dim, out_dim, ffn_ratio=1.):
        super().__init__()

        act = cfg.gnn.act
        self.act = act_dict[act]()

        hid_dim = int(max(in_dim, out_dim) * ffn_ratio)

        self.Gamma = nn.Linear(in_dim, hid_dim)
        self.Lambda = nn.Linear(in_dim, hid_dim, bias=False)

    def forward(self, batch):

        if 'batch' not in batch:
            batch.batch = batch.x.new_zeros(batch.num_nodes)

        edge_batch = batch.batch[batch.edge_index[0]]
        pool_sum_attr = pyg.nn.global_add_pooling(edge_attr, edge_batch)
        unpool_sum_attr = pool_sum_attr[edge_batch]

        edge_attr = self.Gamma(edge_attr) - self.Lambda(unpool_sum_attr)
        batch.edge_attr = self.act(edge_attr)

        return batch
