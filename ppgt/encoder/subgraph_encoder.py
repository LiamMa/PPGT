import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.graphgym.register import register_edge_encoder, act_dict
from torch_geometric.utils import scatter
from torch_geometric.nn import Sequential, GINConv

from ..layer.utils.residual import ResidualLayer


from .utils.full_graph import to_complete_graph


from timm.models.layers import trunc_normal_

def trunc_init_(m, std=1.):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@register_edge_encoder('I2GINEncoder')
class I2GINEncoder(torch.nn.Module):
    '''
        GIN encoder for I2-Subgraph
    '''
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 frozen=False,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')
        self.kwargs = kwargs

        num_layers = self.kwargs.get('num_layers', 5)

        self.graph_pool = self.kwargs.get('graph_pool', 'add')
        act_fn = act_dict[self.kwargs.get('act', 'gelu')]
        norm_fn = act_dict[self.kwargs.get('norm', 'batch_norm')]

        self.pre_fc = nn.Linear(2 * in_dim, out_dim, bias=True)

        sg_gnns = []
        for l in range(num_layers):
            mlp = nn.Sequential(nn.Linear(out_dim, out_dim), act_fn(), nn.Linear(out_dim, out_dim))
            sg_gnns += [
                (GINConv(mlp, train_eps=True), 'x, edge_index, size -> h'),
                (norm_fn(out_dim) if not frozen else nn.Identity(), 'h -> h'),
                (ResidualLayer(), 'h, x -> x'),
            ]

        self.sg_gnns = Sequential('x, edge_index, size', sg_gnns)


        self.to_complete = kwargs.get('to_complete_graph', True)
        self.reduce = kwargs.get('graph_pool', 'sum')
        # self.out_fc = nn.Linear(out_dim, out_dim, bias=True)

        self.pe_fc = nn.Linear(out_dim, out_dim, bias=False)

        raw_norm = kwargs.get('raw_norm', 'none') # raw norm post GNN-encoder and raw-norm for PE
        self.raw_norm = act_dict[raw_norm](2 * in_dim) if not frozen else nn.Identity()
        pool_norm = kwargs.get('pool_norm', 'none')
        self.pool_norm = act_dict[pool_norm](out_dim) if not frozen else nn.Identity()
        post_norm = kwargs.get('post_norm', 'none')
        self.post_norm = act_dict[post_norm](out_dim) if not frozen else nn.Identity()

        self.post_norm_type = 0
        if 'kernel' in post_norm:
            self.post_norm_type = 1

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        self.apply(trunc_init_)


    def forward(self, batch):

        with torch.no_grad():
            I2_edge_index, I2_e_map_index, I2_n_map_index = batch.I2_edge_idx.transpose(0, 1), batch.I2_e_map_index, batch.I2_n_map_index
            # get subgraph_edge_index incremental num for each graph
            batch_index = batch.batch[I2_e_map_index[0]]
            inc = torch.max(I2_edge_index, dim=0).values
            batch_inc = scatter(inc, batch_index, dim=0, dim_size=batch.num_graphs, reduce='max') + 1
            batch_ptr = torch.zeros_like(batch_inc)
            batch_ptr[1:] = torch.cumsum(batch_inc, dim=0)[:-1]
            I2_edge_index = I2_edge_index + batch_ptr[batch_index]


        I2_x = batch.I2_x
        assert I2_edge_index.max() + 1 == I2_x.size(0), (f'The I2_edge_index is not remap to correct label;'
                                                         f' expected to [I2_edge_index.max() + 1 == I2_x.size(0)];'
                                                         f' but get {I2_edge_index.max() + 1} and {I2_x.size(0)}')


        I2_x = self.pre_fc(self.raw_norm(I2_x))
        num_nodes = I2_x.size(0)

        # ----- subgraph GNN ----
        I2_x = self.sg_gnns(x=I2_x, edge_index=I2_edge_index, size=(num_nodes, num_nodes))


        # ----- subgraph pooling to edges ----
        pe_index, pe_val = pyg.utils.coalesce(I2_n_map_index, I2_x, num_nodes=batch.num_nodes,
                                                  reduce=self.reduce,
                                        )


        pe_val = self.pool_norm(pe_val)
        pe_val = self.pe_fc(pe_val)


        if self.to_complete:
            pe_index, pe_val = to_complete_graph(pe_index, pe_val, batch.batch, batch.num_nodes)


        if self.post_norm_type == 1:
            pe_val = self.post_norm(pe_val, pe_index[1])
        else:
            pe_val = self.post_norm(pe_val)

        # reorder to match the aggregation of GRIT
        pe_index = torch.stack([pe_index[1], pe_index[0]], dim=0)


        edge_index, edge_attr = batch.edge_index, batch.edge_attr
        if edge_attr is None:
            edge_attr = pe_val.new_zeros(edge_index.size(1), pe_val.size(1))

        self_loop_attr = pyg.utils.get_self_loop_attr(edge_index, edge_attr, num_nodes=batch.num_nodes)
        if 'x' in batch:
            batch.x = self_loop_attr + batch.x
        else:
            batch.x = self_loop_attr

        # if (edge_index.size(0) == pe_index.size(1)) and (edge_index == pe_index).all():
        #     edge_attr = edge_attr + pe_val
        # else:

        edge_index, edge_attr = pyg.utils.coalesce(torch.cat([edge_index, pe_index], dim=1),
                                                   torch.cat([edge_attr, pe_val], dim=0),
                                                   num_nodes=batch.num_nodes,
                                                   reduce='sum',
                                                   )



        batch.edge_index, batch.edge_attr = edge_index, edge_attr

        return batch

    # def __repr__(self):
    #     return f'{super().__repr__()}(num_layers={self.pe_name})'





@register_edge_encoder('GINSubgraphEncoder')
class SubGraphGNNEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')


        self.pe_name = pe_name

        # self.mlp = MLP(in_dim, out_dim, **kwargs)
        # self.inner_reduce = kwargs.get('inner_reduce', None) # first coalesce for PEs, than merge with edge_index/edge_attr
        # self.reduce = kwargs.get('reduce', 'add')

        self.kwargs = kwargs
        # self.kwargs= kwargs.get('sg_gnn', CN(dict()))
        # kwargs = self.kwargs

        num_layers = self.kwargs.get('num_layers', 5)

        self.graph_pool = self.kwargs.get('graph_pool', 'add')
        act_fn = act_dict[self.kwargs.get('act', 'gelu')]
        norm_fn = act_dict[self.kwargs.get('norm', 'batch_norm')]

        self.pre_fc = nn.Linear(in_dim, out_dim, bias=True)


        sg_gnns = []
        for l in range(num_layers):
            mlp = nn.Sequential(nn.Linear(out_dim, out_dim), act_fn(), nn.Linear(out_dim, out_dim))
            sg_gnns += [
                (GINConv(mlp, train_eps=True), 'x, edge_index, size -> h'),
                (norm_fn(out_dim), 'h -> h'),
                (ResidualLayer(), 'h, x -> x'),
            ]

        self.sg_gnns = Sequential('x, edge_index, size', sg_gnns)

        self.src_fc = nn.Linear(out_dim, out_dim)
        self.dst_fc = nn.Linear(out_dim, out_dim, bias=False)
        self.e_act = act_fn()
        self.e_fc = nn.Linear(out_dim, out_dim)

        self.out_n_norm = norm_fn(out_dim)
        self.out_e_norm = norm_fn(out_dim)
        # self.out_fc = nn.Linear(out_dim, out_dim, bias=True)



    def forward(self, batch):

        edge_index, edge_attr = batch.edge_index, batch.edge_attr

        sg_edge_index = batch.sg_edge_idx.transpose(0, 1)
        sg_map_index = batch.sg_map_index[0]
        sg_e_root_index = batch.sg_e_root_index[0]
        sg_n_root_index = batch.sg_n_root_index[0]
        sg_size = batch.sg_size
        sg_pe = batch.sg_pe

        # --------- Reset the edge_index to separate subgraphs ---------
        sg_size_ = torch.zeros_like(sg_size)
        sg_size_[1:] = torch.cumsum(sg_size, dim=0)[:-1]

        sg_edge_add = sg_size_[sg_e_root_index]
        sg_edge_index = sg_edge_index + sg_edge_add.unsqueeze(0)


        sg_pe = self.pre_fc(sg_pe)
        num_sg_nodes = sg_pe.size(0)
        # ----- subgraph GNN ----
        sg_x = self.sg_gnns(x=sg_pe, edge_index=sg_edge_index, size=(num_sg_nodes, num_sg_nodes))
        sg_e = self.e_fc(self.e_act(self.src_fc(sg_x[sg_edge_index[0]]) + self.dst_fc(sg_x[sg_edge_index[1]])))

        # ----- subgraph pooling ----
        x = scatter(sg_x, sg_n_root_index, dim=0, dim_size=batch.x.size(0), reduce=self.graph_pool)
        e = scatter(sg_e, sg_e_root_index, dim=0, dim_size=batch.x.size(0), reduce=self.graph_pool)

        batch.x = self.out_n_norm(x) + self.out_e_norm(e) + batch.x

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'

















#
#
#
#
#
#
#
#
# @register_edge_encoder('G2SGEncoder')
# class SubGraphGNNEncoder(torch.nn.Module):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  pe_name=None,
#                  **kwargs):
#         super().__init__()
#         # print(f'kwargs is {kwargs}')
#
#
#         self.pe_name = pe_name
#
#         # self.mlp = MLP(in_dim, out_dim, **kwargs)
#         # self.inner_reduce = kwargs.get('inner_reduce', None) # first coalesce for PEs, than merge with edge_index/edge_attr
#         # self.reduce = kwargs.get('reduce', 'add')
#
#         self.kwargs = kwargs
#         # self.kwargs= kwargs.get('sg_gnn', CN(dict()))
#         # kwargs = self.kwargs
#
#
#         self.sg_pe_fc = nn.Linear(in_dim, out_dim, bias=True)
#         self.root_pe_fc = nn.Linear(in_dim, out_dim, bias=True)
#
#
#
#     def _construct_subgraph_index(self, batch):
#         # store the original edge_index and edge_attr
#         edge_index, edge_attr = batch.edge_index, batch.edge_attr
#         batch.raw_edge_index, batch.raw_edge_attr = edge_index, edge_attr
#
#         # ----- reconstruct subgraph edge_index
#         sg_edge_index = batch.sg_edge_idx.transpose(0, 1)
#         sg_e_root_index = batch.sg_e_root_index[0]
#         sg_size = batch.sg_size
#
#         # --------- Reset the edge_index to separate subgraphs ---------
#         sg_size_ = torch.zeros_like(sg_size)
#         sg_size_[1:] = torch.cumsum(sg_size, dim=0)[:-1]
#
#         sg_edge_add = sg_size_[sg_e_root_index]
#         sg_edge_index = sg_edge_index + sg_edge_add.unsqueeze(0)
#
#         batch.sg_edge_index = sg_edge_index
#
#         return batch
#
#
#     def forward(self, batch):
#
#         sg_map_index = batch.sg_map_index[0]
#         sg_n_root_index = batch.sg_n_root_index[0]
#
#         # ----------- construct subgraph index
#         batch = self._construct_subgraph_index(batch)
#
#         # ----------- remap node-attr
#         x = batch.x[sg_map_index]
#         sg_pe = self.root_pe_fc(batch.sg_pe)
#         batch.x = sg_pe + x
#         batch.log_deg = batch.log_deg[sg_n_root_index]
#         batch.deg = batch.deg[sg_n_root_index]
#         batch.num_nodes = None
#
#         # ------------ remap edge-attr
#         # sg_raw_edge_index = batch.sg_raw_edge_index
#         edge_index = batch.sg_edge_index
#         edge_attr = batch.sg_edge_attr
#
#         edge_index_full = full_edge_index(batch.edge_index, batch=batch.sg_n_root_index[0])
#         edge_index, edge_attr = pyg.utils.coalesce(
#             torch.cat([edge_index, edge_index_full], dim=1),
#             torch.cat([edge_attr, edge_attr.new_zeros(edge_index_full.size(1), edge_attr.size(1))], dim=0),
#         )
#
#         batch.edge_index, batch.edge_attr = edge_index, self.sg_pe_fc(edge_attr)
#
#         # # >>> too time costly <<<
#         # batch.edge_index, edge_attr = batch.edge_index, batch.edge_attr
#         # e_map = dict()
#         # e_inv_map = dict()
#         # for i in range(edge_index.size(1)):
#         #     e = edge_index[:, i]
#         #     e_map[(e[0].item(), e[1].item())] = i
#         #
#         # sg_edge_attr = []
#         # for j in range(sg_raw_edge_index.size(1)):
#         #     e = sg_raw_edge_index[:, j]
#         #     index = e_map[(e[0].item(), e[1].item())]
#         #     sg_edge_attr.append(edge_attr[index])
#         #
#         #
#         # sg_edge_attr = torch.cat(sg_edge_attr, dim=0)
#
#         return batch
#
#     def __repr__(self):
#         return f'{super().__repr__()}(pe_name={self.pe_name})'