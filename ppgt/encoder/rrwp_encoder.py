'''
    The RRWP encoder for GRIT (ours)
'''
import torch
from torch import nn
# import torch_sparse

import torch_geometric as pyg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN

from torch_geometric.utils import add_remaining_self_loops
# from torch_scatter import scatter
from torch_geometric.utils import scatter
import warnings


def full_edge_index(edge_index, batch=None):
    """
    Retunr the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter(one, batch,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device=edge_index.device)

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        negative_index_list.append(_edge_index + cum_nodes[i])

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()
    return edge_index_full




@register_node_encoder('rrwp_linear')
class RRWPLinearNodeEncoder(torch.nn.Module):
    """
        FC_1(RRWP) + FC_2 (Node-attr)
        note: FC_2 is given by the Typedict encoder of node-attr in some cases
        Parameters:
        num_classes - the number of classes for the embedding mapping to learn
    """
    def __init__(self, out_dim, use_bias=False, batchnorm=False, layernorm=False, pe_name="rrwp"):
        super().__init__()

        emb_dim = cfg.posenc_RRWP.ksteps
        if cfg.posenc_RRWP.add_attr:
            emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1)

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        self.raw_batchnorm = cfg.posenc_RRWP.get('raw_batchnorm', False)
        if self.raw_batchnorm:
            self.raw_bn = nn.BatchNorm1d(emb_dim)

        # post-fc BN
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        rrwp = batch[f"{self.name}"]
        if self.raw_batchnorm:
            rrwp = self.raw_bn(rrwp)
        rrwp = self.fc(rrwp)

        if self.batchnorm:
            rrwp = self.bn(rrwp)

        if self.layernorm:
            rrwp = self.ln(rrwp)

        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        return batch


@register_edge_encoder('rrwp_linear')
class RRWPLinearEdgeEncoder(torch.nn.Module):
    '''
        Merge RRWP with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False,
                 pe_name='rrwp'):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        emb_dim = cfg.posenc_RRWP.ksteps + 1
        if cfg.posenc_RRWP.add_attr:
            emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1) + 1

        self.pe_name = pe_name
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if 'posenc' not in cfg.gt: cfg.gt.posenc = CN()
        self.batch_norm = cfg.gt.posenc.get('batch_norm', False)
        self.layer_norm = cfg.gt.posenc.get('layer_norm', False)
        self.raw_norm = cfg.gt.posenc.get('raw_norm', False)

        if self.raw_norm:
            if self.batch_norm:
                norm_fn = nn.BatchNorm1d
            elif self.layer_norm:
                norm_fn = nn.LayerNorm
            else:
                norm_fn = nn.Identity

            self.norm_layer = norm_fn(emb_dim)
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
        else:
            self.norm_layer = nn.Identity()

    def forward(self, batch):
        with torch.no_grad():
            rrwp_idx = batch[f'{self.pe_name}_index']
            rrwp_val = batch[f'{self.pe_name}_attr']

            out_idx, out_val = add_remaining_self_loops(rrwp_idx, rrwp_val, num_nodes=batch.num_nodes, fill_value=0.)

            if self.pad_to_full_graph:
                edge_index_full = full_edge_index(out_idx, batch=batch.batch)
                edge_attr_pad = out_val.new_zeros(edge_index_full.size(1), out_val.size(1))
                # zero padding to fully-connected graphs
                out_idx = torch.cat([out_idx, edge_index_full], dim=1)
                out_val = torch.cat([out_val, edge_attr_pad], dim=0)

            # out_idx, out_val = torch_sparse.coalesce(
            #     out_idx, out_val, batch.num_nodes, batch.num_nodes,
            #     op="add"
            # )
            out_idx, out_val = pyg.utils.coalesce(out_idx, out_val, num_nodes=batch.num_nodes, reduce='sum')

        if self.raw_norm:
            out_val = self.norm_layer(out_val)

        out_val = self.fc(out_val)

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), out_val.size(1))
            # zero padding for non-existing edges

        if not self.overwrite_old_attr:
            out_idx, out_val = torch.cat([edge_index, out_idx], dim=1), torch.cat([edge_attr, out_val], dim=0)

        if 'pos' in batch:
            pos_dst, pos_src = batch.pos[out_idx[1]], batch.pos[out_idx[0]]
            out_val = out_val + self.fc_pos_dst(pos_dst) + self.fc_pos_src(pos_src)

        # out_idx, out_val = torch_sparse.coalesce(
        #     out_idx, out_val, batch.num_nodes, batch.num_nodes,
        #     op="add"
        # )
        out_idx, out_val = pyg.utils.coalesce(out_idx, out_val, num_nodes=batch.num_nodes, reduce='sum')
        out_idx, out_val = pyg.utils.coalesce(
            out_idx, out_val, num_nodes=batch.num_nodes,
            reduce="add"
        )

        with torch.no_grad():
            if 'spd_attr' in batch:
                spd_val, spd_idx = batch.spd_val + 1, batch[f'{self.pe_name}_index']
                # out_idx must cover all spd_idx
                spd_idx, spd_val = torch.cat([out_idx, spd_idx], dim=1), torch.cat([out_idx.new_zeros(out_idx.size(1)), spd_val], dim=0)
                _, spd_val = pyg.utils.coalesce(spd_idx, spd_val, num_nodes=batch.num_nodes, reduce='add')
                spd_val = spd_val - 1
                batch.spd_val = spd_val.type(torch.long)


        if not self.raw_norm:
            out_val = self.norm_layer(out_val)

        batch.edge_index, batch.edge_attr = out_idx, out_val
        batch.bias = self.fc.bias

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"



#
# @register_edge_encoder('masked_rrwp_linear')
# class RRWPLinearEdgeMaskedEncoder(torch.nn.Module):
#     '''
#         Merge RRWP with given edge-attr and Zero-padding to all pairs of node
#         FC_1(RRWP) + FC_2(edge-attr)
#         - FC_2 given by the TypedictEncoder in same cases
#         - Zero-padding for non-existing edges in fully-connected graph
#         - (optional) add node-attr as the E_{i,i}'s attr
#             note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     '''
#     def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False,
#                  mask_index_name="edge_index",
#                  ):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#         self.mask_index_name = mask_index_name
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         rrwp_idx = batch.rrwp_index
#         rrwp_val = batch.rrwp_val
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#         rrwp_val = self.fc(rrwp_val)
#         mask_index = batch.get(self.mask_index_name, None)
#         num_nodes = batch.num_nodes
#
#         if edge_attr is None:
#             edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
#             # zero padding for non-existing edges
#
#         if self.overwrite_old_attr:
#             out_idx, out_val = rrwp_idx, rrwp_val
#         else:
#             out_idx, out_val = torch_sparse.coalesce(
#                 torch.cat([edge_index, rrwp_idx], dim=1),
#                 torch.cat([edge_attr, rrwp_val], dim=0),
#                 batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#
#         if mask_index is not None:
#             mask_index, _ = add_remaining_self_loops(mask_index, None, num_nodes=batch.num_nodes)
#             mask_val = mask_index.new_full((mask_index.size(1), ), 1)
#             mask_comp = mask_index.new_full((out_idx.size(1), ), 0)
#             mask_pad = mask_index.new_full((mask_index.size(1), out_val.size(1)), 0)
#             _, masking = torch_sparse.coalesce(
#                 torch.cat([mask_index, out_idx], dim=1),
#                 torch.cat([mask_val, mask_comp], dim=0),
#                 m=num_nodes, n=num_nodes,
#                 op="max",
#             )
#             out_idx, out_val = torch_sparse.coalesce(
#                 torch.cat([mask_index, out_idx], dim=1),
#                 torch.cat([mask_pad, out_val], dim=0),
#                 batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#             masking = masking.type(torch.bool)
#             out_idx, out_val = out_idx[:, masking], out_val[masking]
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}," \
#                f"{self.fc.__repr__()})"
#



#
# @register_edge_encoder('rrwp_linear_cond')
# class RRWPLinearEdgeEncoder(torch.nn.Module):
#     '''
#         Merge RRWP with given edge-attr and Zero-padding to all pairs of node
#         FC_1(RRWP) + FC_2(edge-attr)
#         - FC_2 given by the TypedictEncoder in same cases
#         - Zero-padding for non-existing edges in fully-connected graph
#         - (optional) add node-attr as the E_{i,i}'s attr
#             note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     '''
#     def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  pad_to_full_graph=True, fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False,
#                  pe_name='rrwp'):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         self.pe_name = pe_name
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc = nn.Linear(emb_dim, out_dim * 2, bias=True)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         self.pad_to_full_graph = pad_to_full_graph
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         with torch.no_grad():
#             rrwp_idx = batch[f'{self.pe_name}_index']
#             rrwp_val = batch[f'{self.pe_name}_attr']
#             out_idx, out_val = rrwp_idx, rrwp_val
#
#             if self.pad_to_full_graph:
#                 edge_index_full = full_edge_index(out_idx, batch=batch.batch)
#                 edge_attr_pad = out_val.new_zeros(edge_index_full.size(1), out_val.size(1))
#                 # zero padding to fully-connected graphs
#                 out_idx = torch.cat([out_idx, edge_index_full], dim=1)
#                 out_val = torch.cat([out_val, edge_attr_pad], dim=0)
#                 out_idx, out_val = torch_sparse.coalesce(
#                     out_idx, out_val, batch.num_nodes, batch.num_nodes,
#                     op="add"
#                 )
#
#         out_val = self.fc(out_val)
#         out_val_mul, out_val_bias = out_val[:, :self.out_dim], out_val[:, self.out_dim:]
#
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#
#         if edge_attr is None:
#             edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
#             # zero padding for non-existing edges
#
#         edge_index, edge_attr = torch_sparse.coalesce(
#             torch.cat([edge_index, out_idx], dim=1),
#             torch.cat([edge_attr, torch.zeros_like(out_val_mul)], dim=0),
#             batch.num_nodes, batch.num_nodes,
#             op="add"
#         )
#
#         out_val = edge_attr * out_val_mul + out_val_bias
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}," \
#                f"{self.fc.__repr__()})"
#
#
# @register_edge_encoder('rrwp_linear_dual')
# class RRWPLinearEdgeEncoder(torch.nn.Module):
#     '''
#         Consider the RRWP not only A to B but also B to A
#     '''
#     def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  pad_to_full_graph=True, fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False,
#                  pe_name='rrwp'):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         self.pe_name = pe_name
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc1 = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         self.fc2 = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc1.weight)
#         torch.nn.init.xavier_uniform_(self.fc2.weight)
#
#         self.pad_to_full_graph = pad_to_full_graph
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         rrwp_idx = batch[f'{self.pe_name}_index']
#         rrwp_val = batch[f'{self.pe_name}_attr']
#         rrwp_val1 = self.fc1(rrwp_val)
#         rrwp_val2 = self.fc2(rrwp_val)
#
#         out_idx, out_val = torch_sparse.coalesce(
#             torch.cat([rrwp_idx, torch.stack([rrwp_idx[1], rrwp_idx[0]], dim=0)], dim=1),
#             torch.cat([rrwp_val1, rrwp_val2], dim=0),
#             batch.num_nodes, batch.num_nodes,
#             op='add'
#         )
#
#
#
#         if not self.overwrite_old_attr:
#             edge_index = batch.edge_index
#             edge_attr = batch.edge_attr
#             if edge_attr is None:
#                 edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
#                 # zero padding for non-existing edges
#             edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
#             out_idx, out_val = torch_sparse.coalesce(
#                 torch.cat([edge_index, out_idx], dim=1),
#                 torch.cat([edge_attr, out_val], dim=0),
#                 batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#
#         if self.pad_to_full_graph:
#             edge_index_full = full_edge_index(out_idx, batch=batch.batch)
#             edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
#             # zero padding to fully-connected graphs
#             out_idx = torch.cat([out_idx, edge_index_full], dim=1)
#             out_val = torch.cat([out_val, edge_attr_pad], dim=0)
#             out_idx, out_val = torch_sparse.coalesce(
#                 out_idx, out_val, batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}"
#
#
# @register_node_encoder('rrwp_centroid_linear')
# class RRWPCentroidLinearNodeEncoder(torch.nn.Module):
#     """
#         FC_1(RRWP) + FC_2 (Node-attr)
#         note: FC_2 is given by the Typedict encoder of node-attr in some cases
#         Parameters:
#         num_classes - the number of classes for the embedding mapping to learn
#     """
#     def __init__(self, out_dim, use_bias=False, batchnorm=False, layernorm=False, pe_name="rrwp"):
#         super().__init__()
#
#         emb_dim = cfg.posenc_RRWP.ksteps
#         if cfg.posenc_RRWP.add_attr:
#             emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1)
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         self.name = pe_name
#
#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#
#         # post-fc BN
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#
#         # Encode just the first dimension if more exist
#         rrwp = batch[f"{self.name}_centroid"]
#         rrwp = self.fc(rrwp)
#
#         if self.batchnorm:
#             rrwp = self.bn(rrwp)
#
#         if self.layernorm:
#             rrwp = self.ln(rrwp)
#
#         if "x" in batch:
#             batch.x = batch.x + rrwp
#         else:
#             batch.x = rrwp
#
#         return batch


#
# @register_edge_encoder('rrwp_linear_rwse')
# class RRWP_RWSELinearEdgeEncoder(torch.nn.Module):
#     '''
#         Merge RRWP with given edge-attr and Zero-padding to all pairs of node
#         FC_1(RRWP) + FC_2(edge-attr)
#         - FC_2 given by the TypedictEncoder in same cases
#         - Zero-padding for non-existing edges in fully-connected graph
#         - (optional) add node-attr as the E_{i,i}'s attr
#             note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     '''
#     def __init__(self, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  pad_to_full_graph=True, fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False,
#                  pe_name='rrwp',
#                  abs_pe_name='rrwp'):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         emb_dim = cfg.posenc_RRWP.ksteps + 1
#         if cfg.posenc_RRWP.add_attr:
#             emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1) + 1
#
#         self.pe_name = pe_name
#         self.abs_pe_name = abs_pe_name
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#
#         self.fc_src = nn.Linear(emb_dim-1, out_dim, bias=False)
#         torch.nn.init.xavier_uniform_(self.fc_src.weight)
#         self.fc_dest = nn.Linear(emb_dim-1, out_dim, bias=False)
#         torch.nn.init.xavier_uniform_(self.fc_dest.weight)
#
#         self.pad_to_full_graph = pad_to_full_graph
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         rrwp_idx = batch[f'{self.pe_name}_index']
#         rrwp_val = batch[f'{self.pe_name}_attr']
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#         rrwp_val = self.fc(rrwp_val)
#
#         if edge_attr is None:
#             edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
#             # zero padding for non-existing edges
#
#         if self.overwrite_old_attr:
#             out_idx, out_val = rrwp_idx, rrwp_val
#         else:
#             edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
#             out_idx, out_val = torch_sparse.coalesce(
#                 torch.cat([edge_index, rrwp_idx], dim=1),
#                 torch.cat([edge_attr, rrwp_val], dim=0),
#                 batch.num_nodes, batch.num_nodes,
#                 op="add"
#             )
#
#         if self.pad_to_full_graph:
#             edge_index_full = full_edge_index(out_idx, batch=batch.batch)
#             edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
#             # zero padding to fully-connected graphs
#             out_idx = torch.cat([out_idx, edge_index_full], dim=1)
#             out_val = torch.cat([out_val, edge_attr_pad], dim=0)
#             out_idx, out_val = torch_sparse.coalesce(
#                out_idx, out_val, batch.num_nodes, batch.num_nodes,
#                op="add"
#             )
#
#         rwse = batch[f'{self.abs_pe_name}']
#         src, dest = rwse[out_idx[0]], rwse[out_idx[1]]
#         out_val = out_val + self.fc_src(src) + self.fc_dest(dest)
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}," \
#                f"{self.fc.__repr__()})"
#
#
#
#
# @register_edge_encoder('khop_rrwp_linear')
# class KhopRRWPLinearEdgeEncoder(torch.nn.Module):
#     '''
#         Merge RRWP with given edge-attr and Zero-padding to all pairs of node
#         FC_1(RRWP) + FC_2(edge-attr)
#         - FC_2 given by the TypedictEncoder in same cases
#         - Zero-padding for non-existing edges in fully-connected graph
#         - (optional) add node-attr as the E_{i,i}'s attr
#             note: assuming  node-attr and edge-attr is with the same dimension after Encoders
#     '''
#     def __init__(self, out_dim, batchnorm=False, layernorm=False, use_bias=False,
#                  pad_to_full_graph=True, fill_value=0.,
#                  add_node_attr_as_self_loop=False,
#                  overwrite_old_attr=False,
#                  pe_name='rrwp'):
#         super().__init__()
#         # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
#         emb_dim = cfg.posenc_RRWP.ksteps + 1
#         if cfg.posenc_RRWP.add_attr:
#             emb_dim = cfg.posenc_RRWP.ksteps * (cfg.dataset.edge_encoder_num_types + 1) + 1
#
#         self.khop = cfg.posenc_RRWP.get('khop_attn', -1)
#         self.pe_name = pe_name
#         self.emb_dim = emb_dim
#         self.out_dim = out_dim
#         self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
#         self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
#
#         self.batchnorm = batchnorm
#         self.layernorm = layernorm
#         if self.batchnorm or self.layernorm:
#             warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")
#
#         self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
#         torch.nn.init.xavier_uniform_(self.fc.weight)
#         self.pad_to_full_graph = pad_to_full_graph
#         self.fill_value = 0.
#
#         padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
#         self.register_buffer("padding", padding)
#
#         self.raw_batchnorm = cfg.posenc_RRWP.get('raw_batchnorm', False)
#         if self.raw_batchnorm:
#             self.bn = nn.BatchNorm1d(emb_dim)
#
#         if self.batchnorm:
#             self.bn = nn.BatchNorm1d(out_dim)
#
#         if self.layernorm:
#             self.ln = nn.LayerNorm(out_dim)
#
#     def forward(self, batch):
#         rrwp_idx = batch[f'{self.pe_name}_index']
#         rrwp_val = batch[f'{self.pe_name}_attr']
#
#         if self.khop > 0:
#             mask = torch.sum(rrwp_val[:, 1:1+self.khop], dim=-1)
#             mask = mask > 0
#             rrwp_idx, rrwp_val = rrwp_idx[mask], rrwp_val[mask]
#
#         if self.raw_batchnorm:
#             rrwp_val = self.bn(rrwp_val)
#
#
#         edge_index = batch.edge_index
#         edge_attr = batch.edge_attr
#         rrwp_val = self.fc(rrwp_val)
#
#         if edge_attr is None:
#             edge_attr = torch.zeros(edge_index.size(1), rrwp_val.size(1))
#             # zero padding for non-existing edges
#
#         if self.overwrite_old_attr:
#             out_idx, out_val = rrwp_idx, rrwp_val
#         else:
#             out_idx, out_val = torch.cat([edge_index, rrwp_idx], dim=1), torch.cat([edge_attr, rrwp_val], dim=0)
#
#         out_idx, out_val = add_remaining_self_loops(out_idx, out_val, num_nodes=batch.num_nodes, fill_value=0.)
#         out_idx, out_val = torch_sparse.coalesce(
#             out_idx, out_val,
#             batch.num_nodes, batch.num_nodes,
#             op="add"
#         )
#
#         if self.batchnorm:
#             out_val = self.bn(out_val)
#
#         if self.layernorm:
#             out_val = self.ln(out_val)
#
#
#         batch.edge_index, batch.edge_attr = out_idx, out_val
#
#         return batch
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}" \
#                f"(pad_to_full_graph={self.pad_to_full_graph}," \
#                f"fill_value={self.fill_value}," \
#                f"{self.fc.__repr__()})"
