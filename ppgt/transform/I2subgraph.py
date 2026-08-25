import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data
# from torch_geometric.utils import scatter, scatter_add, scatter_max




# import torch_sparse
# from torch_sparse import SparseTensor


from .utils import dense_to_coo


@torch.no_grad()
def create_ego_subgraph(data,
                       k_hop=4,
                       pe_name=None,
                       pe_remap=False, # if True, map the PE to the subgraphs (remove the True edges)
                       **kwargs
                       ):
    '''
        ego-subgraph
    '''
    # if type(k_hop):
        # k_hop = [k_hop]

    assert (isinstance(data, Data))

    device=data.edge_index.device

    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight
    edge_attr = data.edge_attr
    if pe_remap:
        assert pe_name is not None, "[pe_name] cannot be None if [pe_remap]=True"


    # ------ RPE to APE2Root ----------
    if pe_name is not None:
        pe_index, pe_val = data[f'{pe_name}_index'], data[f'{pe_name}_attr']
        rpe = pyg.utils.to_dense_adj(pe_index, batch=None, edge_attr=pe_val,
                                     max_num_nodes=num_nodes
                                     ).squeeze(0)

        if edge_attr is not None:
            if kwargs.get('edge_to_one_hot', True):
                e_dim = kwargs.get('e_dim', -1)
                edge_attr = F.one_hot(edge_attr, num_classes=e_dim)

            e_feat = pyg.utils.to_dense_adj(edge_index, batch=None, edge_attr=edge_attr,
                                            max_num_nodes=num_nodes
                                            ).squeeze(0)

            rpe = torch.cat([e_feat, rpe], dim=-1) # N x N x D
            pe_index, pe_val = dense_to_coo(rpe)

    # aggregation is e[0]-->e[1] in GRIT
    sg_edge_index_ls = []
    sg_edge_attr_ls = []
    sg_raw_edge_index_ls = []
    sg_e_root_index = []
    sg_n_root_index = []
    sg_map_index = []
    sg_size = []
    sg_pe = []

    for root_node in range(num_nodes):


        subset, sg_edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(node_idx=root_node,
                                                                              num_hops=k_hop, edge_index=edge_index,
                                                                              relabel_nodes=True
                                                                              )
        sg_raw_edge_index = edge_index[:, edge_mask]


        if pe_remap:
            sg_edge_index, sg_edge_attr, edge_mask = pyg.utils.subgraph(subset=subset,
                                                              edge_index=pe_index, edge_attr=pe_val,
                                                              relabel_nodes=True,
                                                              return_edge_mask=True,
                                                              )
            sg_raw_edge_index = pe_index[:, edge_mask]

        sg_edge_index_ls.append(sg_edge_index)
        sg_edge_attr_ls.append(sg_edge_attr)
        sg_raw_edge_index_ls.append(sg_raw_edge_index)

        e_root_index = torch.Tensor([[root_node, root_node]] * sg_edge_index.size(1)).transpose(0, 1)
        n_root_index = torch.Tensor([[root_node, root_node]] * subset.size(0)).transpose(0, 1)
        sg_e_root_index.append(e_root_index)
        sg_n_root_index.append(n_root_index)

        sg_map_index.append(torch.stack([subset, subset], dim=0))
        sg_size.append(subset.size(0))
        if pe_name is not None:
            sg_pe.append(rpe[root_node][subset])


    data.sg_edge_idx = torch.cat(sg_edge_index_ls, dim=1).type(torch.long).transpose(0, 1)
    data.sg_edge_attr = torch.cat(sg_edge_attr_ls, dim=0)
    data.sg_raw_edge_index = torch.cat(sg_raw_edge_index_ls, dim=1).type(torch.long)
    # store as (E, 2) instead of (2, E)
    # we don't want the number changes by batching; therefore, using 'idx' instead of 'index', which will activate type-1 collate in PyG
    data.sg_map_index = torch.cat(sg_map_index, dim=1).type(torch.long)
    data.sg_e_root_index = torch.cat(sg_e_root_index, dim=1).type(torch.long)
    data.sg_n_root_index = torch.cat(sg_n_root_index, dim=1).type(torch.long)
    data.sg_size = torch.LongTensor(sg_size)
    data.sg_pe = torch.cat(sg_pe, dim=0)

    return data










@torch.no_grad()
def create_I2_subgraph(data,
                       k_hop=4,
                       pe_name=None,
                       add_self_loops=True,
                       **kwargs
                       ):
    '''
        Rooted by edges instead of nodes
    '''
    # if type(k_hop):
        # k_hop = [k_hop]

    assert (isinstance(data, Data))

    device = data.edge_index.device

    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight
    if add_self_loops: # add subgraphs for self-loops as well
        edge_index, _ = pyg.utils.add_self_loops(edge_index, num_nodes=num_nodes)


    adj = pyg.utils.to_dense_adj(edge_index, batch=None, edge_attr=edge_weight,
                                 max_num_nodes=num_nodes
                                 ).squeeze(0) # not batch

    # ------ RPE to APE2Root ----------
    if pe_name is not None:
        pe_index, pe_val = data[f'{pe_name}_index'], data[f'{pe_name}_attr']
        rpe = pyg.utils.to_dense_adj(pe_index, batch=None, edge_attr=pe_val,
                               max_num_nodes=num_nodes
                               ).squeeze(0)

    # aggregation is e[0]-->e[1] in GRIT
    I2_edge_index = []
    I2_e_map_index = []
    I2_n_map_index = []
    I2_x = []
    I2_count = []

    # hop_mask = torch.eye(adj.size(0))
    # for i in range(k_hop):
    #     hop_mask = hop_mask @ adj + adj
    #
    # hop_mask = (hop_mask > 0).type(torch.float)

    inc_sg = 0

    root = -1
    for e in edge_index.T:

        if e[0] != root:
            subset, sg_edge_index, mapping, edge_mask = pyg.utils.k_hop_subgraph(node_idx=[e[0]],
                                                                                 num_hops=k_hop, edge_index=edge_index,
                                                                                 relabel_nodes=True)
        root = e[0]

        # sg_adj = hop_mask[e[0]].view(-1, 1) * adj * hop_mask[e[0]].view(1, -1)

        if pe_name is not None:
            root_pe = rpe[e[0]][subset]
            neigh_pe = rpe[e[1]][subset]
            I2_x.append(torch.cat([root_pe, neigh_pe], dim=1))


        sg_edge_index_ = sg_edge_index + inc_sg
        inc_sg += len(subset)
        I2_edge_index.append(sg_edge_index_)
        I2_e_map_index.append(torch.stack([torch.ones_like(sg_edge_index_[0]) * e[0],
                                           torch.ones_like(sg_edge_index_[0]) * e[1]], dim=0))
        I2_n_map_index.append(torch.stack([torch.ones_like(subset) * e[0],
                                           torch.ones_like(subset) * e[1]], dim=0))
        # construct different index for subgraphs

    data.I2_edge_idx = torch.cat(I2_edge_index, dim=1).type(torch.long).transpose(0, 1)
    # Use 'idx' (and transpose) instead of 'index' to avoid auto-add incremental (the number of nodes per graph)
    data.I2_e_map_index = torch.cat(I2_e_map_index, dim=1).type(torch.long)
    data.I2_n_map_index = torch.cat(I2_n_map_index, dim=1).type(torch.long)
    # Use 'index' to align with edge_index
    data.I2_x = torch.cat(I2_x, dim=0)

    return data









@torch.no_grad()
def create_I2_SE(data,
                 k_hop=6,
                 enc_step=6,
                 pe_name=None,
                 add_self_loops=False,
                 **kwargs
                 ):
    '''
        SE on I2-Subgraphs via GIN-like aggregation
    '''
    # if type(k_hop):
        # k_hop = [k_hop]

    assert (isinstance(data, Data))

    device=data.edge_index.device

    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight
    if add_self_loops:
        edge_index = pyg.utils.add_self_loops(edge_index, num_nodes=num_nodes)


    adj = pyg.utils.to_dense_adj(edge_index, batch=None, edge_attr=edge_weight,
                                 max_num_nodes=num_nodes
                                 ).squeeze(0) # not batch

    # ------ RPE to APE2Root ----------
    # if pe_name is not None:
    #     pe_index, pe_val = data[f'{pe_name}_index'], data[f'{pe_name}_attr']
    #     rpe = pyg.utils.to_dense_adj(pe_index, batch=None, edge_attr=pe_val,
    #                            max_num_nodes=num_nodes
    #                            ).squeeze(0)

    # aggregation is e[0]-->e[1] in GRIT
    N = adj.size(0)
    adj = adj + torch.eye(N)
    k_mask = adj

    I2se_index= []
    I2se_attr = []

    k_mask_ls = [k_mask > 0]
    for k_hop in range(k_hop-1):
        k_mask = k_mask @ k_mask
        k_mask_ls.append(k_mask > 0)

    for e in edge_index.T:
        i,j = e
        if j < i: # symmetric encoding, no need to recompute
            continue
        k_se = []
        for k_mask in k_mask_ls:
            mask = k_mask[i] | k_mask[j]
            A = adj[:, mask][mask]
            enc = torch.log(1+GIN_Enc(A, enc_step))
            k_se.append(enc)

        se = torch.cat(k_se, dim=-1).flatten(0)
        I2se_index.append(e)
        I2se_attr.append(se)
        if i != j:
            e[0], e[1] = e[1], e[0]
            I2se_index.append(e)
            I2se_attr.append(se)


    I2se_index = torch.stack(I2se_index, dim=1)
    I2se_attr = torch.stack(I2se_attr, dim=0)

    I2se_index, I2se_attr = pyg.utils.coalesce(I2se_index, I2se_attr, num_nodes=N)

    data.I2se_index = I2se_index
    data.I2se_attr = I2se_attr

    return data

















def GIN_Enc(A, K):
    N = A.size(0)
    # A = A / A.size(0)

    D = A.sum(dim=1, keepdim=True)
    A = A / D * torch.log(D+1)

    out = []
    x = torch.ones(A.size(0))

    for i in range(1, K+1):
        x = A @ x
        out.append(x.mean(dim=0, keepdim=True) * np.log(1+N))

    enc = torch.cat(out, dim=0)

    return enc




