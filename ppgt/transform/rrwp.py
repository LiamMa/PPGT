from typing import Any, Optional
import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from .fps import anchor_pe_fps

# from torch_geometric.utils import scatter, scatter_add, scatter_max




from torch_geometric.transforms import VirtualNode

import warnings


# import torch_sparse
# from torch_sparse import SparseTensor

add_vn = VirtualNode()


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data



@torch.no_grad()
def add_full_rrwp(data,
                  walk_length=8,
                  attr_name_abs="rrwp", # name: 'rrwp'
                  attr_name_rel="rrwp", # name: ('rrwp_idx', 'rrwp_val')
                  sym_norm:bool=False,
                  spd_mode:bool=False, # convert RRWP to as shortest-path distance
                  rwse_mode:bool=False, # only utilize the RWSE part 
                  add_cls_token:bool=False,
                  add_register_tokens:int=0,
                  dual_direction:bool=False,
                  **kwargs
                  ):
                

    add_identity = kwargs.get('add_identity', True)
    add_spd = kwargs.get('add_spd', False)
    scale_factor: float = kwargs.get('scale_factor', 1.)
    k_hop = kwargs.get('k_hop', -1)
    max_rd = kwargs.get('max_rd', -1)
    add_rd: bool = kwargs.get('add_rd', False)
    l2_renorm: bool = kwargs.get('l2_renorm', False)
    self_avoid_walk: bool = kwargs.get('self_avoid_walk', False)
    kp_encoding: bool = kwargs.get('kp_encoding', False)
    kp_order: int = kwargs.get('kp_order', 5)
    add_anchor: bool = kwargs.get('add_anchor', False)
    num_anchors: int = kwargs.get('num_anchors', 8)
    anchor_pe_dim: int = kwargs.get('anchor_pe_dim', 8)

    timesN = kwargs.get('timesN', False)
    log1p = kwargs.get('log1p', False)

    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = pyg.utils.to_dense_adj(edge_index, batch=None, edge_attr=edge_weight,
                                 max_num_nodes=num_nodes
                                 ).squeeze(0) # not batch


    if dual_direction:
        if (adj == adj.T).all():
            warnings.warn("The adjacency matrix is symmetric, cannot compute dual direction RRWP, convert to regular")
            # dual_direction = False


    if (adj != adj.T).any():
        if not dual_direction:
            warnings.warn("The adjacency matrix is not symmetric and not using dual direction RRWP")
            adj = ((adj + adj.T) / 2 > 0).type(torch.float)

    def _rrwp(adj):
        # symmetrize the adjacency matrix
        # Compute D^{-1} A:
        deg = adj.sum(dim=1)
        # store A and D for later use
        A = adj
        D = torch.diag(deg)

        if not sym_norm:
            deg_inv = 1.0 / adj.sum(dim=1)
            deg_inv[deg_inv == float('inf')] = 0
            adj = adj * deg_inv.view(-1, 1)
        else:
            deg_inv = 1.0 / deg.sqrt()
            deg_inv[deg_inv == float('inf')] = 0
            adj = adj * deg_inv.view(-1, 1) * deg_inv.view(1, -1)

        # adj = adj.to_dense()
        pe_list = []
        i = 0
        if add_identity:
            pe_list.append(torch.eye(num_nodes, dtype=torch.float))
            # i = i + 1

        mask = 1 - torch.eye(num_nodes, dtype=torch.float)

        out = adj
        pe_list.append(adj)

        kp_list = []


        if walk_length >= 2:
            for j in range(i + 1, walk_length):
            # for j in tqdm(range(i + 1, walk_length), desc="RRWP"):
                out = out @ adj
                pe_list.append(out)

            out = adj



        pe_list = pe_list

        pe = torch.stack(pe_list, dim=-1) # n x n x k
        if timesN: pe = pe * num_nodes / max(1, scale_factor)
        if log1p:
            pe = torch.log1p(pe)

        return pe, deg

    pe, deg = _rrwp(adj)
    if dual_direction:
        pe_rev, _ = _rrwp(adj.T)
        pe = torch.cat([pe, pe_rev[:, :, 1:]], dim=-1)

    pe_dim = pe.size(-1)

    abs_pe = pe.diagonal().transpose(0, 1)[:, :pe_dim] # n x k

    rel_pe_row, rel_pe_col, rel_pe_val = dense_to_coo(pe)
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0) # in GRIT, it is right matmul --> transpose the sparse matrix
    
    if rwse_mode:
        rel_pe_val = rel_pe_val * 0

    if spd_mode:
        # simple way to convert RRWP to SPD  (for demo only, not efficient method)
        mask = (rel_pe_val > 0).type(torch.float)
        first_nonzero = torch.argmax(mask, dim=-1, keepdim=True)
        spd_val = torch.zeros_like(rel_pe_val)
        spd_val.scatter_(-1, first_nonzero, 1)
        data.spd_attr = spd_val
        data.spd_index = rel_pe_idx


    if k_hop >= 0:
        mask = rel_pe_val[:, :k_hop+1].sum(dim=-1) > 0
        rel_pe_idx, rel_pe_val = rel_pe_idx[:, mask], rel_pe_val[mask]

    if max_rd >= 0:
        # compute RD based on (https://mathworld.wolfram.com/ResistanceDistance.html)
        mask = rel_pe_val[:, -1] <= max_rd
        if not add_rd:
            rel_pe_val = rel_pe_val[:, :-1]

        rel_pe_idx, rel_pe_val = rel_pe_idx[:, mask], rel_pe_val[mask]

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_attr")

    # if 'cls_mask' in data:
    #     deg[data.cls_mask] = sum(~data.cls_mask)

    data.log_deg = torch.log(deg + 1)
    data.deg_sqrt = torch.sqrt(deg)
    data.deg = deg.type(torch.long)

    sparsity = rel_pe_idx.size(1) / (num_nodes * num_nodes)
    data.sparsity = sparsity

    all_kp_enc = []
    if kp_encoding:
        pe = torch.cat([pe, torch.ones_like(pe[..., :1]) * (1/num_nodes)], dim=-1) # N x N x D
        # rel_pe_row, rel_pe_col, rel_pe_val = dense_to_coo(pe)
        # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
        # rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0) # in GRIT, it is right matmul --> transpose the sparse matrix
        # spd_val = (rel_pe_val > 0).type(torch.float)
        # spd_val = torch.argmax(spd_val, dim=-1)
        a_mask = (adj + torch.eye(adj.size(0)) > 0).type(torch.float)

        for kp in range(1, kp_order+1):
            kp_enc = []
            for i in range(num_nodes):
                pe_ = (a_mask[i].view(-1, 1) * a_mask[i].view(1, -1)).unsqueeze(-1) * pe
                kp_enc.append(pe_.sum(dim=0).sum(dim=0))
                # kp_enc.append(torch.sum(rel_pe_val[kp_mask], dim=0))

            a_mask = ((a_mask @ a_mask + a_mask) > 0).type(torch.float)
            all_kp_enc.append(torch.stack(kp_enc, dim=0))

        kp_enc = torch.cat(all_kp_enc, dim=-1)
        data.kp_rrwp = kp_enc



    if add_anchor:
        L = D - A
        N = D.size(0)
        Gamma = L + 1/N
        Ginv = np.linalg.pinv(Gamma, hermitian=True)
        diag_Ginv = np.diag(Ginv)
        RD = diag_Ginv.reshape(-1, 1) + diag_Ginv.reshape(1, -1)  - 2 * Ginv
        RD = torch.Tensor(RD).to(device)
        anchor_pe = anchor_pe_fps(RD, pe[:,:, :anchor_pe_dim], num_anchors)
        data.anchor_pe = anchor_pe
    

    if add_cls_token:
        data = add_vn(data)
        data.cls_mask = torch.zeros_like(data.x[:, 0]).bool()
        data.cls_mask[-1] = True

    if add_register_tokens > 0:
        for i in range(add_register_tokens):
            data = add_vn(data)
        data.register_mask = torch.zeros_like(data.x[:, 0]).bool()
        data.register_mask[-add_register_tokens:] = True

    return data



def dense_to_coo(adj: torch.Tensor):
    adj_ = torch.sum(adj.abs(), dim=-1)
    row, col = adj_.nonzero(as_tuple=True)
    val = adj[row, col]

    return row, col, val














