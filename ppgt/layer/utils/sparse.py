'''
    Alternative functions based on [PyG] without [torch_sparse];
    > for the case that [torch_sparse] is not available to install
'''

import torch


def dense_to_coo(adj: torch.Tensor):
    adj_ = torch.sum(adj.abs().flatten(2), dim=-1)
    row, col = adj_.nonzero(as_tuple=True)
    val = adj[row, col]

    return row, col, val




