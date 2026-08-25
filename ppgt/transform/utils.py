

import torch


def dense_to_coo(adj: torch.Tensor):
    adj_ = torch.sum(adj.abs(), dim=-1)
    row, col = adj_.nonzero(as_tuple=True)
    val = adj[row, col]

    return row, col, val

