import torch
from torch_geometric.utils import scatter, coalesce
import torch_geometric as pyg









def full_edge_index(edge_index, batch_index=None, index_sorted=True, add_self_loops=True):
    """
    Return the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.

    > Note: this require the nodes in a batch is consecutive!!
    """
    device = edge_index.device

    if batch_index is None:
        batch_index = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch_index.max().item() + 1
    one = batch_index.new_ones(batch_index.size(0))
    num_nodes = scatter(one, batch_index,
                        dim=0, dim_size=batch_size, reduce='add')
    cum_nodes = torch.cat([batch_index.new_zeros(1), num_nodes.cumsum(dim=0)])

    negative_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        size = [n, n]
        adj = torch.ones(size, dtype=torch.short,
                         device='cpu')
        if not add_self_loops:
            adj = adj * (1 - torch.eye(size(0)))

        adj = adj.view(size)
        _edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        # _edge_index, _ = remove_self_loops(_edge_index)
        # no need to remove self-loop for Transformers
        negative_index_list.append(_edge_index + cum_nodes[i].item())

    edge_index_full = torch.cat(negative_index_list, dim=1).contiguous()

    # if the batch_index is not sorted, then, the edge_index_full need remapping
    if not index_sorted:
        _, map_index = torch.sort(batch_index)
        edge_index_full = map_index[edge_index_full]


    return edge_index_full.to(device)






def full_edge_index_new(edge_index, batch_index=None, index_sorted=True, add_self_loops=True, ptr=None):
    """
    Return the Full batched sparse adjacency matrices given by edge indices.
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index` while ignoring self-loops.
    Implementation inspired by `torch_geometric.utils.to_dense_adj`
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.

    > Note: this require the nodes in a batch is consecutive!!
    """
    device = edge_index.device

    if batch_index is None:
        if ptr is None:
            ptr = edge_index.max().item() + 1
        index_sorted = True # only with batch_index that index_sorted=False is supported

    if ptr is None:
        ptr = torch.cat([edge_index.new_zeros(1), torch.cumsum(batch_index.bincount(), dim=-1)], dim=0)

    num_nodes = ptr[1:] - ptr[:-1]
    num_batch = ptr.size(0) - 1

    full_index_list = []
    for i in range(num_nodes.size(0)):
        edge_index = torch.combinations(torch.arange(num_nodes[i]), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        if add_self_loops:
            edge_index, _ = pyg.utils.add_self_loops(edge_index, edge_attr=None, num_nodes=num_nodes[i].item())

        edge_index = edge_index + ptr[i].item()

        full_index_list.append(edge_index)

    edge_index_full = torch.cat(full_index_list, dim=1).contiguous()
    # if the batch_index is not sorted, then, the edge_index_full need remapping
    if not index_sorted:
        _, map_index = torch.sort(batch_index)
        edge_index_full = map_index[edge_index_full]


    return edge_index_full.to(device)


def to_complete_graph(edge_index, edge_attr=None, batch_index=None, num_nodes=None, index_sorted=True, add_self_loops=True, ptr=None):
    edge_index_full = full_edge_index_new(edge_index, batch_index=batch_index, index_sorted=index_sorted, add_self_loops=add_self_loops, ptr=ptr)
    if edge_attr is not None:
        edge_attr_pad = edge_attr.new_zeros(edge_index_full.size(1), edge_attr.size(1))
        # zero padding to fully-connected graphs
        edge_attr = torch.cat([edge_attr, edge_attr_pad], dim=0)

    edge_index = torch.cat([edge_index, edge_index_full], dim=1)
    edge_index, edge_attr = coalesce(edge_index, edge_attr,
                                     num_nodes=num_nodes,
                                     reduce='sum'
                                     )

    return edge_index, edge_attr