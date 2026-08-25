import torch
# from torch_scatter import scatter

from torch_geometric.graphgym.register import register_pooling


from typing import Optional

from torch import Tensor

from torch_geometric.utils import scatter


@register_pooling('cls_token')
def get_cls_tokens(x, cls_token):
    return cls_token


@register_pooling('cls_mask')
def get_cls_tokens_by_mask(x, cls_mask):
    emb = x[cls_mask]
    return emb



@register_pooling('log1p_sum_pool')
def global_log1p_sum_pool(x: Tensor, batch: Optional[Tensor],
                    size: Optional[int] = None, ptr: Optional[Tensor] = None) -> Tensor:
    r"""Returns batch-wise graph-level-outputs by log-summing node features
    across the node dimension.

    log1p(num_nodes) * mean(x)

    For a single graph :math:`\mathcal{G}_i`, its output is computed by

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.

    Functional method of the
    :class:`~torch_geometric.nn.aggr.SumAggregation` module.

    Args:
        x (torch.Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node to a specific example.
        size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)
    """
    dim = -1 if isinstance(x, Tensor) and x.dim() == 1 else -2

    if batch is None:
        return x.mean(dim=dim, keepdim=x.dim() <= 2) * torch.log1p(x.shape[0]).view(-1, 1)

    if ptr is not None:
        num_nodes = ptr[1:] - ptr[:-1]
    else:
        num_nodes = scatter(torch.ones_like(x[:, :1]), batch, dim=0, dim_size=size, reduce='sum')


    return scatter(x, batch, dim=dim, dim_size=size, reduce='mean') * torch.log1p(num_nodes).view(-1, 1)