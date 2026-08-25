from functools import partial
from torch_geometric.utils import scatter

from typing import Optional

from torch import Tensor

import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import segment
from torch_geometric.utils.num_nodes import maybe_num_nodes


scatter_add = partial(scatter, reduce='add')
scatter_max = partial(scatter, reduce='max')
scatter_mul = partial(scatter, reduce='mul')






def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
    clamp: float=None,
    return_logit=False,
    zero_token=None,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:
        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """


    if (ptr is not None and src.device.type == 'cpu'
            and torch_geometric.typing.WITH_SOFTMAX
            and not is_compiling()):  # pragma: no cover

        if zero_token is not None:
            raise NotImplementedError('Not imlemented zero_token for using ptr-cpu')
        return pyg_lib.ops.softmax_csr(src, ptr, dim)

    if (ptr is not None and torch_geometric.typing.WITH_TORCH_SCATTER
            and not is_compiling()):
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')

        if zero_token is not None:
            zero_token = zero_token - src_max

        src_max = src_max.repeat_interleave(count, dim=dim)
        logit = src - src_max
        out = logit.exp() if clamp is None else logit.clamp_min(-abs(clamp)).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        if zero_token is not None:
            out_sum += zero_token.exp()

        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        if zero_token is not None:
            zero_token = zero_token - src_max

        logit = src - src_max.index_select(dim, index)
        if clamp is not None:
            logit = logit.clamp_min(-abs(clamp))
        out = logit.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        if zero_token is not None:
            out_sum += zero_token.exp()
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError("'softmax' requires 'index' to be specified")

    if return_logit:
        return out / out_sum, logit

    return out / out_sum


# def pyg_softmax(src, index, num_nodes=None):
#     r"""Computes a sparsely evaluated softmax.
#     Given a value tensor :attr:`src`, this function first groups the values
#     along the first dimension based on the indices specified in :attr:`index`,
#     and then proceeds to compute the softmax individually for each group.
#
#     Args:
#         src (Tensor): The source tensor.
#         index (LongTensor): The indices of elements for applying the softmax.
#         num_nodes (int, optional): The number of nodes, *i.e.*
#             :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
#
#     :rtype: :class:`Tensor`
#     """
#
#     num_nodes = maybe_num_nodes(index, num_nodes)
#
#     max_score = scatter_max(src, index, dim=0, dim_size=num_nodes)
#     # scatter_max in different version of torch_scatter  might return either tuple or matrix.
#     max_score = max_score[0] if isinstance(max_score, tuple) else max_score
#     out = src - max_score[index]
#     out = out.exp()
#     out = out / (
#             scatter_add(out, index, dim=0, dim_size=num_nodes) + 1e-16)[index]
#
#     return out



def pyg_density(src, index, bias=-1e8, batch_index=None, ptr=None, num_nodes=None):
    raise NotImplementedError('Disabled for now. To use, please double check and enable manually')


#     r"""
#         Based on Softmax; allow estimate the density for out of support points
#     """
#     # if bias == -1e16 (default) --> this is equivalent to the standard softmax
#     assert batch_index is not None, 'batch_index should be provided'
#     if ptr is not None:
#         graph_order = ptr[1:] - ptr[:-1]
#     else:
#         graph_order = scatter(torch.ones_like(batch_index),
#                               batch_index, dim=0, dim_size=num_nodes,
#                               reduce='sum')
#
#     shapes = [-1] + [1] * (src.dim() - 1)
#     graph_order = graph_order.view(*shapes)[batch_index]
#
#     num_nodes = maybe_num_nodes(index, num_nodes)
#     src_max = scatter(src, index, dim=0, dim_size=num_nodes, reduce='max')
#     out = src - src_max[index] + bias
#     bias = bias - src_max # estimate the density out of support points (score=0)
#     out = out.exp()
#     # ----
#     with torch.no_grad():
#         support_size = scatter(torch.ones_like(out[:, :1]), index,
#                                dim=0, dim_size=num_nodes, reduce='sum')
#
#     scale_term = scatter(out, index, dim=0, dim_size=num_nodes, reduce='add')
#     scale_term = scale_term + ((graph_order - support_size) * torch.exp(bias))[batch_index]
#
#     out = out / (scale_term + 1e-16)[index]
#
#     return out
#
#     def __repr__(self):
#         return f'{super().__repr__()}(rezero={self.rezero}, layer_scale={self.layer_scale}, layer_scale_init={self.layer_scale_init}, dim={self.dim})'
