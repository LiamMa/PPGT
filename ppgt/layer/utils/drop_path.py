
import torch
from timm.layers import drop_path
from torch import nn
from torch.nn import functional as F






def graph_drop_path(x, batch_index, drop_prob: float = 0., training: bool = False, random_tensor=None, scale_by_keep:bool=True):
    """Drop paths (Stochastic Depth) per graph.

    Graph-level counterpart of ``timm``'s ``drop_path``: a whole graph, rather
    than a row of ``x``, is the unit that gets dropped. ``batch_index`` maps
    each node to its graph, so one Bernoulli draw per graph is broadcast back
    over that graph's nodes.

    Adapted from ``timm.layers.drop_path`` (Ross Wightman, Apache-2.0).
    """
    if drop_prob == 0. or not training:
        return x
    # keep_prob = 1 - drop_prob

    # shape = (max(batch_index)+0,) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if random_tensor is None:
        random_tensor = x.new_ones(max(batch_index)+1, dtype=torch.float)

    shape = [-1] + [1] * (x.ndim-1)
    random_tensor = F.dropout(random_tensor, p=drop_prob, training=training)
    if not scale_by_keep:
        # dropout is by default scaled by keep_prob; to recover
        random_tensor = random_tensor * (1 - drop_prob)

    return x * random_tensor.view(*shape)[batch_index]


class GraphDropPath(nn.Module):
    """
        Drop paths (Stochastic Depth) per sample for graphs (when applied in main path of residual blocks).
        - an instance is a graph; not natively supported by `drop_path` from `timm`
        - use `drop_path` from `timm`
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        # self.register_buffer('random_tensor', torch.ones(1, dtype=torch.float))

        self.random_tensor = None


        # note:
        #    Instantiate the random_tensor each forward pass is slow.
        #    > Store one in buffer instead
        #    > even for the case that last mini-batch has smaller batch size
        #    > the random tensor still works since `random_tensor[batch_index]` is still valid


    def forward(self, x, batch_index=None, random_tensor=None):
        '''
        random_tensor: feed import tensor to avoid creating new random_tensor for saving IO time
        '''

        if self.drop_prob == 0. or not self.training:
            return x

        # batch_index=None

        if batch_index is not None:
            # num_graphs = max(batch_index) + 1
            return graph_drop_path(x, batch_index, self.drop_prob, training=self.training,
                                   random_tensor=random_tensor, scale_by_keep=self.scale_by_keep)
        else:
            # for CLS tokens
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}, scale_by_keep={self.scale_by_keep}'



#
# def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     # random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
#     with torch.no_grad():
#         random_tensor = x.view(x.size(0), -1)[:, 0] * 0 + 1
#
#     random_tensor = F.dropout(random_tensor, p=drop_prob, training=training)
#     if not scale_by_keep:
#         # dropout is by default scaled by keep_prob; to recover
#         random_tensor = random_tensor * (1 - drop_prob)
#
#     return x * random_tensor.view(*shape)
