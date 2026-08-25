
from torch import nn

from torch_geometric.graphgym.register import act_dict

from ...act.norm import *






class NormalizationLayer(nn.Module):
    def __init__(self, norm_name, dim):
        super().__init__()

        self.norm_layer = act_dict[norm_name](dim)
        self.norm_index=None
        if 'graph' in norm_name or 'instance' in norm_name:
            # both 'graph' or 'instance' refer to normalization within each example
            self.norm_index = 'graph'

        # self.graph_norm = isinstance(norm_layer, pyg.nn.GraphNorm)
        # self.group_norm = isinstance(norm_layer, GraphGroupNorm)

    def forward(self, x, batch_index=None):

        if self.norm_index is None:
            y = self.norm_layer(x)

        elif self.norm_index == 'graph':
            y = self.norm_layer(x, batch_index)

        return y


class Batch2BatchNormalizationLayer(nn.Module):
    def __init__(self, norm_name, dim, attr_name='x'):
        super().__init__()
        self.attr_name = attr_name
        # self.norm_layer = act_dict[norm_name](dim)

        self.by_pass = False
        if norm_name == 'none' :
            self.by_pass = True

        self.norm_layer = NormalizationLayer(norm_name, dim)

    def forward(self, batch):

        if self.by_pass:
            return batch

        x = batch[self.attr_name]

        x = self.norm_layer(x, batch.batch)

        batch[self.attr_name] = x
        return batch

    def __repr__(self):
        return f'{super().__repr__()} (attr_name={self.attr_name})'
