
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
# from torch_geometric.utils import scatter, scatter_add, scatter_max
import copy









def add_isolated_vn(data: Data,
                    num_vn: int=1,
                    **kwargs
                    ) -> Data:
    r'''
    Add virtual nodes for each graph:
        - virtual node (VN)
        - VN_mask

    Note!!:
        - no edges added to save memory; need to be followed by ToCompleteGraph in Graph Transformers!
    '''
    num_nodes = data.num_nodes

    old_data = copy.copy(data)

    vn_feat = torch.stack([data.x[0]] * num_vn, dim=0)
    data.x = torch.cat([data.x, vn_feat], dim=0)

    if 'num_nodes' in data:
        data.num_nodes = num_nodes + num_vn

    data.vn_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.vn_mask[-num_vn:] = True
    data.vn_idx = torch.zeros_like(data.vn_mask, dtype=torch.long)

    for i in range(num_vn):
        data.vn_idx[-num_vn+i] = i

    for key, value in old_data.items():
        if key in ['x', 'edge_index', 'edge_type', 'vn_mask', 'vn_idx', 'num_nodes']:
            continue

        if isinstance(value, Tensor):
            dim = old_data.__cat_dim__(key, value)
            size = list(value.size())

            fill_value = None

            if value.size(0) == old_data.num_nodes:
                size[dim] = num_vn
                if key == 'deg':
                    fill_value = old_data.num_nodes
                elif key == 'log_deg':
                    fill_value = np.log(1 + old_data.num_nodes)
                else:
                    fill_value = 0.

            if fill_value is not None:
                new_value = value.new_full(size, fill_value)
                data[key] = torch.cat([value, new_value], dim=dim)

    return data