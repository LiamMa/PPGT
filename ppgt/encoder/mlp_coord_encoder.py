import torch
from torch import nn
from torch_geometric.graphgym.register import register_edge_encoder, act_dict

from ..layer.utils.init import (trunc_init_, uniform_init_, xavier_normal_init_,
                                kaiming_normal_init_,
                                kaiming_normal_linear_init_,
                                default_init_, lecun_normal_init_
                                )





from ..layer.utils.norm import NormalizationLayer


#
#
@register_edge_encoder('MLPCoordEncoder')
class MLPCoordEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 bias=True,
                 add_to_edge=True,
                 index_name='edge',
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        self.pe_name = pe_name
        self.index_name = index_name

        # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)
        self.raw_norm = NormalizationLayer(kwargs.get('raw_norm', 'none'), in_dim)

        act_fn = act_dict[kwargs.get('act', 'gelu')]
        hid_dim= kwargs.get('hid_dim', '')
        if hid_dim == "":
            self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            hid_dim = [int(i) for i in hid_dim.split(',')] + [out_dim]
            fc = [nn.Linear(in_dim, hid_dim[0])]
            for i in range(len(hid_dim)-1):
                fc += [
                    act_fn(),
                    nn.Linear(hid_dim[i], hid_dim[i+1])
                ]
            self.fc = nn.Sequential(*fc)



        self.to_complete = kwargs.get('to_complete_graph', False)

        # post_norm = kwargs.get('post_norm', 'none')
        # self.post_norm = act_dict[post_norm](out_dim)
        self.post_norm = NormalizationLayer(kwargs.get('post_norm', 'none'), out_dim)

        # self.post_norm_type = 0
        # if 'kernel' in post_norm:
        #     self.post_norm_type = 1

        self.add_to_edge = add_to_edge

        self.kwargs= kwargs

        self.default_init = kwargs.get('default_init', False)
        self.trunc_init = kwargs.get('trunc_init', False)
        self.kaiming_linear_init = kwargs.get('kaiming_linear_init', False)
        self.uniform_init = kwargs.get('uniform_init', False)
        self.kaiming_init = kwargs.get('kaiming_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        # self.init_weights()


    def forward(self, batch):

        edge_index, edge_attr = batch[f'{self.index_name}_index'], batch.get(f'{self.index_name}_attr', None)

        coord_attr = batch[f'{self.pe_name}']
        coord_attr = coord_attr[edge_index[1]] - coord_attr[edge_index[0]]

        coord_attr = self.raw_norm(coord_attr, batch.batch[edge_index[1]])
        coord_attr = self.fc(coord_attr)
        coord_attr = self.post_norm(coord_attr, batch.batch[edge_index[1]])

        if edge_attr is not None:
            edge_attr = edge_attr + coord_attr
        else:
            edge_attr = coord_attr


        batch[f'{self.index_name}_index'], batch[f'{self.index_name}_attr'] = edge_index, edge_attr


        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name}, to_complete={self.to_complete})'

    def init_weights(self):
        if self.xavier_init:
            self.apply(xavier_normal_init_)

        elif self.trunc_init:
            self.apply(trunc_init_)

        elif self.kaiming_linear_init:
            self.apply(kaiming_normal_linear_init_)

        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)

        else:
            # if self.default_init:
            self.apply(default_init_)






@register_edge_encoder('MLPCoordSinEncoder')
class MLPCoordSinEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 bias=True,
                 add_to_edge=True,
                 index_name='edge',
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        self.pe_name = pe_name
        self.index_name = index_name

        # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)
        self.raw_norm = NormalizationLayer(kwargs.get('raw_norm', 'none'), in_dim)

        act_fn = act_dict[kwargs.get('act', 'gelu')]
        hid_dim= kwargs.get('hid_dim', '')
        if hid_dim == "":
            self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            hid_dim = [int(i) for i in hid_dim.split(',')] + [out_dim]
            fc = [nn.Linear(in_dim, hid_dim[0])]
            for i in range(len(hid_dim)-1):
                fc += [
                    act_fn(),
                    nn.Linear(hid_dim[i], hid_dim[i+1])
                ]
            self.fc = nn.Sequential(*fc)



        self.to_complete = kwargs.get('to_complete_graph', False)

        # post_norm = kwargs.get('post_norm', 'none')
        # self.post_norm = act_dict[post_norm](out_dim)
        self.post_norm = NormalizationLayer(kwargs.get('post_norm', 'none'), out_dim)

        # self.post_norm_type = 0
        # if 'kernel' in post_norm:
        #     self.post_norm_type = 1

        self.add_to_edge = add_to_edge

        self.kwargs= kwargs

        self.default_init = kwargs.get('default_init', False)
        self.trunc_init = kwargs.get('trunc_init', False)
        self.kaiming_linear_init = kwargs.get('kaiming_linear_init', False)
        self.uniform_init = kwargs.get('uniform_init', False)
        self.kaiming_init = kwargs.get('kaiming_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        # self.init_weights()


    def forward(self, batch):

        edge_index, edge_attr = batch[f'{self.index_name}_index'], batch.get(f'{self.index_name}_attr', None)

        coord_attr = batch[f'{self.pe_name}']
        coord_attr = coord_attr[edge_index[1]] - coord_attr[edge_index[0]]

        coord_attr = self.raw_norm(coord_attr, batch.batch[edge_index[1]])
        coord_attr = self.fc(coord_attr)
        coord_attr = self.post_norm(coord_attr, batch.batch[edge_index[1]])

        if edge_attr is not None:
            edge_attr = edge_attr + coord_attr
        else:
            edge_attr = coord_attr


        batch[f'{self.index_name}_index'], batch[f'{self.index_name}_attr'] = edge_index, edge_attr


        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name}, to_complete={self.to_complete})'

    def init_weights(self):
        if self.xavier_init:
            self.apply(xavier_normal_init_)

        elif self.trunc_init:
            self.apply(trunc_init_)

        elif self.kaiming_linear_init:
            self.apply(kaiming_normal_linear_init_)

        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)

        else:
            # if self.default_init:
            self.apply(default_init_)

