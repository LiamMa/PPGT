'''
    Modules for Positional Encoder
'''
import torch
from torch import nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from yacs.config import CfgNode as CN






class PosEncoder(torch.nn.Module):
    """
        Encoding node and edge Positional Encoding
        Args:
            dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dim_edge=None, **kwargs):
        super().__init__()

        param = cfg.gt.pos_enc
        if dim_edge is None: dim_edge = dim_in

        node_enc_param = param.get('node_encoder', CN({'enable': False}))
        if node_enc_param.get('enable', False):
            encs = node_enc_param.get('name', "").split("+")
            pe_names = node_enc_param.get('pe_name', "").split("+")
            in_dims = node_enc_param.get('in_dim', "").split("+")
            in_dims = [int(dim) if dim != "" else -1 for dim in in_dims]
            if len(pe_names) < len(encs): pe_names += [''] * (len(encs) - len(pe_names))
            if len(in_dims) < len(encs): in_dims += [-1] * (len(encs) - len(in_dims))
            # lazy_init = node_enc_param.get('lazy_init', False)
            # if not lazy_init:
            #     in_dims = node_enc_param.get('in_dim', "").split("+")

            node_enc_param = copy_and_pop(node_enc_param, key_ls=['pe_name', 'in_dim'])
            node_encoders = [register.node_encoder_dict[encs[i]](
                in_dim=in_dims[i],
                out_dim=dim_in,
                pe_name=pe_names[i],
                **node_enc_param.get(encs[i], node_enc_param))
                for i in range(len(encs)) if encs[i] != ""]
            self.node_encoders = nn.Sequential(*node_encoders)



        # abs-coord to relative-pos
        coord_enc_param = param.get('coord_encoder', CN({'enable': False}))
        if coord_enc_param.get('enable', False):
            encs = coord_enc_param.get('name', "").split("+")
            pe_names = coord_enc_param.get('pe_name', "").split("+")
            in_dims = coord_enc_param.get('in_dim', "").split("+")
            in_dims = [int(dim) if dim != "" else -1 for dim in in_dims]
            if len(pe_names) < len(encs): pe_names += [''] * (len(encs) - len(pe_names))
            if len(in_dims) < len(encs): in_dims += [-1] * (len(encs) - len(in_dims))
            # lazy_init = coord_enc_param.get('lazy_init', False)

            coord_enc_param = copy_and_pop(coord_enc_param, key_ls=['pe_name', 'in_dim'])
            coord_encoders = [register.edge_encoder_dict[encs[i]](
                in_dim=in_dims[i],
                out_dim=coord_enc_param.get(encs[i], coord_enc_param).get('out_dim', dim_edge),
                pe_name=pe_names[i],
                **coord_enc_param.get(encs[i], coord_enc_param))
                for i in range(len(encs)) if encs[i] != ""]
            self.coord_encoders = nn.Sequential(*coord_encoders)

        sg_enc_param = param.get('subgraph_encoder', CN({'enable': False}))
        if sg_enc_param.get('enable', False):
            encs = sg_enc_param.get('name', "").split("+")
            pe_names = sg_enc_param.get('pe_name', "").split("+")
            in_dims = sg_enc_param.get('in_dim', "").split("+")
            in_dims = [int(dim) if dim != "" else -1 for dim in in_dims]
            if len(pe_names) < len(encs): pe_names += [''] * (len(encs) - len(pe_names))
            if len(in_dims) < len(encs): in_dims += [-1] * (len(encs) - len(in_dims))
            # lazy_init = coord_enc_param.get('lazy_init', False)

            sg_enc_param = copy_and_pop(sg_enc_param, key_ls=['pe_name', 'in_dim'])
            sg_encoders = [register.edge_encoder_dict[encs[i]](
                in_dim=in_dims[i],
                out_dim=sg_enc_param.get(encs[i], sg_enc_param).get('out_dim', dim_edge),
                pe_name=pe_names[i],
                **sg_enc_param.get(encs[i], sg_enc_param))
                for i in range(len(encs)) if encs[i] != ""]
            self.sg_encoders = nn.Sequential(*sg_encoders)

        edge_enc_param = param.get('edge_encoder', CN({'enable': False}))
        if edge_enc_param.get('enable', False):
            encs = edge_enc_param.get('name', "").split("+")
            pe_names = edge_enc_param.get('pe_name', "").split("+")
            if len(pe_names) < len(encs): pe_names += [''] * (len(encs) - len(pe_names))
            in_dims = edge_enc_param.get('in_dim', "").split("+")
            in_dims = [int(dim) if dim != "" else -1 for dim in in_dims]
            if len(in_dims) < len(encs): in_dims += [-1] * (len(encs) - len(in_dims))
            # lazy_init = edge_enc_param.get('lazy_init', False)
            # if not lazy_init:
            #     in_dims = edge_enc_param.get('in_dim', "").split("+")
            edge_enc_param = copy_and_pop(edge_enc_param, key_ls=['pe_name', 'in_dim'])
            edge_encoders = [register.edge_encoder_dict[encs[i]](
                in_dim=in_dims[i],
                out_dim=edge_enc_param.get(encs[i], edge_enc_param).get('out_dim', dim_edge),
                pe_name=pe_names[i],
                node_dim=dim_in,
                **edge_enc_param.get(encs[i], edge_enc_param))
                for i in range(len(encs)) if encs[i] != ""]
            self.edge_encoders = nn.Sequential(*edge_encoders)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


def copy_and_pop(param, key_ls=[]):
    '''
        copy the dictionary or CfgNode and pop unnecessary keys
    '''
    param = param.copy()
    for k in key_ls:
        param.pop(k)

    return param



class Backbone(torch.nn.Module):
    """Stack of transformer blocks, with optional jumping-knowledge fusion."""
    def __init__(self, layers, jk_layer=None, jk_feat='x', **kwargs):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.jk_layer = jk_layer
        self.jk_feat = jk_feat

    def forward(self, batch):

        jk = []

        for layer in self.layers:
            batch = layer(batch)
            if self.jk_layer is not None:
                jk.append(batch[self.jk_feat])

        x = self.jk_layer(jk)
        batch[self.jk_feat] = x

        return batch

