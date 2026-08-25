import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder, act_dict, register_node_encoder
from torch_geometric.utils import coalesce

from ..layer.utils.init import (trunc_init_, uniform_init_, xavier_normal_init_,
                                kaiming_uniform_init_,  kaiming_normal_init_,
                                apply_weight_norm, apply_spectral_norm,
                                kaiming_uniform_linear_init_, kaiming_normal_linear_init_,
                                default_init_, lecun_normal_init_
                                )
from ..layer.utils.mlp import MLP, SirenMLP
from ..network.initialization import init_weights_vit_timm
from ..network.utils import copy_and_pop

from .utils.full_graph import to_complete_graph

from functools import partial




from ..layer.utils.rbf import RBFLayer, RBFLayerCenter

from ..layer.utils.norm import NormalizationLayer



@register_edge_encoder('MLPEdgeEncoder')
class MLPEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 bias=True,
                 add_invN=False,
                 add_invD=False,
                 add_invDsqrt=False,
                 add_to_edge=True,
                 add_loc:bool=False,
                 loc_dim:int=2,
                 loc_name:str='loc',
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        self.pe_name = pe_name

        # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)
        self.raw_norm = NormalizationLayer(kwargs.get('raw_norm', 'none'), in_dim)

        if pe_name not in ['rrwp']:
            # only applied to positional encoding
            add_invN = False
            add_invD = False

        self.add_invN= add_invN
        if self.add_invN: in_dim += 1
        self.add_invD= add_invD
        if self.add_invD: in_dim += 2
        self.add_invDsqrt= add_invDsqrt
        if self.add_invDsqrt: in_dim += 1

        self.add_loc = add_loc
        self.loc_dim = loc_dim
        self.loc_name = loc_name
        if self.add_loc:
            in_dim += loc_dim

        act_fn = act_dict[kwargs.get('act', 'gelu')]

        hid_dim= kwargs.get('hid_dim', '')
        if hid_dim == "":
            self.mlp = nn.Identity()
            self.out_fc = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            hid_dim = [in_dim] + [int(i) for i in hid_dim.split(',')]
            mlp = []
            for i in range(len(hid_dim)-1):
                mlp += [
                    nn.Linear(hid_dim[i], hid_dim[i+1]),
                    act_fn()
                ]
            self.mlp = nn.Sequential(*mlp)
            self.out_fc = nn.Linear(hid_dim[-1], out_dim, bias=bias)


        self.reduce = kwargs.get('reduce', 'add')
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
        self.kaiming_uniform_init = kwargs.get('kaiming_uniform_init', False)
        self.kaiming_uniform_linear_init = kwargs.get('kaiming_uniform_linear_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        # self.init_weights()


    def forward(self, batch):

        edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']

        if self.add_invD or self.add_invDsqrt:
            if 'deg' not in batch:
                raw_edge_index = batch.raw_edge_index if 'raw_edge_index' in batch else batch.edge_index
                batch.deg = pyg.utils.degree(raw_edge_index, num_nodes=batch.num_nodes, dtype=torch.float)

        if self.to_complete:
            edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)


        if self.add_loc:
            loc = batch[f'{self.loc_name}']
            loc = loc[edge_index[1]] - loc[edge_index[0]]
            edge_attr = torch.cat([edge_attr, loc], dim=-1)


        if self.add_invN:
            invN = (batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
            edge_attr = torch.cat([edge_attr, invN], dim=-1)


        if self.add_invD:
            invD = (1 / batch.deg).view(-1, 1)
            invD[invD==float('inf')] = 0.
            edge_attr = torch.cat([edge_attr, invD[edge_index[1]], invD[edge_index[0]]], dim=-1)


        if self.add_invDsqrt:
            invDsqrt = (1 / torch.sqrt(batch.deg)).view(-1, 1)
            invDsqrt[invDsqrt==float('inf')] = 0.
            edge_attr = torch.cat([edge_attr, invDsqrt[edge_index[1]], invDsqrt[edge_index[0]]], dim=-1)


        edge_attr = self.raw_norm(edge_attr, batch.batch[edge_index[1]])
        edge_attr = self.out_fc(self.mlp(edge_attr))
        edge_attr = self.post_norm(edge_attr, batch.batch[edge_index[1]])

        # if self.post_norm_type == 1:
        #     edge_attr = self.post_norm(edge_attr, edge_index[1])
        # else:

        if self.pe_name == 'edge':
            batch.edge_index,  batch.edge_attr = edge_index, edge_attr
            return batch


        # ------- merge with existing edge_index & edge_attr
        # if 'edge_attr' not in batch and not self.to_complete:
        #     batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)
        #     edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
        #     batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
        #                                                  num_nodes=batch.num_nodes, reduce=self.reduce,
        #                                                  )
        # elif 'edge_attr' not in batch and self.to_complete:
        #     batch.edge_index,  batch.edge_attr = edge_index, edge_attr
        # elif:
        # else:
        #     batch.edge_index,  batch.edge_attr = edge_index, edge_attr

        if self.add_to_edge:
            if 'edge_attr' not in batch:
                batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)

            edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
            batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
                                                         num_nodes=batch.num_nodes, reduce=self.reduce,
                                                         )
        else:
            batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr'] = edge_index, edge_attr

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

        elif self.kaiming_uniform_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)

        elif self.kaiming_uniform_linear_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)

        else:
            # if self.default_init:
            self.apply(default_init_)


@register_edge_encoder('EdgeDropout')
class EdgeDropout(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.in_dim = in_dim
        self.dropout = nn.Dropout(kwargs.get('dropout', 0.))
        self.feat_name = pe_name

    def forward(self, batch):

        batch[self.feat_name] = self.dropout(batch[self.feat_name])

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(dropout={self.dropout})'









@register_edge_encoder('ToCompleteEdgeEncoder')
class ToCompleteEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name="",
                 **kwargs):
        super().__init__()
        self.kwargs= kwargs

        # --------- config --------
        # self.add_vn = kwargs.get('add_vn', False)  # add virtual nodes
        self.reduce = kwargs.get('reduce', 'add')

        self.pe_name = ""

    def forward(self, batch):
        if self.pe_name == "":
            edge_index, edge_attr = batch.edge_index, batch.edge_attr
            batch.edge_index, batch.edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)

        else:
            edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']
            batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr'] = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)


        return batch


#


@register_edge_encoder('EdgeNorm')
class EdgeNormEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()

        self.pe_name = pe_name
        self.norm = act_dict[kwargs.get('norm', 'none')](in_dim)

        self.to_complete_graph = kwargs.get('to_complete_graph', False)

        self.kwargs = kwargs

    def forward(self, batch):
        edge_attr = batch[f'{self.pe_name}_attr']
        edge_index = batch[f'{self.pe_name}_index']

        if self.to_complete_graph:
            if 'raw_edge_index' not in batch:
                batch.raw_edge_index, batch.raw_edge_attr = batch.edge_index, batch.edge_attr
                edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)

            batch[f'{self.pe_name}_index'] = edge_index

        batch[f'{self.pe_name}_attr'] = self.norm(edge_attr)

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'


@register_edge_encoder('RemoveEdgeFeat')
class RemoveEdgeFeatdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')


    def forward(self, batch):

        batch.edge_attr = None

        return batch


@register_edge_encoder('MLPSinEdgeEncoder')
class MLPSinEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 bias:bool=True, # bias might introduce problematic bias without to_complete_graph
                 add_raw_feat:bool=True, # include the raw-input besides the sin encoding
                 add_invN:bool=False,
                 add_invD:bool=False,
                 add_to_edge:bool=True,
                 add_loc:bool=False,
                 loc_dim:int=2,
                 loc_name:str='pos',
                 drop_edge:float=0.0,
                 **kwargs):
        super().__init__()
        '''
        drop_edge: float = 0.0
        - for RPE and/or Transformers 
        - setting the RPE for a node-pair as all-zero randomly
        '''


        # print(f'kwargs is {kwargs}')
        self.pe_name = pe_name

        # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)

        self.kwargs= kwargs

        self.reduce= 'sum'



        self.to_complete = kwargs.get('to_complete_graph', False)
        self.add_raw_feat = add_raw_feat

        self.spectral_norm = kwargs.get('add_spectral_norm', False)
        self.weight_norm = kwargs.get('add_weight_norm', False)
        self.sin_L = kwargs.get('sin_L', 5)
        self.dropout = kwargs.get('dropout', 0.)
        self.add_to_edge = add_to_edge

        self.drop_edge = drop_edge


        # self.window_attn_hop = kwargs.get('window_attn_hop', -1)
        # for window attention --> K-hop


        if pe_name not in ['rrwp']:
            # only applied to positional encoding
            add_invN = False
            add_invD = False

        self.add_invN= add_invN
        self.add_invD = add_invD
        if self.add_invN:
            in_dim = in_dim + 1
        if self.add_invD: # inverse of degree
            in_dim += 2

        self.add_loc = add_loc
        self.loc_dim = loc_dim
        self.loc_name = loc_name
        if self.add_loc:
            in_dim += loc_dim


        self.raw_norm = NormalizationLayer(kwargs.get('raw_norm', 'none'), in_dim)
        self.post_norm = NormalizationLayer(kwargs.get('post_norm', 'none'), out_dim)



        # self.post_norm_type = 0
        # if 'kernel' in post_norm:
        #     self.post_norm_type = 1


        # to enable the sensing the value of num of nodes
        # self.add_log1pN = kwargs.get('add_log1pN', False)
        # --- add log(1+N) as a channel of features --> (~~ number of nodes in the region)
        # self.self_anchor= kwargs.get('self_anchor', False)

        self.pe_name = pe_name


        eps = (2 ** torch.arange(self.sin_L)) * np.pi if self.sin_L > 0 else None
        self.register_buffer('eps', eps)

        pe_dim = self.sin_L * 2 * in_dim

        if self.add_raw_feat:
            pe_dim = pe_dim + in_dim

        in_dim = pe_dim
        # pre_fc or pre_mlp
        act_fn = act_dict[kwargs.get('act', 'gelu')]
        hid_dim= kwargs.get('hid_dim', '')
        if hid_dim == "":
            self.mlp = nn.Identity()
            self.out_fc = nn.Linear(in_dim, out_dim, bias=bias)
        else:
            hid_dim = [in_dim] + [int(i) for i in hid_dim.split(',')]
            mlp = []
            for i in range(len(hid_dim)-1):
                mlp += [
                    nn.Linear(hid_dim[i], hid_dim[i+1]),
                    act_fn()
                ]
            self.mlp = nn.Sequential(*mlp)
            self.out_fc = nn.Linear(hid_dim[-1], out_dim, bias=bias)


        # if self.add_raw_feat:
        #     self.raw_feat_fc = nn.Linear(in_dim, out_dim, bias=False)


        self.default_init = kwargs.get('default_init', False)
        self.trunc_init = kwargs.get('trunc_init', False)
        self.kaiming_linear_init = kwargs.get('kaiming_linear_init', False)
        self.uniform_init = kwargs.get('uniform_init', False)
        self.kaiming_init = kwargs.get('kaiming_init', False)
        self.kaiming_uniform_init = kwargs.get('kaiming_uniform_init', False)
        self.kaiming_uniform_linear_init = kwargs.get('kaiming_uniform_linear_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        # self.init_weights()
        # if self.self_anchor:
        #     self.gamma = nn.Parameter(torch.ones(1, out_dim), requires_grad=True)
        #     self.beta = nn.Parameter(torch.zeros(1, out_dim), requires_grad=True)
        #     nn.init.trunc_normal_(self.beta, std=0.02)


        # self.add2node = kwargs.get('add2node', False)
        # if self.add2node:
        #     node_dim = kwargs.get('node_dim', out_dim)
        #     self.e2n_fc = nn.Linear(out_dim, node_dim)
        #     self.e2n_fc.apply(kaiming_uniform_init_)



    def _sin_pe(self, pe):
        if self.sin_L <= 0:
            return pe

        raw_pe = pe
        pe = pe.unsqueeze(-1) # E x D x 1
        eps = self.eps.view(1, 1, -1)
        sin_pe = torch.sin(eps * pe)
        cos_pe = torch.cos(eps * pe)
        pe = torch.stack([sin_pe, cos_pe], dim=-1)

        # return pe.flatten(1)
        return pe.flatten(1) if not self.add_raw_feat else torch.cat([raw_pe, pe.flatten(1)], dim=-1)


    def forward(self, batch):


        pe_index, pe_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']

        # ---- cache the degree infor ----
        if self.add_invD:
            if 'deg' not in batch:
                raw_edge_index = batch.raw_edge_index if 'raw_edge_index' in batch else batch.edge_index
                batch.deg = pyg.utils.degree(raw_edge_index, num_nodes=batch.num_nodes, dtype=torch.float)
        # --------------------------------

        # if self.window_attn_hop > 0:
        #     '''only supporting RRWP as PE-index; to do with spd'''
        #     assert self.pe_name in ['rrwp'], "window attention only supports RRWP as PE-index"
        #     if self.pe_name == 'rrwp':
        #         # the first column in rrwp is the self-identification
        #         mask = pe_attr[:, :self.window_attn_hop+2].sum(dim=-1) > 0
        #         pe_index, pe_attr = pe_index[mask], pe_attr[mask]
        if self.to_complete:
            if 'raw_edge_index' not in batch:
                batch.raw_edge_index, batch.raw_edge_attr = batch.edge_index, batch.edge_attr
                pe_index, pe_attr = to_complete_graph(pe_index, pe_attr, batch.batch, batch.num_nodes,
                                                          index_sorted=True, add_self_loops=True,
                                                          ptr=batch.get('ptr'))
            else:
                pass
                # with raw_edge_index --> it indicates the edges have been padded to complete in one of the previous modules





        if self.add_loc:
            loc = batch[f'{self.loc_name}']
            loc = loc[pe_index[1]] - loc[pe_index[0]]
            pe_attr = torch.cat([pe_attr, loc], dim=-1)



        if self.add_invN:
            invN = 1 / (batch.ptr[1:] - batch.ptr[:-1])[batch.batch][pe_index[1]].view(-1, 1) # E x 1
            pe_attr = torch.cat([pe_attr, invN], dim=-1)

        if self.add_invD:
            invD = (1 / batch.deg).view(-1, 1)
            invD[invD==float('inf')] = 0.
            pe_attr = torch.cat([pe_attr, invD[pe_index[0]], invD[pe_index[1]]], dim=-1)


        pe_attr = self.raw_norm(pe_attr, batch.batch[pe_index[1]])


        if self.drop_edge > 0.0 and self.training:
            random_mask = F.dropout(pe_attr[:, 0:1] * 0 + (1-self.drop_edge), p=self.drop_edge, training=self.training)
            pe_attr = pe_attr * random_mask


        pe_attr = self.out_fc(self.mlp(self._sin_pe(pe_attr)))
        pe_attr = self.post_norm(pe_attr, batch.batch[pe_index[1]])

        # if self.add2node:
        #     self_loop = pyg.utils.get_self_loop_attr(edge_index, edge_attr, batch.num_nodes)
        #     batch.x = batch.x + self.e2n_fc(self_loop)

        if self.pe_name == 'edge':
            batch.edge_attr, batch.edge_index = pe_attr, pe_index

        elif self.add_to_edge:
            if 'edge_attr' in batch:
                pe_index, pe_attr = torch.cat([pe_index, batch.edge_index], dim=-1), torch.cat([pe_attr, batch.edge_attr], dim=0)

            batch.edge_index, batch.edge_attr = coalesce(pe_index, pe_attr,
                                                         num_nodes=batch.num_nodes, reduce=self.reduce,
                                                         )
        else:
            batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr'] = pe_index, pe_attr

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


        elif self.kaiming_uniform_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)

        elif self.kaiming_uniform_linear_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)


        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        else:
            self.apply(default_init_)

        # self.apply(trunc_init_)
        # use kaiming_init with fan_in mode to adjust to different input-dims
        # self.apply(kaiming_uniform_init_)
        # self.apply(kaiming_uniform_init_)
        # self.apply(kaiming_normal_init_)
        # self.apply(kaiming_normal_linear_init_)
        # self.apply(kaiming_uniform_linear_init_)
        # self.apply(trunc_normal_fan_init_)






@register_edge_encoder('EdgeFFN')
class FFN(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name='edge',
                 edge_mode=True,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        self.edge_mode = edge_mode

        self.pe_name = pe_name if pe_name != '' else 'edge'
        assert in_dim == out_dim

        self.pre_backbone_norm = NormalizationLayer(kwargs.get('pre_backbone_norm', 'none'), in_dim)
        # self.pre_backbone_norm = act_dict[kwargs.get('pre_backbone_norm', 'none')](in_dim)


        self.add_spectral_norm = kwargs.get('add_spectral_norm', False)
        self.add_weight_norm = kwargs.get('add_weight_norm', False)



        act = kwargs.get('act', 'gelu')
        self.act = act
        if act not in ['siren', 'spder']:
            self.num_blocks = kwargs.get('num_blocks', 0)
            # ---- DropPath sccaling -----
            drop_path_rate = kwargs.get('drop_path', 0.)
            kwargs = copy_and_pop(kwargs, ['drop_path'])
            # safe operation, prevent from changing the cfg.
            layerwise_drop_path_rate = kwargs.get('layerwise_drop_path_rate', False)
            # by default use uniform droprate for droppath (from CaiT)
            num_layers =  self.num_blocks
            drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)] if layerwise_drop_path_rate else [drop_path_rate] * num_layers
            # ---------------------------
            if kwargs.get('swiglu', False):
                mlp =  ([(nn.Identity(), 'x -> x')] +
                        [(MLP(out_dim, out_dim, drop_path=drop_path_rate[i],
                              **kwargs), 'x, index, batch_tensor -> x')
                         for i in range(self.num_blocks)])


            else:
                mlp =  ([(nn.Identity(), 'x -> x')] +
                        [(MLP(out_dim, out_dim, drop_path=drop_path_rate[i],
                                                           **kwargs), 'x, index, batch_tensor -> x')
                         for i in range(self.num_blocks)])
            self.mlp = pyg.nn.Sequential('x, index, batch_tensor', mlp)

        else:
            # if kwargs.get('add_spectral_norm', False):
            #     warnings.warn("Siren MLP doesn't support spectral norm for now.")
            # if kwargs.get('add_weight_norm', False):
            #     warnings.warn("Siren MLP doesn't support weight norm for now.")
            self.num_blocks = kwargs.get('num_blocks', 0)
            ffn_ratio = kwargs.get('ffn_ratio', 2)
            omega = kwargs.get('omega', 30)
            mlp_dim = int(out_dim * ffn_ratio)
            mlp = [(nn.Identity(), 'x -> x')] + [(SirenMLP(out_dim, out_dim, mlp_dim,
                                              omega=omega, act=act
                                            ), 'x, index, batch_tensor -> x') for i in range(self.num_blocks)]
            self.mlp = pyg.nn.Sequential('x, index, batch_tensor', mlp)


        # self.post_backbone_norm = act_dict[kwargs.get('post_backbone_norm', 'none')](out_dim)
        self.post_backbone_norm = NormalizationLayer(kwargs.get('post_backbone_norm', 'none'), out_dim)


        self.lecun_init = kwargs.get('lecun_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)

        self.init_weights()

    def forward(self, batch):

        batch_tensor = batch.get('batch_tensor', None)

        if self.edge_mode:
            edge_index, attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']
            batch_index = batch.batch[edge_index[1]]
        else:
            attr = batch[f'{self.pe_name}']
            batch_index = batch.batch

        attr = self.pre_backbone_norm(attr, batch_index)
        attr = self.mlp(attr, batch_index, batch_tensor)
        attr= self.post_backbone_norm(attr, batch_index)

        if self.edge_mode:
            batch[f'{self.pe_name}_attr'] = attr
        else:
            batch[f'{self.pe_name}'] = attr

        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name}, edge_mode={self.edge_mode})'

    def init_weights(self):
        if 'gaussian' in self.act:
            self.mlp.apply(uniform_init_)
        else:
            init_weights_vit_timm(self.mlp)

        if self.lecun_init:
            self.mlp.apply(lecun_normal_init_)

        if self.xavier_init:
            self.mlp.apply(xavier_normal_init_)

        if self.add_weight_norm:
            self.mlp.apply(apply_weight_norm)

        if self.add_spectral_norm:
            self.mlp.apply(apply_spectral_norm)



        # elif self.act in ['siren', 'spder']:
        #     init_weights_vit_timm(self.mlp)



register_node_encoder('NodeFFN', partial(FFN, edge_mode=False))




@register_edge_encoder('MLPSirenEdgeEncoder')
class MLPSirenEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        if kwargs.get('e_dim', None) is not None:
            out_dim = kwargs.e_dim

        self.pe_name = pe_name

        self.add_invN= kwargs.get('add_invN', True)
        if self.add_invN:
            in_dim += 1

        act = kwargs.get('act', 'siren')

        hid_dim= kwargs.get('hid_dim', '')
        if hid_dim == "":
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            hid_dim = hid_dim.replace(' ', '').split(',')
            hid_dim = [int(i) for i in hid_dim]
            omega = kwargs.get('omega', 30)
            self.mlp = SirenMLP(in_dim, out_dim, hid_dim,
                                omega=omega, act=act)

        self.reduce = kwargs.get('reduce', 'add')
        self.kwargs= kwargs

        self.to_complete = kwargs.get('to_complete_graph', False)

        self.add_to_edge = kwargs.get('add_to_edge', True)


    def forward(self, batch):
        edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']

        if self.to_complete:
            edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)

        if self.add_invN:
            invN = 1/(batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
            edge_attr = torch.cat([edge_attr, edge_attr.new_ones(1, 1) * invN], dim=-1)

        edge_attr = self.mlp(edge_attr)

        # ------- merge with existing edge_index & edge_attr
        if self.pe_name == 'edge':
            batch.edge_attr, batch.edge_index = edge_attr, edge_index
        elif self.add_to_edge:
            if 'edge_attr' not in batch:
                batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)

            edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
            batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
                                                         num_nodes=batch.num_nodes, reduce=self.reduce,
                                                         )
        else:
            batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr'] = edge_index, edge_attr

        return batch



    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name}, to_complete={self.to_complete})'


    def init_weights(self):
        return None
        # to call the ini_weights inside the SirenMLP to use the proposed initialization techniques


        # use kaiming_init with fan_in mode to adjust to different input-dims










#
# @register_edge_encoder('MLPSPEEdgeEncoder')
# class MLPSPEEdgeEncoder(torch.nn.Module):
#     '''
#         RRWP learned with SPE(https://proceedings.mlr.press/v235/sun24m.html)
#         - using architecture similar to SPE to enhance
#     '''
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  pe_name=None,
#                  add_pe_to_x:bool=False,
#                  **kwargs):
#         super().__init__()
#         # print(f'kwargs is {kwargs}')
#
#         self.pe_name = pe_name
#
#         # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)
#
#         self.num_blocks = kwargs.get('num_blocks', 0)
#         self.kwargs= kwargs
#
#         self.reduce= 'sum'
#
#         self.to_complete = kwargs.get('to_complete_graph', False)
#
#         post_norm = kwargs.get('post_norm', 'none')
#         self.post_norm = act_dict[post_norm](out_dim)
#
#         self.post_norm_type = 0
#         if 'kernel' in post_norm:
#             self.post_norm_type = 1
#
#         self.add_invN= kwargs.get('add_invN', True)
#         # to enable the sensing the value of num of nodes
#         self.self_anchor= kwargs.get('self_anchor', False)
#
#         self.pe_name = pe_name
#         self.sin_L = kwargs.get('sin_L', 10)
#
#         eps = 2 ** torch.arange(self.sin_L) * np.pi
#         self.register_buffer('eps', eps)
#
#         if self.add_invN:
#             in_dim = in_dim + 1
#
#         spe_in_dim = in_dim * (2 * self.sin_L + 1)
#         spe_hid_dim = kwargs.get('spe_hid_dim', out_dim)
#         spe_out_dim = kwargs.get('spe_out_dim', int(np.sqrt(out_dim)))
#
#         self.spe_in_fc = nn.Linear(spe_in_dim, spe_hid_dim)
#         self.spe_out_fc = nn.Linear(spe_hid_dim, spe_out_dim)
#
#         # pre_fc or pre_mlp
#
#         act = kwargs.get('act', 'gelu')
#         self.act = act_dict[act]()
#         # self.post_fc = nn.Linear(in_dim * spe_out_dim, out_dim)
#         self.post_fc = nn.Linear(spe_out_dim, out_dim)
#
#         mlp = [nn.Identity()] + [MLP(out_dim, out_dim, **kwargs) for i in range(self.num_blocks)]
#         self.mlp = nn.Sequential(*mlp)
#         self.mlp.apply(trunc_init_)
#
#         post_merge_norm = kwargs.get('post_merge_norm', 'none')
#         self.post_merge_norm = act_dict[post_merge_norm](out_dim)
#
#         self.add2node = kwargs.get('add2node', False)
#         if self.add2node:
#             node_dim = kwargs.get('node_dim', out_dim)
#             self.e2n_fc = nn.Linear(out_dim, node_dim)
#
#
#
#     def _sin_pe(self, pe):
#         pe = pe.unsqueeze(-1) # E x D x 1
#         raw_pe = pe
#         eps = self.eps.view(1, 1, -1)
#         sin_pe = torch.sin(eps * pe)
#         cos_pe = torch.cos(eps * pe)
#         pe = torch.cat([raw_pe, sin_pe, cos_pe], dim=-1)
#
#         return pe
#
#
#     def forward(self, batch):
#         if 'raw_edge_index' not in batch:
#             batch.raw_edge_index, batch.raw_edge_attr = batch.edge_index, batch.edge_attr
#
#         edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']
#
#         if self.to_complete:
#             edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)
#
#         if self.add_invN:
#             invN = 1/(batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
#             edge_attr = torch.cat([edge_attr, edge_attr.new_ones(1, 1) * invN], dim=-1)
#
#         edge_attr = self._sin_pe(edge_attr)
#         edge_attr = self.spe_out_fc(torch.sin(self.spe_in_fc(edge_attr.flatten(1))))
#         # edge_attr = edge_attr.transpose(-1, -2)
#         # edge_attr = self.post_fc(self.act(self.pe_mix(edge_attr).flatten(1)))
#         edge_attr = self.post_fc(self.act(edge_attr.flatten(1)))
#
#
#         if self.add2node:
#             self_loop = pyg.utils.get_self_loop_attr(edge_index, edge_attr, batch.num_nodes)
#             batch.x = batch.x + self.e2n_fc(self_loop)
#
#         edge_attr = self.mlp(edge_attr)
#
#
#         # ------- merge with existing edge_index & edge_attr
#         if 'edge_attr' not in batch and not self.to_complete:
#             batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)
#
#         if 'edge_attr' in batch or not self.to_complete:
#             if 'edge_attr' not in batch:
#                 batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)
#
#             edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
#             batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
#                                              num_nodes=batch.num_nodes, reduce=self.reduce,
#                                              )
#         else:
#             batch.edge_index, batch.edge_attr = edge_index, edge_attr
#
#
#         batch.edge_attr = self.post_merge_norm(batch.edge_attr)
#
#         return batch
#
#








@register_edge_encoder('MLPGarfEdgeEncoder')
class MLPGarfEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        if kwargs.get('e_dim', None) is not None:
            out_dim = kwargs.e_dim

        self.pe_name = pe_name

        self.add_invN= kwargs.get('add_invN', True)
        if self.add_invN:
            in_dim += 1

        kwargs['act'] = kwargs.get('act', 'gaussian0.5')

        # assert 'gaussian' in act

        mlp_dim= kwargs.get('mlp_dim', '')
        if mlp_dim == "":
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            mlp_dim = mlp_dim.replace(' ', '').split(',')
            mlp_dim = [int(i) for i in mlp_dim]
            self.mlp = MLP(in_dim, out_dim, mlp_dim,
                           **kwargs
                           )

        self.mlp.apply(uniform_init_)


        self.reduce = kwargs.get('reduce', 'add')
        self.kwargs= kwargs

        self.bypass =  cfg.get('bypass', False)
        if self.bypass:
            self.bypass_fc = nn.Linear(in_dim, out_dim, bias=False)
            nn.init.xavier_normal_(self.bypass_fc.weight)

        self.to_complete = kwargs.get('to_complete_graph', False)

        self.add2node = kwargs.get('add2node', False)
        if self.add2node:
            node_dim = kwargs.get('node_dim', out_dim)
            self.e2n_fc = nn.Linear(out_dim, node_dim)


    def forward(self, batch):
        edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']

        if self.to_complete:
            edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)

        if self.add_invN:
            invN = 1/(batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
            edge_attr = torch.cat([edge_attr, edge_attr.new_ones(1, 1) * invN], dim=-1)



        bypass = self.bypass_fc(edge_attr)  if self.bypass else 0
        edge_attr = self.mlp(edge_attr) + bypass

        if self.add2node:
            self_loop = pyg.utils.get_self_loop_attr(edge_index, edge_attr, batch.num_nodes)
            batch.x = batch.x + self.e2n_fc(self_loop)

        if self.pe_name == 'edge':
            batch.edge_index,  batch.edge_attr = edge_index, edge_attr
            return batch


        # ------- merge with existing edge_index & edge_attr
        if 'edge_attr' in batch or not self.to_complete:
            if 'edge_attr' not in batch:
                batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)

            edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
            batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
                                             num_nodes=batch.num_nodes, reduce=self.reduce,
                                             )
        else:
            batch.edge_index,  batch.edge_attr = edge_index, edge_attr

        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'




#
#
# @register_edge_encoder('MLFFEdgeEncoder')
# class MLPFourierFeaturesEdgeEncoder(torch.nn.Module):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  pe_name=None,
#                  **kwargs):
#         super().__init__()
#         # print(f'kwargs is {kwargs}')
#
#         if kwargs.get('e_dim', None) is not None:
#             out_dim = kwargs.e_dim
#
#         self.pe_name = pe_name
#
#         self.add_invN= kwargs.get('add_invN', True)
#         if self.add_invN:
#             in_dim += 1
#
#         kwargs['act'] = kwargs.get('act', 'gaussian0.5')
#
#         # assert 'gaussian' in act
#
#         mlp_dim= kwargs.get('mlp_dim', '')
#         if mlp_dim == "":
#             self.mlp = nn.Linear(in_dim, out_dim)
#         else:
#             mlp_dim = mlp_dim.replace(' ', '').split(',')
#             mlp_dim = [int(i) for i in mlp_dim]
#             self.mlp = MLP(in_dim, out_dim, mlp_dim,
#                            **kwargs
#                            )
#
#         self.mlp.apply(uniform_init_)
#
#
#         self.reduce = kwargs.get('reduce', 'add')
#         self.kwargs= kwargs
#
#         self.bypass =  cfg.get('bypass', False)
#         if self.bypass:
#             self.bypass_fc = nn.Linear(in_dim, out_dim, bias=False)
#             nn.init.xavier_normal_(self.bypass_fc.weight)
#
#         self.to_complete = kwargs.get('to_complete_graph', False)
#
#         self.add2node = kwargs.get('add2node', False)
#         if self.add2node:
#             node_dim = kwargs.get('node_dim', out_dim)
#             self.e2n_fc = nn.Linear(out_dim, node_dim)
#
#
#     def forward(self, batch):
#         edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']
#
#         if self.to_complete:
#             edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)
#
#         if self.add_invN:
#             invN = (batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
#             edge_attr = torch.cat([edge_attr, edge_attr.new_ones(1, 1) * invN], dim=-1)
#
#
#
#         bypass = self.bypass_fc(edge_attr)  if self.bypass else 0
#         edge_attr = self.mlp(edge_attr) + bypass
#
#         if self.add2node:
#             self_loop = pyg.utils.get_self_loop_attr(edge_index, edge_attr, batch.num_nodes)
#             batch.x = batch.x + self.e2n_fc(self_loop)
#
#         if self.pe_name == 'edge':
#             batch.edge_index,  batch.edge_attr = edge_index, edge_attr
#             return batch
#
#
#         # ------- merge with existing edge_index & edge_attr
#         if 'edge_attr' in batch or not self.to_complete:
#             if 'edge_attr' not in batch:
#                 batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)
#
#             edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
#             batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
#                                              num_nodes=batch.num_nodes, reduce=self.reduce,
#                                              )
#         else:
#             batch.edge_index,  batch.edge_attr = edge_index, edge_attr
#
#         return batch
#
#
#     def __repr__(self):
#         return f'{super().__repr__()}(pe_name={self.pe_name})'









@register_edge_encoder('MLPRBFEdgeEncoder')
class MLPRBFEdgeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        # print(f'kwargs is {kwargs}')

        if kwargs.get('e_dim', None) is not None:
            out_dim = kwargs.e_dim

        self.pe_name = pe_name

        self.xN = kwargs.get('xN', False)

        # assert 'gaussian' in act
        mlp_dim= kwargs.get('mlp_dim', '')
        if mlp_dim != "":
            mlp_dim = mlp_dim.replace(' ', '').split(',')
        else:
            mlp_dim = []
        mlp_dim = [int(i) for i in mlp_dim] + [out_dim]

        self.add_invN= kwargs.get('add_invN', True)
        if self.add_invN:
            self.rbf = RBFLayerCenter(in_dim, mlp_dim[0], affine=True, bias=True)
        else:
            self.rbf = RBFLayer(in_dim, mlp_dim[0], affine=True, bias=True)

        act_fn = act_dict[kwargs.get('act', 'gelu')]
        mlp = [nn.Identity()]
        for i in range(len(mlp_dim)-1):
            mlp.append(act_fn())
            mlp.append(nn.Linear(mlp_dim[i], mlp_dim[i+1]))

        self.mlp = nn.Sequential(*mlp)
        self.mlp.apply(trunc_init_)


        self.reduce = kwargs.get('reduce', 'add')
        self.kwargs= kwargs

        # self.bypass =  cfg.get('bypass', False)
        # if self.bypass:
        #     self.bypass_fc = nn.Linear(in_dim, out_dim, bias=False)
        #     nn.init.xavier_normal_(self.bypass_fc.weight)

        self.to_complete = kwargs.get('to_complete_graph', False)

        self.add2node = kwargs.get('add2node', False)
        if self.add2node:
            node_dim = kwargs.get('node_dim', out_dim)
            self.e2n_fc = nn.Linear(out_dim, node_dim)


    def forward(self, batch):
        edge_index, edge_attr = batch[f'{self.pe_name}_index'], batch[f'{self.pe_name}_attr']

        if self.to_complete:
            edge_index, edge_attr = to_complete_graph(edge_index, edge_attr, batch.batch, batch.num_nodes)

        if self.add_invN:
            invN = 1 / (batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
            edge_attr = self.rbf(edge_attr, invN)
        else:
            if self.xN:
                N = (batch.ptr[1:] - batch.ptr[:-1])[batch.batch][edge_index[1]].view(-1, 1)
                edge_attr = edge_attr * N

            edge_attr = self.rbf(edge_attr)


        # bypass = self.bypass_fc(edge_attr)  if self.bypass else 0
        # edge_attr = self.mlp(edge_attr) + bypass
        edge_attr = self.mlp(edge_attr)

        if self.add2node:
            self_loop = pyg.utils.get_self_loop_attr(edge_index, edge_attr, batch.num_nodes)
            batch.x = batch.x + self.e2n_fc(self_loop)

        if self.pe_name == 'edge':
            batch.edge_index,  batch.edge_attr = edge_index, edge_attr
            return batch


        # ------- merge with existing edge_index & edge_attr
        if 'edge_attr' in batch or not self.to_complete:
            if 'edge_attr' not in batch:
                batch.edge_attr = edge_attr.new_zeros(batch.edge_index.size(1), edge_attr.size(1), dtype=torch.float)

            edge_index, edge_attr = torch.cat([edge_index, batch.edge_index], dim=-1), torch.cat([edge_attr, batch.edge_attr], dim=0)
            batch.edge_index, batch.edge_attr = coalesce(edge_index, edge_attr,
                                             num_nodes=batch.num_nodes, reduce=self.reduce,
                                             )
        else:
            batch.edge_index,  batch.edge_attr = edge_index, edge_attr

        return batch


    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'
