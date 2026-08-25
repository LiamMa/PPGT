import torch
from torch import nn
# from torch.nn.parallel.comm import scatter

import torch_geometric as pyg
from torch_geometric.utils import scatter


from torch_geometric.graphgym.register import register_node_encoder, act_dict
from einops import repeat
from ..layer.utils.init import trunc_init_, uniform_init_, kaiming_uniform_init_, kaiming_normal_linear_init_, kaiming_uniform_linear_init_, kaiming_normal_init_, xavier_normal_init_, default_init_, lecun_normal_init_

from ..layer.utils.norm import NormalizationLayer
from warnings import warn


@register_node_encoder('GetGraphOrder')
class GetGraphOrder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.in_dim = in_dim

    def forward(self, batch):

        if 'ptr' in batch:
            order = batch.ptr[1:] - batch.ptr[:-1]
        else:
            order = scatter(torch.ones_like(batch.batch), batch.batch, dim=0, dim_size=max(batch.batch)+1, reduce='sum')

        batch.graph_order = order[batch.batch]

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'



@register_node_encoder('NodeDropout')
class NodeDropout(torch.nn.Module):
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




@register_node_encoder('NodeMask')
class NodeMask(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.in_dim = in_dim
        self.p = kwargs.get('mask_rate', 0.)
        self.feat_name = pe_name

    def forward(self, batch):

        if not self.training or self.p == 0.:
            return batch

        x= batch[self.feat_name]

        batch[self.feat_name] = x * (torch.rand(x.size(0), 1, device=x.device, dtype=x.dtype) > self.p)

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(NodeMaskRate={self.p})'





@register_node_encoder('MLPNodeEncoder')
class MLPNodeEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 add_to_x=True,
                 add_invN:bool=False,
                 add_invD:bool=False,
                 bias:bool=True,
                 add_loc:bool=False,
                 loc_dim:int=2,
                 loc_name:str='loc',
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs


        self.pe_name = pe_name
        # self.raw_norm = act_dict[kwargs.get('raw_norm', 'none')](in_dim)
        self.raw_norm = NormalizationLayer(kwargs.get('raw_norm', 'none'), in_dim)

        if pe_name not in ['rrwp']:
            # only applied to positional encoding
            add_invN = False
            add_invD = False
        self.add_invN = add_invN
        self.add_invD = add_invD
        if self.add_invN: in_dim += 1
        if self.add_invD: in_dim += 1

        self.add_loc = add_loc
        self.loc_name = loc_name
        if self.add_loc: in_dim += loc_dim



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


        # self.post_norm = act_dict[kwargs.get('post_norm', 'none')](out_dim)
        self.post_norm = NormalizationLayer(kwargs.get('post_norm', 'none'), out_dim)

        self.add_to_x = add_to_x
        self.in_dim = in_dim

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

        if 'order' in self.pe_name and 'order' not in batch:
            if 'ptr' in batch:
                order = batch.ptr[1:] - batch.ptr[:-1]
            else:
                order = scatter(torch.ones_like(batch.batch), batch.batch, dim=0, dim_size=max(batch.batch)+1, reduce='sum')
            batch.order = order[batch.batch]
            if self.pe_name == 'log_order':
                batch.log_order = torch.log1p(batch.order)
            # store graph order

        if self.pe_name == 'deg' and 'deg' not in batch:
            raw_edge_index = batch.raw_edge_index if 'raw_edge_index' in batch else batch.edge_index
            batch.deg = pyg.utils.degree(raw_edge_index, num_nodes=batch.num_nodes, dtype=torch.float)


        attr = batch[self.pe_name].type(torch.float)
        attr = attr.view(attr.size(0), -1)

        if self.add_invN:
            invN = 1 / (batch.ptr[1:] - batch.ptr[:-1])[batch.batch].view(-1, 1) # E x 1
            attr = torch.cat([attr, invN], dim=-1)

        if self.add_invD:
            if 'deg' not in batch:
                raw_edge_index = batch.raw_edge_index if 'raw_edge_index' in batch else batch.edge_index
                batch.deg = pyg.utils.degree(raw_edge_index, num_nodes=batch.num_nodes, dtype=torch.float)

            invD = (1 / batch.deg).view(-1, 1)
            invD[invD==float('inf')] = 0.
            attr = torch.cat([attr, invD], dim=-1)

        if self.add_loc:
            loc = batch[f'{self.loc_name}']
            attr = torch.cat([attr, loc], dim=-1)

        x = self.post_norm(self.out_fc(self.mlp(self.raw_norm(attr, batch.batch))), batch.batch)

        if self.add_to_x:
            if 'x' in batch:
                batch.x = batch.x + x
            else:
                batch.x = x
        else:
            batch[self.pe_name] = x

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'

    def init_weights(self):
        # use kaiming_init with fan_in mode to adjust to different input-dims
        # self.apply(kaiming_uniform_init_)
        # self.apply(kaiming_normal_init_)
        # self.apply(kaiming_normal_linear_init_)
        # self.apply(kaiming_uniform_linear_init_)

        if self.xavier_init:
            self.apply(xavier_normal_init_)

        elif self.trunc_init:
            self.apply(trunc_init_)

        elif self.kaiming_linear_init:
            self.apply(kaiming_normal_linear_init_)

        elif self.kaiming_uniform_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)

        elif self.kaiming_uniform_linear_init:
            self.mlp.apply(kaiming_uniform_init_)
            self.out_fc.apply(kaiming_uniform_linear_init_)


        elif self.kaiming_init:
            self.apply(kaiming_normal_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        else:
            self.apply(default_init_)
        # self.apply(trunc_init_)
        # self.apply(trunc_normal_fan_init_)



@register_node_encoder('CLSEmb')
class CLSEmbedding(torch.nn.Module):
    '''
    Add extra embedding for nodes with cls mask
    '''
    def __init__(self,
                 in_dim=None,
                 out_dim=None,
                 batch_size=None,
                 add_log_order:bool=False,
                 **kwargs):
        super().__init__()
        self.kwargs= kwargs
        self.add_log_order = add_log_order

        self.cls_token = nn.Parameter(torch.zeros(1, out_dim))

        if self.add_log_order:
            self.order_fc = nn.Linear(1, out_dim)

        self.init_weights()

    def forward(self, batch):
        assert'cls_mask' in batch, "CLSMaskEmbedding requires cls_mask in batch"

        batch.x[batch.cls_mask] = self.cls_token
        if self.add_log_order:
            if 'ptr' in batch:
                order = batch.ptr[1:] - batch.ptr[:-1]
            else:
                order = scatter(torch.ones_like(batch.batch), batch.batch, dim=0, dim_size=max(batch.batch)+1, reduce='sum')
            log_order = torch.log1p(order).view(-1, 1)
            batch.x[batch.cls_mask] = batch.x[batch.cls_mask] + self.order_fc(log_order)

        batch.cls = batch.x[batch.cls_mask] 
        # save cls-tokens separately for Class-Attention in CaiT

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(cls_token={self.cls_token.shape}, add_log_order={self.add_log_order})'

    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.add_log_order:
            nn.init.trunc_normal_(self.order_fc.weight, std=0.02)




@register_node_encoder('TargetNodeEmb')
class TargetNodeEmbedding(torch.nn.Module):
    '''
    Add extra embedding for target nodes
    '''
    def __init__(self,
                 in_dim=None,
                 out_dim=None,
                 batch_size=None,
                 **kwargs):
        super().__init__()
        self.kwargs= kwargs
        self.target_token = nn.Parameter(torch.zeros(1, out_dim))
        self.init_weights()

    def forward(self, batch):
        assert'target_mask' in batch, "TargerNodeEmbedding requires target_mask in batch"

        batch.x[batch.target_mask] += self.target_token
        return batch

    def __repr__(self):
        return f'{super().__repr__()}(target_token={self.target_token.shape})'

    def init_weights(self):
        nn.init.trunc_normal_(self.target_token, std=0.02)



@register_node_encoder('NodeNorm')
class NodeNormEncoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 pe_name=None,
                 **kwargs):
        super().__init__()

        self.pe_name = pe_name
        self.norm = act_dict[kwargs.get('norm', 'none')](in_dim)
        self.kwargs = kwargs

    def forward(self, batch):

        batch[f'{self.pe_name}'] = self.norm(batch[f'{self.pe_name}'])

        return batch

    def __repr__(self):
        return f'{super().__repr__()}(pe_name={self.pe_name})'



@register_node_encoder('CLSTokenEmb')
class ClassTokenEmbedding(torch.nn.Module):
    '''
        Add embedding for virtual nodes (remove the original node)
    '''
    def __init__(self,
                 in_dim=None,
                 out_dim=None,
                 pe_name='cls_token',
                 batch_size=None,
                 add_log_order:bool=False,
                 add_order:bool=False,
                 **kwargs):
        super().__init__()
        self.kwargs= kwargs
        self.add_log_order = add_log_order
        self.add_order = add_order

        if out_dim is None: out_dim = in_dim
        self.cls_name = pe_name

        self.cls_token = nn.Parameter(torch.zeros(1, out_dim))

        if self.add_order and self.add_log_order: 
            raise ValueError('add_order and add_log_order cannot be True at the same time')
        if self.add_log_order or self.add_order:
            self.order_fc = nn.Sequential(nn.Linear(1, out_dim//2), nn.GELU(), nn.Linear(out_dim//2, out_dim))

        # note:
        #    Instantiate the new_tensor or x.expand() in each forward pass is slow.
        #    > Store one in buffer instead
        #    > even for the case that last mini-batch has smaller batch size
        #    > the random tensor still works since `random_tensor[batch_index]` is still valid

    def forward(self, batch):

        # cls_tokens = self.pad_tensor[:num_graphs] * self.cls_token
        if 'ptr' in batch:
            num_graphs = batch.ptr.size(-1) - 1
        else:
            num_graphs = max(batch.batch)+1 if 'batch' in batch else 1

        cls_token = self.cls_token.expand(num_graphs, -1)

        if self.add_log_order or self.add_order:
            if 'ptr' in batch:
                order = batch.ptr[1:] - batch.ptr[:-1]
            else:
                order = scatter(torch.ones_like(batch.batch), batch.batch, dim=0, dim_size=max(batch.batch)+1, reduce='sum')

            order = torch.log1p(order).view(-1, 1) if self.add_log_order else order.view(-1, 1).float()
            cls_token = cls_token + self.order_fc(order)        

        if self.cls_name.endswith("mask"):
            batch.x[batch[self.cls_name]] = cls_token
        
        else:
            batch[self.cls_name] = cls_token

        return batch


    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.add_log_order:
            nn.init.trunc_normal_(self.order_fc[0].weight, std=0.02)
            nn.init.trunc_normal_(self.order_fc[2].weight, std=0.02)






@register_node_encoder('RegisterTokenEmb')
class RegisterTokenEmbedding(torch.nn.Module):
    '''
    Register Token embeddings
    '''
    def __init__(self,
                 in_dim=None,
                 out_dim=None,
                 pe_name='register_mask',
                 batch_size=None,
                 **kwargs):
        super().__init__()
        self.kwargs= kwargs
        self.mask_name = pe_name
        self.num_register_tokens = in_dim
        self.register_tokens = nn.Parameter(torch.zeros(in_dim, out_dim))
        self.init_weights()

        self.token_name = pe_name 
        if self.token_name != 'register_mask':
            warn(f'RegisterTokenEmb requires [register_mask] in batch, not using {self.mask_name}')

    def forward(self, batch):

        register_token = batch.x[batch[self.token_name]] 

        batch_size = batch.x[batch[self.token_name]].size(0) // self.num_register_tokens

        dim = register_token.size(-1)

        register_tokens = repeat(self.register_tokens, 'n d -> b n d', b=batch_size) 

        batch.x[batch[self.token_name]] = register_tokens.reshape(-1, dim)
        
        return batch

    def __repr__(self):
        return f'{super().__repr__()}(register_tokens={self.register_tokens.shape})'

    def init_weights(self):
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

