import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder
from ..layer.utils.init import *


@register_node_encoder('LinearNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(cfg.share.dim_in, emb_dim)

        kwargs = cfg.dataset
        self.default_init = kwargs.get('default_init', False)
        self.trunc_init = kwargs.get('trunc_init', False)
        self.kaiming_linear_init = kwargs.get('kaiming_linear_init', False)
        self.uniform_init = kwargs.get('uniform_init', False)
        self.kaiming_init = kwargs.get('kaiming_init', False)
        self.kaiming_uniform_init = kwargs.get('kaiming_uniform_init', False)
        self.kaiming_uniform_linear_init = kwargs.get('kaiming_uniform_linear_init', False)
        self.xavier_init = kwargs.get('xavier_init', False)
        self.lecun_init = kwargs.get('lecun_init', False)

        self.init_weights()

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch


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
            self.apply(kaiming_uniform_init_)

        elif self.kaiming_uniform_linear_init:
            self.apply(kaiming_uniform_linear_init_)

        elif self.uniform_init:
            self.apply(uniform_init_)

        elif self.lecun_init:
            self.apply(lecun_normal_init_)
        else:
            self.apply(default_init_)




@register_node_encoder('LinearNodeV2')
class LinearNodeV2Encoder(torch.nn.Module):
    '''
        Not add to x; keep the original name
    '''
    def __init__(self, emb_dim, out_dim, attr_name='pe', **kwargs):
        super().__init__()

        self.encoder = torch.nn.Linear(emb_dim, out_dim)
        self.name = attr_name

    def forward(self, batch):
        x = batch[self.name]
        if x.dim() > 2:
            x = x.transpose(1, -1)

        x = self.encoder(x)

        if x.dim() > 2:
            x = x.transpose(1, -1)

        batch[self.name] = x
        return batch
