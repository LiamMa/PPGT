import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder

from ..layer.utils.init import *



@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10']:
            self.in_dim = 1
        elif cfg.dataset.name.startswith('attributed_triangle-'):
            self.in_dim = 2
        else:
            self.in_dim =cfg.dataset.edge_encoder_num_types
            # raise ValueError("Input edge feature dim is required to be hardset "
            #                  "or refactored to use a cfg option.")

        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)


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
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim))

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

        # self.apply(trunc_init_)

