import torch.nn as nn
from torch_geometric.graphgym.register import register_act



@register_act('batch_renorm')
class BatchRenormalization(nn.Module):
    '''
        Implementation of batch-renorm
        > fixme: to verify; not sure about the correctness
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn = nn.BatchNorm1d(*args, **kwargs)

    def forward(self, x):
        if self.training:
            _ = self.bn(x)
            self.bn.eval()
            x = self.bn(x)
            self.bn.train()
        else:
            x = self.bn(x)

        return x


