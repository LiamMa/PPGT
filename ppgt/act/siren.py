import torch

from torch_geometric.graphgym.register import register_act






@register_act('siren')
class Siren(torch.nn.Module):
    r'''
        Sine Activation as in Siren
    '''
    def __init__(self, omega=30.):
        super().__init__()

        self.omega = omega

    def forward(self, x):

        return torch.sin(self.omega * x)

    def extra_repr(self) -> str:
        return f"omega={self.omega}"




@register_act('spder')
class SPDER(torch.nn.Module):
    r'''
        SPDER: (https://openreview.net/pdf?id=92btneN9Wm)
        - SIREN * \sqrt(|x|)
    '''
    def __init__(self, omega=30.):
        super().__init__()

        self.omega = omega


    def forward(self, x):
        x = self.omega * x
        return torch.sin(x) * (x.abs() + 1e-31).sqrt()

    # backward of 0.sqrt() is nan

    def extra_repr(self) -> str:
        return f"omega={self.omega}"