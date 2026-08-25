import torch

from torch_geometric.graphgym.register import register_act

from functools import partial


class GaussianAct(torch.nn.Module):
    '''
        Gaussian Activation from
        - https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930139.pdf
        - https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930259.pdf




    '''
    def __init__(self, sigma=0.1, trainable=False, minus_mu=False, dim=1):
        super().__init__()
        self.init_sigma = sigma
        self.minus_mu = minus_mu
        # first layer in GARF minuses mu; other layers do not

        if trainable:
            # Todo: To consider different channel with different sigma
            self.sigma = torch.nn.Parameter(torch.ones(1, dim=dim) * sigma, requires_grad=True)
        else:
            self.sigma = sigma

        self.trainable = trainable

    def forward(self, x):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        if self.minus_mu:
            mu = x.mean(dim=-1, keepdim=True)
        else:
            mu = 0



        if self.trainable:
            shapes = [1] * (x.dim()-1) + [-1]
            sigma = self.sigma.view(*shapes)
        else:
            sigma = self.sigma

        k1 = (-0.5*(x - mu)**2/sigma**2).exp()
        return k1

    def extra_repr(self) -> str:
        return f"sigma={self.init_sigma}, trainable_sigma={self.trainable}, minus_mu={self.minus_mu}"



register_act('gaussian0.5', GaussianAct)
register_act('gaussian0.5_trainable', partial(GaussianAct, trainable=True))