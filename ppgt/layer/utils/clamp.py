import torch













def straight_through_clamp(x, min=None, max=None):
    '''
        Clamping with [straight through estimator](https://arxiv.org/pdf/1308.3432); also utilized in [Gumbel-softmax](https://arxiv.org/pdf/1611.01144)
    '''
    if min is not None:
        x = x + torch.relu(min- x).detach()

    if max is not None:
        x = x - torch.relu(x - max).detach()

    return x