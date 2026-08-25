from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('example')
def example_losses(pred, true):
    if cfg.model.loss_fun == 'example':
        return 1, 1

    '''
    graphgym will iterate all registered loss;
    - if return is not None; will use this loss to backward
    - else:
        go over other loss
    
    
    
    '''




