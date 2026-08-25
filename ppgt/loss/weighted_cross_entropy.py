import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
import torch.nn as nn


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    # Lazy implementation--> the mini-batch distribution might not be representative of the entire dataset
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true)
        label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        cluster_sizes[torch.unique(true)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)


# # not verified yet; might not be correct
# @register_loss('focal_loss')
# def focal_loss(pred, true):
#     """
#     Args:
#         pred: Logits of shape (B, C)
#         true: Labels of shape (B)
#         alpha: Tensor of shape (C,) for class weights (optional)
#         gamma: Focusing parameter
#     """
#     # 1. Calculate standard Cross Entropy (reduction='none' to keep per-sample loss)
#     # We pass alpha (weights) here if provided, to leverage PyTorch's optimized implementation

#     # alpha = cfg.model.get('focal_loss_alpha', 1.0)
#     gamma = cfg.model.get('focal_loss_gamma', 2.0)
#     pred = F.log_softmax(pred, dim=-1)
#     ce_loss = F.nll_loss(pred, true, reduction='none')
    
#     # 2. Calculate probabilities (pt)
#     # ce_loss = -log(pt), so pt = exp(-ce_loss)
#     pt = torch.exp(-ce_loss)
    
#     # 3. Calculate Focal component
#     # formula: (1 - pt)^gamma * -log(pt)
#     # Note: ce_loss includes the -log(pt) and the alpha (if passed to F.cross_entropy)
#     focal_loss = (1 - pt) ** gamma * ce_loss

#     return focal_loss.mean(), pred



@register_loss('smoothed_cross_entropy')
def smoothed_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'smoothed_cross_entropy':
        # label smoothing
        # pred = F.log_softmax(pred, dim=-1

        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.model.get('label_smoothing', 0.1))

        return loss_fn(pred, true), F.log_softmax(pred, dim=-1)