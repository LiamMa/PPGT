import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


'''
    To be aligned with GraphGPS
'''


@register_loss('multilabel_cross_entropy')
def multilabel_cross_entropy(pred, true):
    """Multilabel cross-entropy loss.
    """

    pred_, true_ = pred, true
    if cfg.dataset.task_type == 'classification_multilabel':

        if cfg.model.loss_fun == 'cross_entropy':
            bce_loss = nn.BCEWithLogitsLoss()
            is_labeled = true == true  # Filter our nans.
            loss = bce_loss(pred[is_labeled], true[is_labeled].float())

        elif cfg.model.loss_fun == 'weighted_cross_entropy':
            is_labeled = true == true  # Filter our nans.
            # is_labeled = is_labeled.type(torch.float)



            num_pos = torch.where(is_labeled, true, torch.zeros_like(true)).sum(dim=0, keepdim=True)
            num_neg = torch.where(is_labeled, 1-true, torch.zeros_like(true)).sum(dim=0, keepdim=True)

            batch_size = pred.size(0)


            pos_weight = (num_neg / num_pos).clamp(0, batch_size)
            # clamp pos_weight to avoid 'inf' weight

            bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            bce_loss = nn.BCELoss(pos_weight=pos_weight)

            pred = torch.sigmoid(pred)
            true = torch.where(is_labeled, true, torch.zeros_like(true))
            pred = torch.where(is_labeled, pred, torch.zeros_like(pred))

            loss = bce_loss(pred, true.float())


        elif cfg.model.loss_fun == 'asymmetric_loss':
            # is_labeled = true == true  # Filter our nans.
            loss, _ = asymmetric_loss(pred, true)
        else:
            raise ValueError("Only '(weighted) cross_entropy' loss_fun supported with "
                             "'classification_multilabel' task_type.")


        return loss, pred_




def asymmetric_loss(pred, true):
    # for unbalanced multi-label classifications

    # INSERT_YOUR_CODE
    # Asymmetric loss for unbalanced multi-label classification
    # Reference: https://arxiv.org/abs/2009.14119 (ASL: Asymmetric Loss For Multi-Label Classification)
    # This implementation assumes pred: logits, true: binary labels (0/1)
    # Typical gamma_pos=0, gamma_neg=4, clip=0.05

    gamma_pos = 0.0
    gamma_neg = 4.0
    clip = 0.05

    # Sigmoid on logits
    pred_sigmoid = torch.sigmoid(pred)

    # Prevent nan in log
    eps = 1e-8

    # Positive and negative targets
    pos_inds = (true == 1)
    neg_inds = (true == 0)

    # Asymmetric Clipping (for negative targets)
    if clip is not None and clip > 0:
        pred_sigmoid = torch.clamp(pred_sigmoid, min=clip, max=1-clip)

    # Loss for positive targets
    pos_loss = -torch.log(pred_sigmoid + eps) * torch.pow(1 - pred_sigmoid, gamma_pos)
    # Loss for negative targets
    neg_loss = -torch.log(1 - pred_sigmoid + eps) * torch.pow(pred_sigmoid, gamma_neg)

    # Only keep losses for relevant targets
    loss = torch.where(pos_inds, pos_loss, torch.zeros_like(pos_loss)) + \
           torch.where(neg_inds, neg_loss, torch.zeros_like(neg_loss))

    # Optionally, mask out unlabeled (nan) targets
    is_labeled = true == true  # filter out nans
    loss = loss[is_labeled]

    # Mean over all labeled elements
    loss = loss.mean()

    return loss, pred
    