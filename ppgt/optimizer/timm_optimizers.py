import math

from torch import nn
from torch.optim import AdamW, Optimizer

import torch_geometric.graphgym.register as register

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.optim import create_optimizer_v2

from torch_geometric.graphgym.config import cfg


@register.register_optimizer('timm_adamW')
def timm_adamW_optimizer(model: nn.Module, base_lr: float,
                         weight_decay: float=0.,
                         ) -> AdamW:
    # note: use timm optimizer creater to allow [filter_bias_and_bn=True] for weight decay
    #       - will exclude 1d vectors (bias and affine vectors in linear and/or norm-layers, etc) in weight-decaying
    beta1 = cfg.optim.get('beta1', 0.9)
    beta2 = cfg.optim.get('beta2', 0.999)

    return create_optimizer_v2(model, opt='adamw', lr=base_lr,
                               weight_decay=weight_decay,
                               filter_bias_and_bn=True,
                               betas= (beta1, beta2)
                               )





@register.register_optimizer('timm_muon')
def timm_muon_optimizer(model: nn.Module, base_lr: float,
                        weight_decay: float=0.,
                        ) -> Optimizer:
    # note: use timm optimizer creater to allow [filter_bias_and_bn=True] for weight decay
    #       - will exclude 1d vectors (bias and affine vectors in linear and/or norm-layers, etc) in weight-decaying

    return create_optimizer_v2(model, opt='muon', lr=base_lr,
                               weight_decay=weight_decay,
                               filter_bias_and_bn=True
                               )






@register.register_scheduler('timm_cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int,
                                 min_lr:float=0.,
                                 num_cycles=1.,
                                 reduce_factor=1., # not used in cosine with warmup    
                                 warmup_lr_init=1e-7,
                                 ):

    if num_cycles == 0.5:
        num_cycles = 1.
        # the timm.CosineLRScheduler has different interpretation of num_cycles compared with customized version

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=(max_epoch-num_warmup_epochs) // num_cycles,
        warmup_t=num_warmup_epochs,
        lr_min=min_lr,
        warmup_lr_init=warmup_lr_init,
        warmup_prefix=True,
        cycle_limit=num_cycles,
        # cycle_decay=reduce_factor,
    )

    if not hasattr(scheduler, 'get_last_lr'):
        # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
        # pytorch1.10; we add it here for consistency with other schedulers.
        def get_last_lr(self):
            """ Return last computed learning rate by current scheduler.
            """
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            return self._last_lr

        scheduler.get_last_lr = get_last_lr.__get__(scheduler)

    def modified_state_dict(ref):
        """Returns the state of the scheduler as a :class:`dict`.
        Additionally modified to ignore 'get_last_lr', 'state_dict'.
        Including these entries in the state dict would cause issues when
        loading a partially trained / pretrained model from a checkpoint.
        """
        return {key: value for key, value in ref.__dict__.items()
                if key not in ['sparsifier', 'get_last_lr', 'state_dict']}

    scheduler.state_dict = modified_state_dict.__get__(scheduler)


    return scheduler






@register.register_scheduler('timm_warmup_stable_decay')
def warmup_stable_decay_scheduler(
        optimizer: Optimizer,
        num_warmup_epochs: int,
        max_epoch: int,
        min_lr: float = 0.,
        num_cycles: float = 0.0,
        reduce_factor: float = 0.9,
        warmup_lr_init: float = 1e-8,
):
    """Warmup–Stable–Decay (WSD) schedule over epochs.

    Signature matches `timm_cosine_with_warmup` for drop-in config compatibility.

    Schedule:
    - Warmup: linearly increase from warmup_lr_init -> base_lr
    - Stable: keep base_lr constant for some epochs
    - Decay: cosine (default) or linear decay from base_lr -> min_lr

    How WSD parameters are encoded with existing args:
    - `reduce_factor` controls stable length.
      - If 0 < reduce_factor < 1: interpreted as fraction of (max_epoch - num_warmup_epochs)
      - If reduce_factor >= 1: interpreted as absolute number of stable epochs
    - `num_cycles` controls decay type:
      - num_cycles == 0: linear decay
      - otherwise: cosine decay

    Notes:
    - Implemented with a timm-compatible `.step(epoch=..., metric=...)` API.
    """
    if max_epoch <= 0:
        raise ValueError(f"max_epoch must be positive, got {max_epoch}")
    if num_warmup_epochs < 0:
        raise ValueError(
            f"num_warmup_epochs must be >= 0, got {num_warmup_epochs}"
        )

    base_lr = optimizer.param_groups[0]["lr"]
    if base_lr <= 0:
        raise ValueError(f"optimizer base lr must be > 0, got {base_lr}")

    warmup_steps = int(num_warmup_epochs)
    total_steps = int(max_epoch)
    remaining = max(0, total_steps - warmup_steps)

    if 0.0 < float(reduce_factor) < 1.0:
        stable_steps = int(round(float(reduce_factor) * remaining))
    else:
        stable_steps = int(max(0, round(float(reduce_factor))))
    stable_steps = min(stable_steps, remaining)

    decay_steps = max(0, total_steps - warmup_steps - stable_steps)
    decay = "linear" if float(num_cycles) == 0.0 else "cosine"

    warmup_ratio = float(warmup_lr_init) / \
        float(base_lr) if warmup_steps > 0 else 1.0
    min_ratio = float(min_lr) / float(base_lr) if base_lr > 0 else 0.0
    if min_ratio < 0:
        raise ValueError(f"min_lr must be >= 0, got {min_lr}")

    def _ratio_at_epoch(epoch_idx: int) -> float:
        # Warmup (epoch_idx counted from 0)
        if warmup_steps > 0 and epoch_idx < warmup_steps:
            t = float(epoch_idx) / float(max(1, warmup_steps))
            return warmup_ratio + t * (1.0 - warmup_ratio)

        # Stable
        if epoch_idx < warmup_steps + stable_steps:
            return 1.0

        # Decay
        if decay_steps <= 0:
            return max(min_ratio, 1.0)

        decay_epoch = epoch_idx - warmup_steps - stable_steps
        progress = float(decay_epoch) / float(max(1, decay_steps))
        progress = min(max(progress, 0.0), 1.0)

        if decay == "cosine":
            ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay == "linear":
            ratio = 1.0 - progress
        else:
            raise ValueError(f"Unsupported decay='{decay}'")

        return (1.0 - min_ratio) * ratio + min_ratio

    class _TimmCompatibleWsdScheduler:
        def __init__(self, optimizer_: Optimizer):
            self.optimizer = optimizer_
            self.base_values = [group["lr"] for group in optimizer_.param_groups]
            # Ensure epoch 0 uses warmup start lr (training loop calls step() after epoch 0).
            self.epoch = 0
            ratio0 = _ratio_at_epoch(0)
            self._last_lr = []
            for base_lr_i, group in zip(self.base_values, self.optimizer.param_groups):
                lr_i = float(base_lr_i) * float(ratio0)
                group["lr"] = lr_i
                self._last_lr.append(lr_i)

        def step(self, epoch=None, metric=None):
            # timm style: step(epoch=..., metric=...)
            # Also tolerate step(metric) or step() usage elsewhere.
            if epoch is None and isinstance(metric, (int, float)):
                # called as step(metric) by non-timm paths; ignore metric and advance epoch
                epoch = self.epoch + 1
            elif epoch is None:
                epoch = self.epoch + 1

            self.epoch = int(epoch)
            ratio = _ratio_at_epoch(self.epoch)
            self._last_lr = []
            for base_lr_i, group in zip(self.base_values, self.optimizer.param_groups):
                lr_i = float(base_lr_i) * float(ratio)
                group["lr"] = lr_i
                self._last_lr.append(lr_i)

        def get_last_lr(self):
            return self._last_lr

        def state_dict(self):
            return {
                "epoch": self.epoch,
                "base_values": self.base_values,
                "_last_lr": self._last_lr,
                # schedule params (for transparency / robustness)
                "warmup_steps": warmup_steps,
                "stable_steps": stable_steps,
                "decay_steps": decay_steps,
                "warmup_ratio": warmup_ratio,
                "min_ratio": min_ratio,
                "decay": decay,
            }

        def load_state_dict(self, state_dict):
            self.epoch = int(state_dict.get("epoch", -1))
            self.base_values = list(state_dict.get("base_values", self.base_values))
            self._last_lr = list(state_dict.get("_last_lr", self._last_lr))
            # Do not overwrite schedule hyperparams computed from current cfg.

    return _TimmCompatibleWsdScheduler(optimizer)