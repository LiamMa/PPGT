"""PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which breaks loading
older checkpoints that contain NumPy arrays, optimizer state, etc. For trusted local
checkpoints, restore the legacy default (full unpickling) when the caller omits
``weights_only``.
"""

import torch

_ORIG_LOAD = torch.load


def _load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    try:
        return _ORIG_LOAD(*args, **kwargs)
    except TypeError:
        # Very old PyTorch without ``weights_only``
        kwargs.pop("weights_only", None)
        return _ORIG_LOAD(*args, **kwargs)


def patch_torch_load_for_trusted_checkpoints():
    torch.load = _load
