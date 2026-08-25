from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_wandb')
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False

    # W&B entity (team or user) to log to. Must already exist; set it to
    # your own before enabling `wandb.use`.
    cfg.wandb.entity = ""

    # W&B project name; created in your entity if it does not exist yet.
    cfg.wandb.project = "ppgt"

    # Optional run name
    cfg.wandb.name = ""
