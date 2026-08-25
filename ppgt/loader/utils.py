

import torch

from torch_geometric.graphgym.config import cfg
# from box import Box


'''
    import necessary modules from graphgym
'''
from torch_geometric.graphgym.loader import create_dataset 
from yacs.config import CfgNode as CN


from .custom_loader import get_loader


def create_loader(rank=None, world_size=None):
    """Create the train/val/test data loaders.

    Customized from ``torch_geometric.graphgym.loader.create_loader``.

    Args:
        rank: process rank for multi-GPU (DDP) training; ``None`` for a
            single-device run.
        world_size: total number of DDP processes; ``None`` for a single-device
            run.

    Returns:
        List of PyTorch data loaders, ``[train, val, test]``.
    """
    distributed = rank is not None and world_size is not None
    dataset = create_dataset()

    if cfg.dataset.task == 'bypass':
        # ``bypass`` mode: return dataset directly
        return dataset

    if distributed and cfg.dataset.task != 'graph':
        raise NotImplementedError(
            "Multi-GPU training is only implemented for graph-level tasks "
            f"(got `dataset.task: {cfg.dataset.task}`). Run on a single "
            "device by passing a single `accelerator`, e.g. "
            "`accelerator cuda:0`."
        )

    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        train_sampler = None
        if distributed:
            # Each rank sees a disjoint shard; the sampler owns the shuffling.
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset[id], num_replicas=world_size, rank=rank, shuffle=True)
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, distributed_sampler=train_sampler)
        ]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True)
        ]
    
    if 'val' not in cfg:
        cfg.val = CN()
    if "batch_size" not in cfg.val:
        cfg.val.batch_size = cfg.train.batch_size


    # val and test loaders
    for i in range(cfg.share.num_splits - 1):
        if cfg.dataset.task == 'graph':

            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]] 
            loaders.append(
                get_loader(dataset[id], cfg.val.sampler, cfg.val.batch_size,
                           shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(
                get_loader(dataset, cfg.val.sampler, cfg.val.batch_size,
                           shuffle=False))


    return loaders