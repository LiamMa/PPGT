import os.path as osp


from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import (
    ClusterLoader,
    DataLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborSampler,
    RandomNodeLoader,
)

from torch_geometric.utils import (
    index_to_mask,
)

index2mask = index_to_mask  # TODO Backward compatibility



def get_loader(dataset, sampler, batch_size, shuffle=True,
               distributed_sampler=None):
    """Build a data loader.

    Args:
        distributed_sampler: a ``DistributedSampler`` for multi-GPU (DDP) runs.
            When given, it takes over shuffling, so ``shuffle`` must be False
            for the underlying :class:`DataLoader`.
    """
    pw = False
    if sampler == "full_batch" or len(dataset) > 1:
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle if distributed_sampler is None
                                  else False,
                                  sampler=distributed_sampler,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False, persistent_workers=pw)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True, persistent_workers=pw)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True,
                                        persistent_workers=pw)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True,
                                  persistent_workers=pw)
    elif sampler == "cluster":
        loader_train = ClusterLoader(
            dataset[0],
            num_parts=cfg.train.train_parts,
            save_dir=osp.join(
                cfg.dataset.dir,
                cfg.dataset.name.replace("-", "_"),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=pw,
        )

    else:
        raise NotImplementedError(f"'{sampler}' is not implemented")

    return loader_train

