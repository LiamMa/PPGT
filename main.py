#!/usr/bin/env python
import datetime
import os
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# note: set cuda environ for reproducing experiments with non-deterministic operations
#    need to impornt before pytorch
#    will be slower than not setting it

# import graph_tool as gt
# import graph_tool.topology as gt_topology
# import graph_tool as gt
# Note: ------- need to import graphtool ahead of torch to avoid bugs:
#   >>>  version `GOMP_5.0' not found .....
import torch
import logging

import ppgt  # noqa: F401  (imported for its side effect: registers custom modules)
from ppgt.optimizer.extra_optimizers import ExtendedSchedulerConfig

from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             # set_agg_dir,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
# from torch_geometric.graphgym.loader import create_loader
from ppgt.loader.utils import create_loader # use customized create_loader instead of the default one.
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.train import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric import seed_everything

from ppgt.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained, set_new_cfg_allowed
from ppgt.logger import create_logger
# try:
#     import deepspeed
# except ImportError:
#     pass

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum,
                           )



def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
        num_cycles=cfg.optim.num_cycles,
        min_lr_mode=cfg.optim.min_lr_mode
    )


def custom_set_out_dir(cfg, cfg_fname, name_tag):
    """Set custom main output directory path to cfg.
    Include the config filename and name_tag in the new :obj:`cfg.out_dir`.

    Args:
        cfg (CfgNode): Configuration node
        cfg_fname (string): Filename for the yaml format configuration file
        name_tag (string): Additional name tag to identify this execution of the
            configuration file, specified in :obj:`cfg.name_tag`
    """
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(cfg.out_dir, run_name)


def custom_set_run_dir(cfg, run_id):
    """Custom output directory naming for each experiment run.

    Args:
        cfg (CfgNode): Configuration node
        run_id (int): Main for-loop iter id (the random seed or dataset split)
    """
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Make output directory
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def run_loop_settings():
    """Create main loop execution settings based on the current cfg.

    Configures the main execution loop to run in one of two modes:
    1. 'multi-seed' - Reproduces default behaviour of GraphGym when
        args.repeats controls how many times the experiment run is repeated.
        Each iteration is executed with a random seed set to an increment from
        the previous one, starting at initial cfg.seed.
    2. 'multi-split' - Executes the experiment run over multiple dataset splits,
        these can be multiple CV splits or multiple standard splits. The random
        seed is reset to the initial cfg.seed value for each run iteration.

    Returns:
        List of run IDs for each loop iteration
        List of rng seeds to loop over
        List of dataset split indices to loop over
    """
    if len(cfg.run_multiple_splits) == 0:
        # 'multi-seed' run mode
        num_iterations = args.repeat
        seeds = [cfg.seed + x for x in range(num_iterations)]
        split_indices = [cfg.dataset.split_index] * num_iterations
        run_ids = seeds
    else:
        # 'multi-split' run mode
        if args.repeat != 1:
            raise NotImplementedError("Running multiple repeats of multiple "
                                      "splits in one run is not supported.")
        num_iterations = len(cfg.run_multiple_splits)
        seeds = [cfg.seed] * num_iterations
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices

def setup_multi_gpu(rank, world_size, device):
    """Initialise the process group for multi-device (DDP) training.

    ``MASTER_ADDR`` / ``MASTER_PORT`` can be overridden from the environment so
    that several jobs can run on the same machine without clashing.
    """
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    backend = "nccl" if str(device).startswith("cuda") else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup_multi_gpu():
    dist.destroy_process_group()

def run(rank:int=None, world_size:int=None, cfg=None):
    # Set configurations for each run


    seed_everything(cfg.seed)
    # note: enable to not use `auto_select_device`; for multiple jobs on multiple gpus
    #   please set `accelerator={device}` instead

    if rank is not None:
        # Multi-device (DistributedDataParallel) training: one process per
        # device. `mp.spawn` starts fresh interpreters, so the GraphGym global
        # config singleton that the rest of the code reads (`from
        # torch_geometric.graphgym.config import cfg`) still holds the
        # defaults here -- only this function received the loaded config, as a
        # pickled copy. Merge it back into the singleton before anything else
        # touches it.
        from torch_geometric.graphgym.config import cfg as global_cfg
        set_new_cfg_allowed(global_cfg, True)
        global_cfg.merge_from_other_cfg(cfg)
        cfg = global_cfg

        # Pin this process to a single device. `cfg.accelerator` is read
        # directly by GraphGym's `create_model`, so it has to be a plain
        # device string here, not the comma-separated list.
        device = cfg.accelerator.split(",")[rank].strip()
        cfg.device = device
        cfg.accelerator = device
        setup_multi_gpu(rank, world_size, device)
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
    else:
        device = cfg.device


    # ------------------------------------------------------------

    if cfg.pretrained.dir:
        cfg = load_pretrained_model_cfg(cfg)


    logging.info(f"[*] Run ID {cfg.run_id}: seed={cfg.seed}, "
                 f"split_index={cfg.dataset.split_index}")
    logging.info(f"    Starting now: {datetime.datetime.now()}")
    # Set machine learning pipeline

    loaders = create_loader(rank, world_size)
    # Every rank needs a logger object; only rank 0 actually writes epoch stats.
    loggers = create_logger()

    if cfg.train.mode == "download_data":
        return None

    model = create_model()
    model = model.to(device)

    if rank is not None:
        torch_device = torch.device(device)
        model = DDP(model,
                    device_ids=[torch_device.index]
                    if torch_device.type == "cuda" else None)


    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head,
            seed=cfg.seed
        )

    optimizer = create_optimizer(model.parameters(),
                                    new_optimizer_config(cfg))


    scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))
    # Print model info
    logging.info(model)
    logging.info(cfg)
    cfg.params = params_count(model)
    logging.info('Num parameters: %s', cfg.params)
    # Start training
    if cfg.train.mode == 'standard':
        if cfg.wandb.use:
            logging.warning("[W] WandB logging is not supported with the "
                            "default train.mode, set it to `custom`")
        if cfg.mlflow.use:
            logging.warning("[ML] MLflow logging is not supported with the "
                            "default train.mode, set it to `custom`")
        train(loggers, loaders, model, optimizer, scheduler)
    else:
        if rank is not None:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler, rank=rank,
                                       world_size=world_size)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)





if __name__ == '__main__':


    # Load cmd line args
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    # ----- note: allow to change config -----------
    cfg.set_new_allowed(True)
    cfg.work_dir = os.getcwd()
    # -----------------------------
    load_cfg(cfg, args)
    cfg.cfg_file = args.cfg_file
    custom_set_out_dir(cfg, args.cfg_file, cfg.name_tag)
    dump_cfg(cfg)

    # if cfg.get('use_deterministic_algorithms', True):
    #     torch.use_deterministic_algorithms(True)
    world_size = None
    if cfg.get("auto_select_device", False):
        auto_select_device()
    else:
        # `accelerator` is a single device ("cuda:0") or a comma-separated
        # list ("cuda:0,cuda:1") requesting one DDP process per device.
        devices = [d.strip() for d in str(cfg.accelerator).split(",") if d.strip()]
        cfg.device = devices[0]
        if len(devices) > 1:
            world_size = len(devices)


    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    # Repeat for multiple experiment runs
    for run_id, seed, split_index in zip(*run_loop_settings()):
        custom_set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        if world_size is None:
            # single-GPU trainin
            print(f"Running single-GPU training with seed {seed}")
            run(None, None, cfg)
        else:
            # multi-GPU training
            import torch.multiprocessing as mp
            mp.spawn(run, args=(world_size, cfg), nprocs=world_size, join=True)




    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
    logging.info(f"[*] All done: {datetime.datetime.now()}")
