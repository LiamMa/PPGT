import logging
import time
from os.path import join

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train

from ppgt.utils import cfg_to_dict, make_wandb_name, mlflow_log_cfgdict

from .custom_train import eval_epoch

# warnings.filterwarnings('ignore')







@register_train('inference-visual')
def inference_visual(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    if cfg.mlflow.use:
        try:
            import mlflow
        except:
            raise ImportError('MLflow is not installed.')
        if cfg.mlflow.name == '':
            MLFLOW_NAME = make_wandb_name(cfg)
        else:
            MLFLOW_NAME = cfg.mlflow.name

        if cfg.name_tag != '':
            MLFLOW_NAME = MLFLOW_NAME + '-' + cfg.name_tag

        if cfg.mlflow.get('out_dir', None) is not None:
            mlflow.set_tracking_uri(cfg.mlflow.out_dir)

        experiment = mlflow.set_experiment(cfg.mlflow.project)
        mlflow.start_run(run_name=MLFLOW_NAME)
        mlflow.pytorch.log_model(model, "model")
        mlflow_log_cfgdict(cfg_to_dict(cfg), mlflow_func=mlflow)
        if cfg.get('cfg_file', None) is not None: mlflow.log_artifact(cfg.cfg_file) # log the whole config-file
        mlflow.log_artifact(join(cfg.run_dir, 'logging.log')) # log the whole config-file


    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    # split_names = ['test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    cfg.visual_attn = True

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i], one_batch=True)
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()



