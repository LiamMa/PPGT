'''
    Customized trainer for BREC dataset ('https://github.com/GraphPKU/BREC')
'''


import time
from os.path import join

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_train, loader_dict

from torch_geometric.loader import DataLoader
from torch.nn import CosineEmbeddingLoss


# ---------- create model,optimizer,scheduler ---------
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from ppgt.optimizer.extra_optimizers import ExtendedSchedulerConfig


from loguru import logger

from tqdm import tqdm


# warnings.filterwarnings('ignore')


''' Global configuration '''
NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0. # not Loss Threshold for Graph Transformers
# LOSS_THRESHOLD = 1e-3
# LOSS_THRESHOLD = 5e-2

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
''' ----------------------------- '''


# def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
#     model.train()
#     optimizer.zero_grad()
#     time_start = time.time()
#     for iter, batch in enumerate(loader):
#         # ipdb.set_trace()
#
#         if cfg.train.get('drop_last', False) and 'batch' in batch:
#             # to set drop_last without changing the DataLoader
#             if max(batch.batch)+1 < cfg.train.batch_size:
#                 continue
#
#         batch.split = 'train'
#         batch.to(torch.device(cfg.device))
#         pred, true = model(batch)
#
#         if cfg.dataset.name == 'ogbg-code2':
#             loss, pred_score = subtoken_cross_entropy(pred, true)
#             _true = true
#             _pred = pred_score
#         else:
#             loss, pred_score = compute_loss(pred, true)
#             _true = true.detach().to('cpu', non_blocking=True)
#             _pred = pred_score.detach().to('cpu', non_blocking=True)
#
#         if 'reg_term' in batch:
#             if isinstance(batch.reg_term, list):
#                 batch.reg_term = sum(batch.reg_term) # / len(batch.reg_term)
#             loss += batch.reg_term * cfg.optim.get('weight_decay', 0.)
#
#
#         # if 'filter_reg' in batch:
#         #     reg = batch.filter_reg
#         #     loss += reg
#
#         loss.backward()
#         # Parameters update after accumulating gradients for given num. batches.
#         if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
#             if cfg.optim.clip_grad_norm:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.get('clip_grad_norm_value', 1.0))
#
#             optimizer.step()
#             optimizer.zero_grad()
#         logger.update_stats(true=_true,
#                             pred=_pred,
#                             loss=loss.detach().cpu().item(),
#                             lr=scheduler.get_last_lr()[0],
#                             time_used=time.time() - time_start,
#                             params=cfg.params,
#                             dataset_name=cfg.dataset.name)
#         time_start = time.time()
#
#
# @torch.no_grad()
# def eval_epoch(logger, loader, model, split='val'):
#     model.eval()
#     time_start = time.time()
#     for batch in loader:
#         batch.split = split
#         batch.to(torch.device(cfg.device))
#         if cfg.gnn.head == 'inductive_edge':
#             pred, true, extra_stats = model(batch)
#         else:
#             pred, true = model(batch)
#             extra_stats = {}
#         if cfg.dataset.name == 'ogbg-code2':
#             loss, pred_score = subtoken_cross_entropy(pred, true)
#             _true = true
#             _pred = pred_score
#         else:
#             loss, pred_score = compute_loss(pred, true)
#             _true = true.detach().to('cpu', non_blocking=True)
#             _pred = pred_score.detach().to('cpu', non_blocking=True)
#         logger.update_stats(true=_true,
#                             pred=_pred,
#                             loss=loss.detach().cpu().item(),
#                             lr=0, time_used=time.time() - time_start,
#                             params=cfg.params,
#                             dataset_name=cfg.dataset.name,
#                             **extra_stats)
#         time_start = time.time()
#






@register_train('brec')
def brec_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.
    Args:
        loggers: List of loggers
        loaders: the BREC dataset (cfg.dataset.task=='bypass')
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler


    Modified from https://github.com/GraphPKU/BREC/blob/d09e8c349a8bbc0882d2932f7b37b2726f576ce9/KP-GNN/test_BREC.py#L387
    """
    dataset = loaders
    LOG_name = join(cfg.run_dir, 'log.txt')
    logger.remove(handler_id=None)
    logger.add(LOG_name)

    loss_threshold = cfg.train.get('loss_threshold', LOSS_THRESHOLD)

    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=cfg.train.batch_size)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(cfg.device))[0].detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()
    # Do something

    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    result_dict = dict()
    index = ['part_name', 'num_crt', 'num_total', 'crt_rate', 'fail_rlb', 'time_cost']

    cfi_only = cfg.train.get('cfi_only', False)


    # ------- to use subset ---- set dataset.name as "subset-{}"  from  ['Basic', 'Regular', 'Extension', 'CFI', '4-Vertex_Condition', 'Distance_Regular']
    name = cfg.dataset.name

    threshold = 0
    if 'subset' in name:
        subset_name = name.replace("subset-", "")
        subset_name = subset_name.lower()
        if 'hard_' in subset_name:
            subset_name = subset_name.replace('hard_', '')
            if subset_name.lower() == 'regular':
                threshold = 50
    else:
        subset_name = None


    for idx in index:
        result_dict[idx]=[]

    print(part_dict.keys())


    for part_name, part_range in part_dict.items():
        if cfi_only:
            if 'cfi' not in part_name.lower():
                continue

        if subset_name is not None:
            if part_name.lower() != subset_name.lower():
                continue

        if threshold > 0:
            part_name = 'Hard_' + part_name

        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()


        part_range = (part_range[0]+threshold, part_range[1])

        pbar = tqdm(range(part_range[0], part_range[1]))
        total_ex = part_range[1] - part_range[0]

        for id in pbar:


            logger.info(f"ID: {id - part_range[0]}")
            model = create_model(dim_out=cfg.gnn.get('out_dim', 16))
            optimizer = create_optimizer(model.parameters(),
                                         new_optimizer_config(cfg))
            scheduler = create_scheduler(optimizer, new_scheduler_config(cfg))

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=args.factor)

            dataset_traintest = dataset[
                id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
            ]
            dataset_reliability = dataset[
                (id + SAMPLE_NUM)
                * NUM_RELABEL
                * 2 : (id + SAMPLE_NUM + 1)
                * NUM_RELABEL
                * 2
            ]

            if cfg.dataset.brec_pe_transform:
                # dataset_traintest = dataset_traintest.copy()  # to avoid affecting original datasets
                # # dataset_reliability = dataset_reliability.copy()
                s_t = time.time()
                pe_transform = loader_dict['pe_transform']
                dataset_traintest = pe_transform(dataset_traintest)
                dataset_reliability = pe_transform(dataset_reliability)
                e_t = time.time()
                if e_t - s_t > 30:
                    print(f'Pre-Process done: {e_t - s_t} sec - Id:{id}')



            # model.reset_parameters() # directly re-instantiate the model instead of reset_parameters()
            model.train()
            best_loss = 1e8


            for i in range(cfg.optim.max_epoch):
                traintest_loader = DataLoader(dataset_traintest, batch_size=cfg.train.batch_size)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(cfg.device))[0]

                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(cfg.device),
                    )

                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()

                loss_all /= NUM_RELABEL
                logger.info(f"Loss: {loss_all}")

                if loss_all < best_loss:
                    best_loss = loss_all
                    state_dict = model.state_dict()

                if loss_all < loss_threshold:
                    logger.info("Early Stop Here")
                    break

                if "timm" in cfg.optim.scheduler:
                    scheduler.step(epoch=i+1, metric=loss_all)
                elif cfg.optim.scheduler == 'reduce_on_plateau' or "timm" in cfg.optim.scheduler:
                    scheduler.step(loss_all)
                else:
                    scheduler.step()

                # Todo: add model selection based on validation set ....
                #     The original implementation in BREC doesn't include this

                # scheduler.step(loss_all)
            model.load_state_dict(state_dict)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, True)
            T_square_reliability = T2_calculation(dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            # if not isomorphic_flag: breakpoint()

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id - part_range[0])

            logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1

            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

            pbar.set_description(f"[{part_name} - correct: {cnt_part}/{total_ex}] - fail-reliability: {fail_in_reliability_part}")


        end = time.process_time()
        time_cost_part = round(end - start, 2)


        # index = ['part_name', 'num_crt', 'num_total', 'crt_rate', 'time_cost']
        num_examples = part_range[1] - part_range[0]
        result_dict['part_name'].append(part_name)
        result_dict['num_crt'].append(cnt_part)
        result_dict['fail_rlb'].append(fail_in_reliability_part)
        result_dict['num_total'].append(num_examples)
        result_dict['crt_rate'].append(cnt_part/num_examples)
        result_dict['time_cost'].append(time_cost_part)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {num_examples}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {num_examples}"
        )


        print(result_dict)

    for k in range(len(result_dict['part_name'])):
        part_name, time_cost_part = result_dict['part_name'][k], result_dict['time_cost'][k]
        cnt_part, total_part = result_dict['num_crt'][k], result_dict['num_total'][k]

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {total_part}"
        )

    time_cost_part = sum(result_dict['time_cost'])
    cnt_part, total_part = sum(result_dict['num_crt']), sum(result_dict['num_total'])

    logger.info(
        f"all parts costs time {time_cost_part}; Correct in {cnt_part} / {total_part}"
    )




def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer,
                           base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay,
                           momentum=cfg.optim.momentum)


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

