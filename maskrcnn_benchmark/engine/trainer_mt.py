# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

import numpy as np

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def update_ema_variables(model_params, ema_model_params, alpha, global_step):
    for ema_param, param in zip(ema_model_params, model_params):
    # model_param_table = model.named_parameters()
    # for ema_name, ema_param in ema_model.named_parameters():
    #     param = model_param_table[ema_name]
        # print((ema_param-param).mean())
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        # print((ema_param-param).mean())

def do_train(
    cfg,
    model,
    # ema_model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    tflogger=None,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    if cfg.MODEL.MT_ON:
        # get ema params
        model_params, ema_params = model.mt_params_to_update()
        model.init_mt_params()
        ema_model = None
    else:
        # create ema model by cloning
        import copy
        ema_model = copy.deepcopy(model)
        model_params = [p for p in model.parameters()]
        ema_params = [p for p in ema_model.parameters()]
        for param in ema_params:
            # param.data = param.clone().data
            param.detach_()
            # param.requires_grad = False
        ema_model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        # compability
        kwargs = dict()
        if isinstance(images, dict):
            tmp = images.pop("images")
            kwargs.update(images)
            images = tmp
        if isinstance(targets, dict):
            tmp = targets.pop("targets")
            kwargs.update(targets)
            targets = tmp
        images = images.to(device)
        targets = [target.to(device) for target in targets]

        model_out = model(images, targets, **kwargs)
        loss_dict = model_out["losses"]
        # model_output = model_out["result"]
        model_out = model_out["output"]
        # print(model_out)

        if not ema_model is None:
            ema_model_out = ema_model(images, targets, **kwargs)["output"]
            # print(ema_model_out)
            # ema_loss_dict = ema_model_out["losses"]
            # ema_loss = sum(loss for loss in ema_loss_dict.values())
            if isinstance(model_out, dict):
                ema_loss = dict(ema_cons_loss=sum([F.mse_loss(model_out[key], ema_model_out[key]) for key in model_out.keys()]) / len(model_out.keys()))
            else:
                ema_loss = dict(ema_cons_loss=F.mse_loss(model_out, ema_model_out))
        else:
            ema_loss = {k:v for k,v in loss_dict.items() if k.startswith("ema_")} # loss_dict["ema_cons_loss"]

        # consistency loss
        if cfg.TRAINER.MT.CONSISTENCY_WEIGHT:
            consistency_weight = cfg.TRAINER.MT.CONSISTENCY_WEIGHT * sigmoid_rampup((iteration-cfg.TRAINER.MT.EMA_DECAY_START), cfg.TRAINER.MT.CONSISTENCY_RAMPUP)
        else:
            consistency_weight = 0

        ema_loss = {k:(consistency_weight*v) for k,v in ema_loss.items()}
        loss_dict.update(ema_loss)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        # model_param_table = model.state_dict()
        # model_params, ema_model_params = [], []
        # for ema_name, ema_param in ema_model.named_parameters():
        #     ema_model_params.append(ema_param)
        #     model_params.append(model_param_table[ema_name])

        # Use the true average until the exponential average is more correct
        warmup_coef = cfg.TRAINER.MT.EMA_DECAY_WARMUP * (1-cfg.TRAINER.MT.EMA_DECAY)
        alpha = min(1 - 1 / ((iteration-cfg.TRAINER.MT.EMA_DECAY_START) / warmup_coef + 1), cfg.TRAINER.MT.EMA_DECAY) if iteration > cfg.TRAINER.MT.EMA_DECAY_START else 0
        update_ema_variables(model_params, ema_params, alpha, iteration)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            model.mt_params_to_update()
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            if tflogger:
                for name in loss_dict_reduced:
                    tflogger.add_scalar("losses/"+name, loss_dict_reduced[name], iteration)
                    tflogger.add_scalar("loss_sum", losses_reduced, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
