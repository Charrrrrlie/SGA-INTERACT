import logging
import torch
import numpy as np

_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"


def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

  
def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):
    logging.basicConfig(filename=path, 
                    level=level, 
                    format=format,
                    datefmt=datefmt)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

def tb_vis(tb_logger, cur_step, total_loss, loss_dict, info_dict, scheduler):
    if tb_logger is None:
        return

    if total_loss is not None:
        tb_logger.add_scalar('training_loss/total_loss', total_loss, cur_step)
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            tb_logger.add_scalar('training_loss/{}'.format(key), \
                                value.mean().detach().data.cpu().numpy(), cur_step)
        else:
            tb_logger.add_scalar('training_loss/{}'.format(key), value, cur_step)
    if info_dict is not None:
        for key, value in info_dict.items():
            tb_logger.add_scalar('accuracy/{}'.format(key), value, cur_step)
    if scheduler is not None:
        tb_logger.add_scalar('meta/lr', scheduler.get_last_lr()[0], cur_step)

def lambda_with_warmup(ratio, max_epoch, warm_up):
    warm_up_num = warm_up * ratio
    max_num = max_epoch * ratio
    lr_lambda = lambda num: num / warm_up_num \
                    if num < warm_up_num else \
                    0.5 * (np.cos((num - warm_up_num) / (max_num - warm_up_num) * np.pi) + 1)

    return lr_lambda