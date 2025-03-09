import os
import logging
import numpy as np

from easydict import EasyDict as edict
from tqdm import trange, tqdm
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import tb_vis, lambda_with_warmup
from metrics import LocEval

class Trainer:
    def __init__(
        self,
        config: edict,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_dir: str,
        logger: Optional[logging.Logger] = None,
        checkpoint_path: str = None,
        mode: str = 'train',
    ) -> None:

        self.gpu_id = int(os.environ['LOCAL_RANK'])

        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizer = optimizer

        self.epochs_run = 0
        self.config = config

        self.save_dir = save_dir

        # init scheduler
        if not 'scheduler' in config.train_params or mode == 'eval':
            self.scheduler = None
        elif config.train_params.scheduler.name == 'MultiStepLR':
            self.scheduler = MultiStepLR(self.optimizer,
                                         milestones=config.train_params.scheduler.milestones,
                                         gamma=config.train_params.scheduler.gamma)
        elif config.train_params.scheduler.name == 'LambdaLR':
            # NOTE(yyc): in MPGCN source code, they set ratio = len(train_loader)
            # while it makes the scheduler kept using warm_up steps / linear schedule.
            # It seems to be a potential bug and we deprecated this setting in our new baseline.
            if 'use_ratio' in config.train_params.scheduler and config.train_params.scheduler.use_ratio:
                ratio = len(train_loader)
            else:
                ratio = 1
            lr_lambda = lambda_with_warmup(ratio,
                                           config.train_params.num_epochs,
                                           config.train_params.scheduler.warm_up)
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise NotImplementedError

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path, mode, logger)

        # wrap model
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.best_stats = {}

    def _load_checkpoint(self, checkpoint_path, mode, logger):
        loc = f'cuda:{self.gpu_id}'
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        self.model.load_state_dict(checkpoint['model'])

        # NOTE(yyc): do not load optimizer during finetune
        if mode == 'train':
            self.epochs_run = checkpoint['epochs']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if logger is not None:
                logger.info(f'Resuming training from checkpoint at Epoch {self.epochs_run}')
        elif mode == 'finetune':
            if logger is not None:
                logger.info(f'Finetuning from checkpoint at Epoch {self.epochs_run}')
        elif mode == 'eval':
            if logger is not None:
                logger.info(f'Evaluating from checkpoint at Epoch {self.epochs_run}')
        else:
            raise NotImplementedError

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'model': self.model.module.state_dict(),
            'epochs': epoch,
            'optimizer': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        if is_best:
            torch.save(checkpoint, os.path.join(self.save_dir, 'best_ckpt.pth.tar'))
        else:
            torch.save(checkpoint, os.path.join(self.save_dir, '{:05d}_ckpt.pth.tar'.format(epoch)))

    def convert_data_to_device(self, x):
        for key in x:
            if isinstance(x[key], torch.Tensor):
                x[key] = x[key].to(self.gpu_id)
            elif isinstance(x[key], dict):
                x[key] = self.convert_data_to_device(x[key])
            elif isinstance(x[key], np.ndarray):
                x[key] = torch.tensor(x[key]).to(self.gpu_id)

        return x

    def train(self, tb_logger, logger):
        if self.gpu_id == 0:
            logger.info('Start training')
        num_epochs = self.config.train_params.num_epochs
        ckpt_save_freq = self.config.train_params.ckpt_save_freq

        for epoch in trange(self.epochs_run, num_epochs, disable=(self.gpu_id != 0)):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()

            self.train_step(epoch, tb_logger, logger)

            if self.scheduler is not None:
                self.scheduler.step()

            info, per_action_info = self.test(epoch, tb_logger, logger)

            if self.gpu_id == 0:
                self.update_stats(epoch, num_epochs, info, per_action_info, ckpt_save_freq, logger)

    def train_step(self, epoch, tb_logger, logger):
        for iter_num, x in enumerate(tqdm(self.train_loader, leave=False, disable=(self.gpu_id != 0))):
            cur_step = epoch * len(self.train_loader) + iter_num

            x = self.convert_data_to_device(x)

            # NOTE(yyc): only used for COMPOSER
            if self.config.model_params.model_name == 'composer':
                # normalize the prototypes
                with torch.no_grad():
                    w = self.model.module.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.model.module.prototypes.weight.copy_(w)

            loss_dict, info_dict = self.model(x)

            loss_values = [val.mean() for val in loss_dict.values()]
            loss = sum(loss_values)

            tb_log_total_loss = loss.item()

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.gpu_id == 0:
                tb_vis(tb_logger,
                        cur_step,
                        tb_log_total_loss,
                        loss_dict,
                        info_dict=None,
                        scheduler=self.scheduler)

                logger.info('Train [e{:02d}][{}/{}]'.format(epoch + 1, iter_num + 1, len(self.train_loader)))
                # NOTE(yyc): DEBUG use
                # if cur_step >= 10:
                #     raise ValueError

        if self.gpu_id == 0:
            logger.info('Epoch {} training finished'.format(epoch + 1))

    def test(self, epoch, tb_logger, logger):
        raise NotImplementedError

    def update_stats(self, epoch, num_epochs, info, per_action_info, ckpt_save_freq, logger):
        raise NotImplementedError

class TrainerGAR(Trainer):
    def __init__(
        self,
        config: edict,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_dir: str,
        logger: Optional[logging.Logger] = None,
        checkpoint_path: str = None,
        mode: str = 'train',
    ) -> None:

        super().__init__(config, model, train_loader, test_loader, optimizer, save_dir, logger, checkpoint_path, mode)

    def update_stats(self, epoch, num_epochs, info, per_action_info, ckpt_save_freq, logger):
        if len(self.best_stats) == 0 or info['group_acc1'] > self.best_stats['group_acc1']:
            self.best_stats = info
            self.best_stats['per_action_info'] = per_action_info
            self._save_checkpoint(epoch, is_best=True)
        if epoch % ckpt_save_freq == 0 or epoch == num_epochs - 1:
            self._save_checkpoint(epoch)

        for key in info:
            logger.info(f'Test {key}: {info[key]}')
        for key in self.best_stats:
            logger.info(f'Best {key}: {self.best_stats[key]}')

    @torch.no_grad()
    def test(self, epoch, tb_logger, logger, record_per_item_stats=False):
        if self.gpu_id == 0:
            logger.info('Start testing')
        self.model.eval()
        info = {}
        per_action_info = {}
        per_action_count = {
            'group': {},
        }
        count = 0

        if record_per_item_stats:
            assert int(os.environ['WORLD_SIZE']) == 1, 'record_per_item_stats only support single GPU'
            per_item_pred_list = []
            per_item_gt_list = []

        for iter_num, x in enumerate(tqdm(self.test_loader, leave=False, disable=(self.gpu_id != 0))):
            x = self.convert_data_to_device(x)

            # NOTE(yyc): only used for COMPOSER
            if self.config.model_params.model_name == 'composer':
                # normalize the prototypes
                with torch.no_grad():
                    w = self.model.module.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.model.module.prototypes.weight.copy_(w)
            if record_per_item_stats:
                x['record_per_item_stats'] = record_per_item_stats
                loss_dict, info_dict, per_item_stats = self.model(x)
                # to calculate confusion matrix
                per_item_pred_list.append(per_item_stats['pred_group'])
                per_item_gt_list.append(x['group_label'])
            else:
                loss_dict, info_dict = self.model(x)

            local_batch_size = x['group_label'].size(0)
            count += torch.tensor(local_batch_size).to(self.gpu_id)

            for key in info_dict:
                if key not in info:
                    info[key] = 0
                info[key] += info_dict[key].sum()
            # process per action acc
            for action_name in x['group_label_name']:
                if action_name not in per_action_count['group']:
                    per_action_count['group'][action_name] = 0
                per_action_count['group'][action_name] += torch.tensor(1).to(self.gpu_id)
            for key in info_dict:
                if key not in per_action_info:
                    per_action_info[key] = {}
                for idx_action, value_action in enumerate(info_dict[key]):
                    if 'group' in key:
                        action_name = x['group_label_name'][idx_action]
                        if action_name not in per_action_info[key]:
                            per_action_info[key][action_name] = 0
                        per_action_info[key][action_name] += value_action
                    elif 'person' in key:
                        pass
                    else:
                        raise ValueError('Unknown Labels')

        # vis confusion matrix
        if record_per_item_stats and self.gpu_id == 0 and 'Basketball' in self.config.dataset.name:
            from vis import draw_confusion_matrix
            per_item_pred = torch.cat(per_item_pred_list, dim=0).cpu().numpy()
            per_item_gt = torch.cat(per_item_gt_list, dim=0).cpu().numpy()
            label_names = [self.test_loader.dataset.id2action[i] for i in range(len(self.test_loader.dataset.id2action))]
            draw_confusion_matrix(per_item_pred, per_item_gt, label_names, self.save_dir)

        # calculate acc in all workers
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        for key in info:
            dist.all_reduce(info[key], op=dist.ReduceOp.SUM)
            info[key] /= count

        # TODO(yyc): nccl deadlock bugs when using 8 GPUs. Tried dist.barrier() but not work.
        # It is confusing that without train_step, the deadlock does not happen.
        for key in per_action_count:
            action_name_list = list(per_action_count[key].keys())
            for action_name in action_name_list:
                dist.all_reduce(per_action_count[key][action_name], op=dist.ReduceOp.SUM)

        for key in per_action_info:
            for action_name in per_action_info[key]:
                dist.all_reduce(per_action_info[key][action_name], op=dist.ReduceOp.SUM)
                per_action_info[key][action_name] /= per_action_count['group'][action_name]
            info[key + '_mean_acc'] = sum(per_action_info[key].values()) / (len(per_action_info[key]) + 1e-6)

        if self.gpu_id == 0:
            tb_vis(tb_logger,
                    epoch,
                    total_loss=None,
                    loss_dict={},
                    info_dict=info,
                    scheduler=None)
        return info, per_action_info


def gather_dict(data_dict, gpu_id, max_len, VALID_INF=999):
    new_data_dict = {}
    for key, value in data_dict.items():
        value = torch.stack(value)
        dtype = value.dtype
        if value.size(0) < max_len:
            pad = torch.ones(max_len - value.size(0)).to(gpu_id) * -1 * VALID_INF
            pad = pad.to(dtype)
            value = torch.cat([value, pad], dim=0)
        elif value.size(0) > max_len:
            raise ValueError('Data size is larger than max_len')

        tensor_list = [torch.zeros(max_len, dtype=dtype).to(gpu_id) for _ in range(dist.get_world_size())]
        dist.all_gather(tensor_list, value)
        gather_value = torch.concat(tensor_list)

        # remove padding
        valid_idx = gather_value != -1 * VALID_INF
        gather_value = gather_value[valid_idx]
        new_data_dict[key] = gather_value

    return new_data_dict

def convert_to_list(data):
    for key, value in data.items():
        data[key] = value.cpu().numpy().tolist()

class TrainerGAL(Trainer):
    def __init__(
        self,
        config: edict,
        model: torch.nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_dir: str,
        logger: Optional[logging.Logger] = None,
        checkpoint_path: str = None,
        mode: str = 'train',
    ) -> None:

        super().__init__(config, model, train_loader, test_loader, optimizer, save_dir, logger, checkpoint_path, mode)

        self.valid_action_threshold = config.dataset.valid_action_threshold
        self.tiou_thresholds = np.linspace(0.5, 0.95, 10)

    def update_stats(self, epoch, num_epochs, info, per_action_info, ckpt_save_freq, logger):
        if len(self.best_stats) == 0 or info['group_average_mAP'] > self.best_stats['group_average_mAP']:
            self.best_stats = info
            self._save_checkpoint(epoch, is_best=True)
        if epoch % ckpt_save_freq == 0 or epoch == num_epochs - 1:
            self._save_checkpoint(epoch)

        for key in info:
            logger.info(f'Test {key}: {info[key]}')
        for key in self.best_stats:
            logger.info(f'Best {key}: {self.best_stats[key]}')

    @torch.no_grad()
    def test(self, epoch, tb_logger, logger, record_per_item_stats=False):
        if self.gpu_id == 0:
            logger.info('Start testing')
        self.model.eval()
        info = {}
        per_action_info = {}
        pred = {
            'video-id' : [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }

        gt = {
            'video-id' : [],
            't-start' : [],
            't-end': [],
            'label': []
        }

        for iter_num, x in enumerate(tqdm(self.test_loader, leave=False, disable=(self.gpu_id != 0))):
            x = self.convert_data_to_device(x)

            loss_dict, info_dict = self.model(x)

            for i in range(len(x['group_segment'])):
                valid_idx = x['group_label'][i] != -1
                gt['video-id'].extend(torch.tensor([x['video_id'][i]] * sum(valid_idx)).to(self.gpu_id))
                gt['t-start'].extend(x['group_segment'][i][valid_idx, 0])
                gt['t-end'].extend(x['group_segment'][i][valid_idx, 1])
                gt['label'].extend(x['group_label'][i, valid_idx])

            for i in range(len(info_dict['group_cls'])):
                valid_idx = info_dict['group_cls_confidence'][i] > self.valid_action_threshold
                pred['video-id'].extend(torch.tensor([x['video_id'][i]] * sum(valid_idx)).to(self.gpu_id))
                pred['t-start'].extend(info_dict['group_segment'][i][valid_idx, 0])
                pred['t-end'].extend(info_dict['group_segment'][i][valid_idx, 1])
                pred['label'].extend(info_dict['group_cls'][i, valid_idx])
                pred['score'].extend(info_dict['group_cls_confidence'][i, valid_idx])

        # aggregate results
        dist.barrier()
        max_len = self.config.dataset.max_action_num * len(self.test_loader.dataset)
        gt = gather_dict(gt, self.gpu_id, max_len)
        convert_to_list(gt)
        pred = gather_dict(pred, self.gpu_id, max_len)
        convert_to_list(pred)

        # post process
        if self.gpu_id == 0:
            calculator = LocEval(gt,
                                tiou_thresholds=self.tiou_thresholds)
            mAP, average_mAP, mRecall, info_block = calculator.evaluate(pred)
            info = {
                'group_average_mAP': average_mAP * 100,
                f'group_mAP@{self.tiou_thresholds[-1]}': mAP[-1] * 100,
            }

            if record_per_item_stats:
                # calculator.ap stores N_thres * N_class matrix
                from vis import draw_mAP_curve
                draw_mAP_curve(mAP * 100, self.tiou_thresholds, self.save_dir)

            logger.info(info_block)

        dist.barrier()

        if self.gpu_id == 0:
            tb_vis(tb_logger,
                    epoch,
                    total_loss=None,
                    loss_dict={},
                    info_dict=info,
                    scheduler=None)
        return info, per_action_info