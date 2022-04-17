import os
import math
import torch
import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from pytorch_lightning import LightningModule

from .losses import LOSSES
from .metric import METRICS
from .utils import colorstr
from .registry import Registry, build_from_cfg


DATASETS = Registry('dataset')
ARCHITECTURES = Registry('architecture')
POSTPROCESS = Registry('postprocess')


def build_dataset(cfg, default_args=None):
    pipeline = build_from_cfg(cfg, DATASETS, default_args)
    return pipeline


def build_model(cfg):
    return build_from_cfg(cfg, ARCHITECTURES)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


def build_metric(cfg):
    return build_from_cfg(cfg, METRICS)


def build_postprocess(cfg):
    return build_from_cfg(cfg, POSTPROCESS)


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        data,
        strategy,
        architecture,
        loss,
        metric=None,
        postprocess=None
    ):
        super(BaseLitModule, self).__init__()
        self.data = data
        self.strategy = strategy
        self.batch_size = data.batch_size_per_device
        self.num_workers = min(data.workers_per_device,
                               os.cpu_count() // max(torch.cuda.device_count(),1),
                               self.batch_size if self.batch_size > 1 else 0)
        self.pin_memory = data.pin_memory

        self.model = build_model(architecture)
        self.loss = build_loss(loss)
        self.metric = None if metric is None else build_metric(metric)
        self.postprocess = None if postprocess is None else build_postprocess(postprocess)

    def setup(self, stage=None):
        self.stage = stage
        self.trainset_size = -1
        self.valset_size = -1
        if stage == 'fit' or stage is None:
            self.trainset = build_dataset(self.data.train)
            self.trainset_size = len(self.trainset)
            self.valset = build_dataset(self.data.val)
            self.valset_size = len(self.valset)
        if stage == 'validate':
            self.valset = build_dataset(self.data.val)
            self.valset_size = len(self.valset)
        if stage == 'test' or stage is None:
            self.testset = build_dataset(self.data.test)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=2*self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=2*self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=False)

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        epochs = self.strategy['epochs']
        warmup_epochs = self.strategy['warmup_epochs']
        lr0 = self.strategy['lr0']
        lrf = self.strategy['lrf']
        momentum = self.strategy['momentum']
        weight_decay = self.strategy['weight_decay']
        cos_lr = self.strategy['cos_lr']
        optim = self.strategy['optim']

        max_steps = self.trainer.max_steps
        warmup_steps = max(1500, max_steps//epochs * warmup_epochs)
        
        
        # Optimizer
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        if optim == 'Adam':
            optimizer = Adam(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif optim == 'AdamW':
            optimizer = AdamW(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=lr0, momentum=momentum, nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        print(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
              f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
        del g0, g1, g2

        lf = lambda x: lrfun_with_warmup(x, max_steps, warmup_steps, lrf, cos_lr)
        scheduler = {
            "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=lf),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def lrfun_with_warmup(step, max_steps, warmup_steps, lrf, cos_lr=True):

    if cos_lr:
        lf = lambda x: ((1 - math.cos(x * math.pi / max_steps)) / 2) * (lrf - 1) + 1
    else:
        lf = lambda x: (1 - x / max_steps) * (1.0 - lrf) + lrf

    if step < warmup_steps:
        return np.interp(step, [0, warmup_steps], [0, lf(warmup_steps)])
    else:
        return lf(step)