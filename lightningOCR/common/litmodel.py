import math
import torch
import numpy as np
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from pytorch_lightning import LightningModule

from .utils import colorstr


class BaseLitModule(LightningModule, metaclass=ABCMeta):
    def __init__(self, strategy):
        super(BaseLitModule, self).__init__()
        self.strategy = strategy

    def forward(self, x):
        return self.model(x)

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        
        gpus = self.strategy['gpus']
        epochs = self.strategy['epochs']
        warmup_epochs = self.strategy['warmup_epochs']
        lr0 = self.strategy['lr0']
        lrf = self.strategy['lrf']
        momentum = self.strategy['momentum']
        weight_decay = self.strategy['weight_decay']
        cos_lr = self.strategy['cos_lr']

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

        if self.strategy == 'Adam':
            optimizer = Adam(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        elif self.strategy == 'AdamW':
            optimizer = AdamW(g0, lr=lr0, betas=(momentum, 0.999))  # adjust beta1 to momentum
        else:
            optimizer = SGD(g0, lr=lr0, momentum=momentum, nesterov=True)

        optimizer.add_param_group({'params': g1, 'weight_decay': weight_decay})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
        print(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
              f"{len(g0)} weight (no decay), {len(g1)} weight, {len(g2)} bias")
        del g0, g1, g2

        lf = lambda x: lrfun_with_warmup(x, max_steps, warmup_steps, lr0, lrf, cos_lr)
        scheduler = {
            "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=lf),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def lrfun_with_warmup(step, max_steps, warmup_steps, lr0, lrf, cos_lr=True):

    if cos_lr:
        lf = lambda x: ((1 - math.cos(x * math.pi / max_steps)) / 2) * (lrf - 1) + 1
    else:
        lf = lambda x: (1 - x / max_steps) * (1.0 - lrf) + lrf

    if step < warmup_steps:
        return np.interp(step, [0, warmup_steps], [0, lr0 * lf(warmup_steps)])
    else:
        return lr0 * lf(step)