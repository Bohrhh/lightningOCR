import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE, build_model, build_loss
from .resnet import *


def get_acc(logits, labels):
    """
    Args:
        logits (Tensor): shape (N,classes)
        labels (Tensor): shape (N,)

    Returns:
        corrects : correct number
        batch_num : batch number
    """
    _, pred_labels = torch.max(logits, dim=1)
    if len(labels.shape) > 1:
        _, labels = labels.max(dim=1)
    corrects = (pred_labels == labels).float().sum().item()
    batch_num = len(pred_labels)
    return corrects, batch_num


@LIGHTNING_MODULE.register()
class Classifier(BaseLitModule):
    def __init__(self, data_cfg, strategy, architecture, loss_cfg):
        super(Classifier, self).__init__(data_cfg, strategy)
        self.model = build_model(architecture)
        self.loss = build_loss(loss_cfg)
        self.train_corrects = 0
        self.train_samples = 0
        self.val_corrects = 0
        self.val_samples = 0

    def training_step(self, batch, batch_idx):
        x = batch['image']
        label = batch['label']
        logits = self.model(x)
        loss = self.loss(logits, label)

        # Log and Plot
        c, s = get_acc(logits, label)
        self.train_corrects += c
        self.train_samples += s
        self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=True, logger=False)
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)
        if self.global_rank in [-1, 0] and self.global_step < 3:
            # do plot
            pass

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log('acc/train', self.train_corrects / self.train_samples, prog_bar=False, logger=True)
        self.train_corrects = 0
        self.train_samples = 0

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        label = batch['label']
        logits = self.model(x)
        c, s = get_acc(logits, label)
        return c, s

    def validation_step_end(self, val_step_outputs):
        if self.strategy['gpus'] <= 1:
            val_step_outputs = [val_step_outputs]
        for (c, s) in val_step_outputs:
            self.val_corrects += c
            self.val_samples += s

    def validation_epoch_end(self, val_step_outputs):
        self.log('acc/val', self.val_corrects / self.val_samples, prog_bar=True, logger=True)