import os
import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE
from lightningOCR.common.metric import RecAcc, RecF1


@LIGHTNING_MODULE.register()
class Recognizer(BaseLitModule):
    def __init__(
        self,
        data_cfg,
        strategy,
        architecture,
        loss_cfg,
        metric_cfg=None,
        postprocess_cfg=None
    ):
        super(Recognizer, self).__init__(
            data_cfg, strategy, architecture, loss_cfg, metric_cfg, postprocess_cfg
        )
        assert isinstance(self.metric, (RecAcc, RecF1))
        assert self.postprocess is not None
        self.register_buffer('train_corrects', torch.tensor(0.0))
        self.register_buffer('train_gt_samples', torch.tensor(0.0))
        self.register_buffer('train_pred_samples', torch.tensor(0.0))
        self.register_buffer('val_corrects', torch.tensor(0.0))
        self.register_buffer('val_gt_samples', torch.tensor(0.0))
        self.register_buffer('val_pred_samples', torch.tensor(0.0))

    def forward(self, x):
        """
        Args:
            x (Tensor): normalized img, shape (n, c, h, w)
        Return:
            results (Dict): {'text':[t1, t2, ..., tn], 'prob':[p1, p2, ..., pn]} 
        """
        x = self.model(x)
        results = self.postprocess(x)
        return results

    def training_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)
        pred, gt = self.postprocess(pred, gt)

        # Metric
        if isinstance(self.metric, RecAcc):
            c, s = self.metric(pred, gt)
            self.train_corrects += c
            self.train_gt_samples += s
            self.log('acc/train', self.train_corrects / self.train_gt_samples, prog_bar=True, logger=False)
        else:
            match_chars, gt_chars, pred_chars = self.metric(pred, gt)
            self.train_corrects += match_chars
            self.train_gt_samples += gt_chars
            self.train_pred_samples += pred_chars
            precision = self.train_corrects / self.train_pred_samples
            recall = self.train_corrects / self.train_gt_samples
            f1 = 2 * precision * recall / (precision + recall)
            self.log('precision/train', precision, prog_bar=True, logger=False)
            self.log('recall/train', recall, prog_bar=True, logger=False)
            self.log('f1/train', f1, prog_bar=True, logger=False)
        # Log and Plot
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)
        if self.global_rank in [-1, 0] and self.global_step < 6 and hasattr(self.trainset, 'plot_batch'):
            # do plot
            self.trainset.plot_batch(batch, os.path.join(self.logger.log_dir, f'train_batch_{self.global_step}.jpg'))

        return loss

    def training_epoch_end(self, training_step_outputs):
        train_corrects = self.all_gather(self.train_corrects)
        train_gt_samples = self.all_gather(self.train_gt_samples)
        train_pred_samples = self.all_gather(self.train_pred_samples)
        if isinstance(self.metric, RecAcc):
            self.log('acc/train', train_corrects.sum() / train_gt_samples.sum(), prog_bar=False, logger=True)
        else:
            precision = train_corrects.sum() / train_pred_samples.sum()
            recall = train_corrects.sum() / train_gt_samples.sum()
            f1 = 2 * precision * recall / (precision + recall)
            self.log('precision/train', precision, prog_bar=False, logger=True)
            self.log('recall/train', recall, prog_bar=False, logger=True)
            self.log('f1/train', f1, prog_bar=False, logger=True)
        self.train_corrects.zero_()
        self.train_gt_samples.zero_()
        self.train_pred_samples.zero_()

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        pred, gt = self.postprocess(pred, gt)
        if isinstance(self.metric, RecAcc):
            c, s = self.metric(pred, gt)
            self.val_corrects += c
            self.val_gt_samples += s
        else:
            match_chars, gt_chars, pred_chars = self.metric(pred, gt)
            self.val_corrects += match_chars
            self.val_gt_samples += gt_chars
            self.val_pred_samples += pred_chars

    def validation_epoch_end(self, val_step_outputs):
        val_corrects = self.all_gather(self.val_corrects)
        val_gt_samples = self.all_gather(self.val_gt_samples)
        val_pred_samples = self.all_gather(self.val_pred_samples)

        if isinstance(self.metric, RecAcc):
            self.log('acc/val', val_corrects.sum() / val_gt_samples.sum(), prog_bar=True, logger=True, rank_zero_only=True)
        else:
            precision = val_corrects.sum() / val_pred_samples.sum()
            recall = val_corrects.sum() / val_gt_samples.sum()
            f1 = 2 * precision * recall / (precision + recall)
            self.log('precision/train', precision, prog_bar=True, logger=True, rank_zero_only=True)
            self.log('recall/train', recall, prog_bar=True, logger=True, rank_zero_only=True)
            self.log('f1/train', f1, prog_bar=True, logger=True, rank_zero_only=True)
        self.val_corrects.zero_()
        self.val_gt_samples.zero_()
        self.val_pred_samples.zero_()