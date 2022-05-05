import os
import torch
import torch.nn as nn

from lightningOCR.common import BaseLitModule
from lightningOCR.common import LIGHTNING_MODULE
from lightningOCR.common.metric import RecAcc, RecF1
from lightningOCR.common.utils import plot_reclabels


@LIGHTNING_MODULE.register()
class Recognizer(BaseLitModule):
    def __init__(
        self,
        data,
        strategy,
        architecture,
        loss,
        metric=None,
        postprocess=None,
    ):
        super(Recognizer, self).__init__(
            data, strategy, architecture, loss, metric, postprocess
        )
        assert isinstance(self.metric, (RecAcc, RecF1))
        assert self.postprocess is not None
        self.register_buffer('train_corrects', torch.tensor(0.0))
        self.register_buffer('train_gt_samples', torch.tensor(0.0))
        self.register_buffer('train_pred_samples', torch.tensor(0.0))
        self.register_buffer('val_corrects', torch.tensor(0.0))
        self.register_buffer('val_gt_samples', torch.tensor(0.0))
        self.register_buffer('val_pred_samples', torch.tensor(0.0))

        self.center = {}

    def forward(self, x):
        """
        Args:
            x (Tensor): normalized img, shape (n, c, h, w)
        Return:
            results (Dict): {'text':[t1, t2, ..., tn], 'prob':[p1, p2, ..., pn]} 
        """
        x = self.model(x)
        results, _ = self.postprocess(x)
        return results

    def log_f1(self, match_chars, gt_chars, pred_chars, mode, prog_bar, logger):
        precision = match_chars / (pred_chars + 1e-8)
        recall = match_chars / (gt_chars + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        self.log(f'precision/{mode}', precision, prog_bar=prog_bar, logger=logger)
        self.log(f'recall/{mode}', recall, prog_bar=prog_bar, logger=logger)
        self.log(f'f1/{mode}', f1, prog_bar=prog_bar, logger=logger)

    def on_train_start(self):
        # Plot distribution of trainset and valset
        if not hasattr(self.trainset, 'lmdb_sets'):
            self.trainset.open_lmdb()
        if not hasattr(self.valset, 'lmdb_sets'):
            self.valset.open_lmdb()
        plot_reclabels(self.trainset, self.valset, os.path.join(self.logger.log_dir, f'labels.jpg'))

    def training_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)
        pred, gt = self.postprocess(pred, gt)

        # Log loss
        for k, v in loss.items():
            self.log(f'{k}/train', v.item(), prog_bar=False, logger=True, on_epoch=True, on_step=False, batch_size=len(x))

        # Log metric
        if isinstance(self.metric, RecAcc):
            c, s, _ = self.metric(pred, gt)
            self.train_corrects += c
            self.train_gt_samples += s
            self.log('acc/train', self.train_corrects / self.train_gt_samples, prog_bar=True, logger=False)
        else:
            match_chars, gt_chars, pred_chars, _ = self.metric(pred, gt)
            self.train_corrects += match_chars
            self.train_gt_samples += gt_chars
            self.train_pred_samples += pred_chars
            self.log_f1(self.train_corrects, self.train_gt_samples, self.train_pred_samples, 'train', True, False)

        # Log lr
        for j, para in enumerate(self.optimizers().param_groups):
            self.log(f'x/lr{j}', para['lr'], prog_bar=False, logger=True)

        # Plot
        if self.global_rank in [-1, 0] and self.global_step < 6 and hasattr(self.trainset, 'plot_batch'):
            # do plot
            os.makedirs(self.logger.log_dir, exist_ok=True)
            self.trainset.plot_batch(batch, os.path.join(self.logger.log_dir, f'train_batch_{self.global_step}.jpg'))

        return loss.pop('loss')

    def training_epoch_end(self, training_step_outputs):
        train_corrects = self.all_gather(self.train_corrects)
        train_gt_samples = self.all_gather(self.train_gt_samples)
        train_pred_samples = self.all_gather(self.train_pred_samples)
        if isinstance(self.metric, RecAcc):
            self.log('acc/train', train_corrects.sum() / train_gt_samples.sum(), prog_bar=False, logger=True)
        else:
            self.log_f1(train_corrects.sum(), train_gt_samples.sum(), train_pred_samples.sum(), 'train', False, True)
        self.train_corrects.zero_()
        self.train_gt_samples.zero_()
        self.train_pred_samples.zero_()

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        gt = batch['gt']
        pred = self.model(x)
        loss = self.loss(pred, gt)
        pred_result, gt_result = self.postprocess(pred, gt)

        # Log loss
        for k, v in loss.items():
            self.log(f'{k}/val', v.item(), prog_bar=False, logger=True, on_epoch=True, on_step=False, batch_size=len(x))

        # Update for metric
        if isinstance(self.metric, RecAcc):
            c, s, wrong_index = self.metric(pred_result, gt_result)
            self.val_corrects += c
            self.val_gt_samples += s
        else:
            match_chars, gt_chars, pred_chars, wrong_index = self.metric(pred_result, gt_result)
            self.val_corrects += match_chars
            self.val_gt_samples += gt_chars
            self.val_pred_samples += pred_chars

        # Plot
        if self.global_rank in [-1, 0] and self.current_epoch == 0 and \
           batch_idx < 6 and hasattr(self.valset, 'plot_batch'):
            # do plot
            os.makedirs(self.logger.log_dir, exist_ok=True)
            self.valset.plot_batch(batch, os.path.join(self.logger.log_dir, f'val_batch_{batch_idx}.jpg'))

        # Save wrong predicts
        if self.stage == 'validate' and self.save_fault:
            falut_dir = os.path.join(self.logger.log_dir, 'fault')
            os.makedirs(falut_dir, exist_ok=True)
            for index in wrong_index:
                title = f'pred:{pred_result["text"][index]}\ngt:{gt_result["text"][index]}'
                save_img = os.path.join(falut_dir, f'{batch_idx}.{self.global_rank}.{index}.jpg')
                self.valset.plot(batch['image'][index], title, save_img)

        # Update center
        if self.stage == 'validate' and self.save_center:
            feats = pred['feats'] # (N, T, C)
            logits = pred['logits'] # (N, T, D)
            indexs = torch.argmax(logits, dim=2) # (N, T)
            indexs = indexs.cpu().numpy()
            N, T = indexs.shape
            for i in range(N):
                if i not in wrong_index:
                    feat = feats[i]
                    index = indexs[i]
                    for j in range(T):
                        if index[j] in self.center:
                            self.center[index[j]][0] = \
                                (self.center[index[j]][0] * self.center[index[j]][1] + feat[j]) / (self.center[index[j]][1] + 1)
                            self.center[index[j]][1] += 1
                        else:
                            self.center[index[j]] = [feat[j], 1]

    def validation_epoch_end(self, val_step_outputs):
        val_corrects = self.all_gather(self.val_corrects)
        val_gt_samples = self.all_gather(self.val_gt_samples)
        val_pred_samples = self.all_gather(self.val_pred_samples)

        if isinstance(self.metric, RecAcc):
            self.log('acc/val', val_corrects.sum() / val_gt_samples.sum(), prog_bar=True, logger=True, rank_zero_only=True)
        else:
            self.log_f1(val_corrects.sum(), val_gt_samples.sum(), val_pred_samples.sum(), 'val', True, True)

        self.val_corrects.zero_()
        self.val_gt_samples.zero_()
        self.val_pred_samples.zero_()

        if self.stage == 'validate' and self.save_center:
            torch.save(self.center, 'center.pth')