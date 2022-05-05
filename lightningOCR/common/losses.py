import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .registry import Registry, build_from_cfg

LOSSES = Registry('loss')


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


@LOSSES.register()
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fun = nn.CrossEntropyLoss()
    
    def forward(self, pred, gt):
        logits = pred['logits']
        targets = gt['target']
        loss = self.loss_fun(logits, targets)
        return {'loss': loss}


@LOSSES.register()
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1, loss_weight=1.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.loss_weight = loss_weight

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.mean()*self.loss_weight
        return {'loss': loss}


@LOSSES.register()
class CTCLoss(nn.Module):
    def __init__(self, use_focal_loss=False, alpha=1, gamma=2):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        logits = pred['logits']
        targets = gt['target']
        target_lengths = gt['target_length']

        N, T, _ = logits.shape
        x = torch.log_softmax(logits, dim=2)
        x_for_loss = x.permute(1, 0, 2).contiguous()  # (T, N, D)
        x_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long, device=logits.device)
        target_lengths = torch.clamp(target_lengths, min=1, max=T).long()

        loss = self.loss_func(x_for_loss, targets, x_lengths, target_lengths) # (N, )

        if self.use_focal_loss:
            weight = (1 - torch.exp(-loss)) ** self.gamma
            loss = loss * weight * self.alpha
        loss = torch.mean(loss)

        return {'loss': loss}


@LOSSES.register()
class CenterLoss(nn.Module):
    """
    Reference: Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    """

    def __init__(self, num_classes=6625, feat_dim=96, center_file_path=None):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.register_buffer('centers', torch.randn(self.num_classes, self.feat_dim, dtype=torch.float64))

        assert center_file_path is not None and os.path.exists(center_file_path)
        char_dict = torch.load(center_file_path, map_location='cpu')
        for k, v in char_dict.items():
            self.centers[k] = v[0]

    def forward(self, pred, gt):
        logits = pred['logits'] # (N, T, D)
        feats = pred['feats'] # (N, T, C)
        num_classes = logits.shape[2]

        feats = feats.reshape(-1, feats.shape[-1]).to(torch.float64) # (N * T, C)
        label = torch.argmax(logits, dim=2)
        label = label.flatten() # (N * T, )

        batch_size = feats.shape[0]

        #calc l2 distance between feats and centers
        square_feats = torch.square(feats).sum(dim=1, keepdim=True) # (N * T, 1)
        square_feats = square_feats.expand(batch_size, num_classes) # (N * T, D)

        square_center = torch.square(self.centers).sum(dim=1, keepdim=True) # (D, 1)
        square_center = square_center.expand(num_classes, batch_size).to(torch.float64) # (D, N * T)
        square_center = square_center.transpose(1, 0) # (N * T, D)

        distmat = square_feats + square_center
        feat_dot_center = torch.matmul(feats, torch.transpose(self.centers, 1, 0)) # (N * T, D)
        distmat = distmat - 2.0 * feat_dot_center # (N * T, D)

        #generate the mask
        classes = torch.arange(num_classes, dtype=torch.long, device=self.centers.device)
        label = torch.unsqueeze(label, 1).expand(batch_size, num_classes) # (N * T, D)
        mask = (classes.expand(batch_size, num_classes) == label).to(torch.float64)
        dist = torch.multiply(distmat, mask)

        loss = torch.sum(torch.clip(dist, min=1e-12, max=1e+12)) / batch_size
        return {'loss': loss}


@LOSSES.register()
class ACELoss(nn.Module):
    def __init__(self):
        super(ACELoss, self).__init__()

    def forward(self, pred, gt):
        logits = pred['logits'] # (N, T, D)
        target_length = gt['target_length'].to(torch.float32) # (N, )
        target_ace = gt['target_ace'].to(torch.float32) # (N, D)

        N, T = logits.shape[:2]

        probs = nn.functional.softmax(logits, dim=-1)
        aggregation_preds = torch.mean(probs, dim=1) # (N, D)

        target_ace[:, 0] = T - target_length
        target_ace = target_ace / T # (N, D)

        log_preds = torch.log(aggregation_preds + 1e-7)
        loss = - torch.sum(target_ace * log_preds) / N
        return {'loss': loss}


@LOSSES.register()
class CombinedLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CombinedLoss, self).__init__()
        self.loss_weight = []
        self.loss_func = nn.ModuleList()
        self.loss_name = []
        for k, loss_cfg in kwargs.items():
            weight = 1.0 if 'weight' not in loss_cfg else loss_cfg.pop('weight')
            self.loss_weight.append(weight)
            self.loss_func.append(build_loss(loss_cfg))
            self.loss_name.append(k)

    def forward(self, pred, gt):
        loss_total = 0
        loss = {}
        for name, func, weight in zip(self.loss_name, self.loss_func, self.loss_weight):
            loss[name] = func(pred, gt)['loss'] * weight
            loss_total += loss[name]

        loss['loss'] = loss_total
        return loss
