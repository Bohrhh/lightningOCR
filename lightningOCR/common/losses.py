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
    def __init__(self):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, pred, gt):
        logits = pred['logits']
        targets = gt['target']
        target_lengths = gt['target_length']

        N, T, _ = logits.shape
        x = torch.log_softmax(logits, dim=2)
        x_for_loss = x.permute(1, 0, 2).contiguous()  # T * N * C
        x_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long, device=logits.device)
        target_lengths = torch.clamp(target_lengths, min=1, max=T).long()

        loss = self.loss_func(x_for_loss, targets, x_lengths, target_lengths)
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

    def forward(self, pred, gt):
        logits = pred['logits']
        features = pred['feats']

        feats_reshape = torch.reshape(
            features, [-1, features.shape[-1]]).to(torch.float64)
        label = torch.argmax(logits, axis=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        #calc l2 distance between feats and centers  
        square_feat = torch.sum(torch.square(feats_reshape),
                                 axis=1,
                                 keepdim=True)
        square_feat = square_feat.expand(batch_size, self.num_classes)

        square_center = torch.sum(torch.square(self.centers),
                                   axis=1,
                                   keepdim=True)
        square_center = square_center.expand(self.num_classes, batch_size).to(torch.float64)
        square_center = torch.transpose(square_center, 1, 0)

        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape,
                                        torch.transpose(self.centers, 1, 0))
        distmat = distmat - 2.0 * feat_dot_center

        #generate the mask
        classes = torch.arange(self.num_classes, dtype=torch.long, device=self.centers.device)
        label = torch.unsqueeze(label, 1).expand(batch_size, self.num_classes)
        mask = (classes.expand(batch_size, self.num_classes) == label).to(torch.float64)
        dist = torch.multiply(distmat, mask)

        loss = torch.sum(torch.clip(dist, min=1e-12, max=1e+12)) / batch_size
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
