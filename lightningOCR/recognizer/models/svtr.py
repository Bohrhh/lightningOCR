import torch
import torch.nn as nn
import torch.nn.functional as F

from lightningOCR.common import ARCHITECTURES
from .modules import MobileNetV1Enhance, SequenceEncoder, CTCHead


@ARCHITECTURES.register()
class SVTR(nn.Module):
    """CTC-loss based recognizer.
    
    Shape:
        x (Tensor): shape (N, C, H, W)
    
    Return:
        results (Dict): {'logits': Tensor of shape (N, T, D) e.g. (64, 80, 6625)
                         'feats': Tensor of shape (N, T, C) e.g. (64, 80, 96)} 
    """
    def __init__(self, return_feats=False):
        super(SVTR, self).__init__()
        self.backbone = MobileNetV1Enhance(scale=0.5, last_conv_stride=[1,2], last_pool_type='avg')
        self.neck = SequenceEncoder(self.backbone.out_channels, encoder_type='svtr',
            dims=64, depth=2, hidden_dims=120, use_guide=True)
        self.head = CTCHead(self.neck.out_channels, fc_decay=0.00001, return_feats=return_feats)
        self.return_feats = return_feats

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)

        if self.return_feats:
            feats, logits = self.head(x)
            return {'logits': logits, 'feats': feats}
        else:
            logits = self.head(x)
            return {'logits': logits}
