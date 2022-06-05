import torch
import torch.nn as nn
import torch.nn.functional as F

from lightningOCR.common import ARCHITECTURES
from .modules import MobileNetV3, RSEFPN, DBHead


@ARCHITECTURES.register()
class DBNet(nn.Module):
    """CTC-loss based recognizer.
    
    Shape:
        x (Tensor): shape (N, C, H, W)
    
    Return:
        results (Dict): {'logits': Tensor of shape (N, T, D) e.g. (64, 80, 6625)
                         'feats': Tensor of shape (N, T, C) e.g. (64, 80, 96)} 
    """
    def __init__(self):
        super(DBNet, self).__init__()
        self.backbone = MobileNetV3(scale=0.5, model_name='large', disable_se=True)
        self.neck = RSEFPN(self.backbone.out_channels, out_channels=96, shortcut=True)
        self.head = DBHead(self.neck.out_channels, k=50)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        shrink_maps, threshold_maps, binary_maps = self.head(x)
        return {'shrink_maps': shrink_maps, 'threshold_maps':threshold_maps, 'binary_maps': binary_maps}