from .config import Config
from .registry import Registry, build_from_cfg
from .datawork import BaseDataModule, DATASETS, Compose
from .litmodel import BaseLitModule
from .pipelines import PIPELINES
from .losses import LOSSES
from .utils import LitProgressBar

__all__ = ['PIPELINES', 'DATASETS', 'LIGHTNING_DATAS', 'BACKBONES', 'ARCHITECTURES',
           'Config', 'build_lightning_data', 'build_backbone', 'build_lightning_model']

LIGHTNING_DATAS = Registry('lightdata')
LIGHTNING_MODULE = Registry('lightmodule')
ARCHITECTURES = Registry('architecture')


def build_lightning_data(cfg):
    return BaseDataModule(cfg)


def build_model(cfg):
    return build_from_cfg(cfg, ARCHITECTURES)


def build_lightning_model(cfg):
    return build_from_cfg(cfg, LIGHTNING_MODULE)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)