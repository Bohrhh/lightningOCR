from .config import Config
from .registry import Registry, build_from_cfg
from .datawork import Compose, BaseDataset
from .pipelines import PIPELINES
from .utils import LitProgressBar
from .activation import Activation
from .litmodel import DATASETS, BaseLitModule, ARCHITECTURES, POSTPROCESS

__all__ = ['PIPELINES', 'DATASETS'
           'Config', 'build_lightning_model']

LIGHTNING_MODULE = Registry('lightmodule')


def build_lightning_model(cfg):
    return build_from_cfg(cfg, LIGHTNING_MODULE)
