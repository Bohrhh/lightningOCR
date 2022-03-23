import os
from pytorch_lightning.callbacks import TQDMProgressBar


class LitProgressBar(TQDMProgressBar):

    def __init__(self):
        super().__init__()

    def get_metrics(self, *args, **kwargs):
        # don't show the version number and loss_step
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def update_cfg(cfg, opt):
    # update data
    cfg['data']['batch_size_per_gpu'] = opt.batch_size // max(opt.gpus, 1)
    cfg['data']['workers_per_gpu'] = min([opt.workers, os.cpu_count() // max(opt.gpus, 1), opt.batch_size // max(opt.gpus, 1)])

    # update strategy
    cfg['strategy']['gpus'] = opt.gpus
    cfg['strategy']['epochs'] = opt.epochs
    cfg['strategy']['optim'] = opt.optim

    # update lightning model
    cfg['model']['strategy'] = cfg['strategy']
    cfg['model']['data_cfg'] = cfg['data']
    return cfg
