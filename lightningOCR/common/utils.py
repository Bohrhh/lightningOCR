import os
import re
import signal
import platform
import contextlib
import matplotlib.pyplot as plt
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
    cfg['data']['batch_size_per_device'] = opt.batch_size
    cfg['data']['workers_per_device'] = min([opt.workers, os.cpu_count() // max(opt.devices, 1), opt.batch_size])

    # update strategy
    cfg['strategy']['devices'] = opt.devices
    cfg['strategy']['accelerator'] = opt.accelerator
    cfg['strategy']['epochs'] = opt.epochs
    cfg['strategy']['lr0'] = opt.lr
    cfg['strategy']['optim'] = opt.optim

    # update lightning model
    cfg['model']['strategy'] = cfg['strategy']
    cfg['model']['data'] = cfg['data']
    return cfg


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


class Timeout(contextlib.ContextDecorator):
    # Usage: @Timeout(seconds) decorator or 'with Timeout(seconds):' context manager
    def __init__(self, seconds, *, timeout_msg='', suppress_timeout_errors=True):
        self.seconds = int(seconds)
        self.timeout_message = timeout_msg
        self.suppress = bool(suppress_timeout_errors)

    def _timeout_handler(self, signum, frame):
        raise TimeoutError(self.timeout_message)

    def __enter__(self):
        if platform.system() != 'Windows':  # not supported on Windows
            signal.signal(signal.SIGALRM, self._timeout_handler)  # Set handler for SIGALRM
            signal.alarm(self.seconds)  # start countdown for SIGALRM to be raised

    def __exit__(self, exc_type, exc_val, exc_tb):
        if platform.system() != 'Windows':
            signal.alarm(0)  # Cancel SIGALRM if it's scheduled
            if self.suppress and exc_type is TimeoutError:  # Suppress TimeoutError
                return True


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return True if re.search('[\u4e00-\u9fff]', str(s)) else False


def has_ctcloss(loss_cfg):
    if loss_cfg.type.lower() == 'ctcloss':
        return True
    if loss_cfg.type.lower() == 'combinedloss':
        loss_cfg.pop('type')
        for k,v in loss_cfg.items():
            if v.type.lower() == 'ctcloss':
                return True
    return False


@try_except
@Timeout(30)
def plot_reclabels(trainset, valset=None, out_file='./label.jpg'):
    train_labels = trainset.get_labels()
    val_labels = valset.get_labels()
    train_labels = list(train_labels.items())
    train_labels = sorted(train_labels, key=lambda x: x[1], reverse=True)
    y1 = [x[1] if x[1] > 0 else 1e-7 for x in train_labels]
    max_y1 = max(y1)
    
    plt.figure(figsize=(10,4))
    ax = plt.plot(y1)
    ax[0].set_label('train labels')

    if val_labels is not None:
        y2 = [val_labels[x[0]] if val_labels[x[0]] > 0 else 1e-7 for x in train_labels]
        val_char_num = sum(y2)
        num = 0
        for i,v in enumerate(y2):
            num += v
            if num / val_char_num > 0.99:
                ax = plt.plot([i,i], [0,max_y1], '--', color='red')
                ax[0].set_label('0.99 val')
                break
        ax = plt.plot(y2)
        ax[0].set_label('val labels')

    plt.xlabel('chars', fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.ylim(1, max_y1)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches='tight')


# def plot_haha(trainset, valset=None, wrong=None):
#     train_labels = trainset.get_labels()
#     val_labels = valset.get_labels()
#     import copy
#     wrong_labels = copy.deepcopy(val_labels)
#     for k in wrong_labels:
#         wrong_labels[k] = 0
#     with open(wrong, 'r') as f:
#         f = [i.strip('\n') for i in f]
#     for i in f:
#         if i in wrong_labels:
#             wrong_labels[i] += 1

#     train_labels = list(train_labels.items())
#     train_labels = sorted(train_labels, key=lambda x: x[1], reverse=True)
#     y1 = [x[1] if x[1] > 0 else 1e-7 for x in train_labels]
#     max_y1 = max(y1)
    
#     plt.figure(figsize=(10,4))
#     ax = plt.plot(y1)
#     ax[0].set_label('train labels')

#     if val_labels is not None:
#         y2 = [val_labels[x[0]] if val_labels[x[0]] > 0 else 1e-7 for x in train_labels]
#         val_char_num = sum(y2)
#         num = 0
#         for i,v in enumerate(y2):
#             num += v
#             if num / val_char_num > 0.99:
#                 ax = plt.plot([i,i], [0,max_y1], '--', color='red')
#                 ax[0].set_label('0.99 val')
#                 break
#         ax = plt.plot(y2)
#         ax[0].set_label('val labels')

#     if wrong_labels is not None:
#         y2 = [wrong_labels[x[0]] if wrong_labels[x[0]] > 0 else 1e-7 for x in train_labels]
#         val_char_num = sum(y2)
#         num = 0
#         for i,v in enumerate(y2):
#             num += v
#             if num / val_char_num > 0.99:
#                 ax = plt.plot([i,i], [0,max_y1], '--', color='black')
#                 ax[0].set_label('0.99 wrong')
#                 break
#         ax = plt.plot(y2)
#         ax[0].set_label('wrong labels')

#     plt.xlabel('chars', fontsize=15)
#     plt.ylabel('count', fontsize=15)
#     plt.ylim(1, max_y1)
#     plt.yscale('log')
#     plt.legend(fontsize=12)
#     plt.tight_layout()