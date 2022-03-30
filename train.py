import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers

from lightningOCR import classifier
from lightningOCR import recognizer
from lightningOCR.common import Config, LitProgressBar
from lightningOCR.common import build_lightning_model
from lightningOCR.common.utils import intersect_dicts, update_cfg


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',            type=str,  default='configs/resnet.py',
                        help='train config file path')
    parser.add_argument('--weights',        type=str,  default='',
                        help='initial weights path')
    parser.add_argument('--gpus',           type=int,  default=1),
    parser.add_argument('--epochs',         type=int,  default=10),
    parser.add_argument('--batch-size',     type=int,  default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--workers',        type=int,  default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--optim',          type=str,  choices=['SGD', 'Adam', 'AdamW'], default='SGD',
                        help='optimizer')
    parser.add_argument('--seed',           type=int,  default=None,
                        help='random seed')
    parser.add_argument('--project',        type=str,  default='../runs/train',
                        help='save to project/name')
    parser.add_argument('--name',           type=str,  default='cls',
                        help='save to project/name')
    parser.add_argument('--fp16',           action='store_true',
                        help='use fp16')
    parser.add_argument('--sync-bn',        action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--resume',         type=str,  default=None,
                        help='resume model from file')
    parser.add_argument('--val-period',     type=int,  default=1,
                        help='resume model from file')
    opt = parser.parse_known_args()[0]
    return opt


def main():

    # =============================================
    # parse args
    opt = parse_opt()
    opt.cfg = 'configs/crnn.py'
    # opt.weights = 'pretrained/ch_ptocr_v2_rec_infer.pth'
    cfg = Config.fromfile(opt.cfg)
    cfg = update_cfg(cfg, opt)

    if opt.seed is not None:
        pl.seed_everything(opt.seed, workers=True)
    gpus = cfg['strategy']['gpus']
    epochs = cfg['strategy']['epochs']
    batch_size_per_gpu = cfg['data']['batch_size_per_gpu']
    batch_size = batch_size_per_gpu * gpus

    tb_logger = loggers.TensorBoardLogger(save_dir=opt.project, name=opt.name)


    # =============================================
    # build lightning model
    lmodel = build_lightning_model(cfg.model)
    lmodel.prepare_data()
    lmodel.setup(stage='fit')

    if opt.weights:
        assert os.path.exists(opt.weights)
        csd = torch.load(opt.weights, map_location='cpu')['state_dict']
        csd = intersect_dicts(csd, lmodel.model.state_dict())  # intersect
        lmodel.model.load_state_dict(csd, strict=False)
        print(f'Transferred {len(csd)}/{len(lmodel.state_dict())} items from {opt.weights}')  # report

    # =============================================
    # callbacks
    bar = LitProgressBar()

    # =============================================
    # build trainer
    trainer = Trainer(
        gpus=gpus,
        max_epochs=epochs,
        max_steps=(lmodel.trainset_size + batch_size - 1) // batch_size * epochs,
        logger=tb_logger,
        callbacks=[bar],
        sync_batchnorm=opt.sync_bn and gpus > 1,
        precision=16 if opt.fp16 else 32,
        deterministic=False if opt.seed is None or cfg.model.loss_cfg.type=='CTCLoss' else True,
        benchmark=True if opt.seed is None else False,
        strategy='ddp' if gpus > 1 else None,
        check_val_every_n_epoch=opt.val_period
    )
    trainer.fit(lmodel)


if __name__ == '__main__':
    main()