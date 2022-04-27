import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from lightningOCR import classifier
from lightningOCR import recognizer
from lightningOCR.common import Config
from lightningOCR.common import build_lightning_model
from lightningOCR.common.utils import intersect_dicts, update_cfg, colorstr
from lightningOCR.common.utils import LitProgressBar


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',           type=str,   choices=['train', 'val', 'test'], default='train',
                        help='program mode')
    parser.add_argument('--cfg',            type=str,   default='configs/crnn.py',
                        help='train config file path')
    parser.add_argument('--weights',        type=str,   default='',
                        help='initial weights path')
    parser.add_argument('--accelerator',    type=str,   default='gpu'),
    parser.add_argument('--devices',        type=int,   default=1),
    parser.add_argument('--epochs',         type=int,   default=50),
    parser.add_argument('--lr',             type=float, default=0.01,
                        help='max learning rate during training'),
    parser.add_argument('--batch-size',     type=int,   default=64,
                        help='batch size per device (gpu)')
    parser.add_argument('--workers',        type=int,   default=8,
                        help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--optim',          type=str,   choices=['SGD', 'Adam', 'AdamW'], default='SGD',
                        help='optimizer')
    parser.add_argument('--seed',           type=int,   default=77,
                        help='random seed')
    parser.add_argument('--project',        type=str,   default=None,
                        help='save to project/name')
    parser.add_argument('--name',           type=str,   default='rec',
                        help='save to project/name')
    parser.add_argument('--fp16',           action='store_true',
                        help='use fp16')
    parser.add_argument('--sync-bn',        action='store_true',
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--resume',         type=str,   default=None,
                        help='resume model from file')
    parser.add_argument('--val-period',     type=int,   default=1,
                        help='resume model from file')
    parser.add_argument('--noval',          action='store_true',
                        help='only validate final epoch')
    parser.add_argument('--nosave',         action='store_true',
                        help='do not save checkpoint')
    parser.add_argument('--save-fault',     action='store_true',
                        help='save wrong prediction')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main():

    # =============================================
    # parse args
    opt = parse_opt()
    cfg = Config.fromfile(opt.cfg)
    cfg = update_cfg(cfg, opt)

    if opt.seed is not None:
        pl.seed_everything(opt.seed, workers=True)
    devices = cfg['strategy']['devices']
    epochs = cfg['strategy']['epochs']
    batch_size_per_device = cfg['data']['batch_size_per_device']
    batch_size_total = batch_size_per_device * devices

    project = f'../runs/{opt.mode}' if opt.project is None else opt.project
    tb_logger = loggers.TensorBoardLogger(save_dir=project, name=opt.name)
    print(f'Results saved to {colorstr("bold", tb_logger.log_dir)}')

    # =============================================
    # build lightning model
    lmodel = build_lightning_model(cfg.model)
    lmodel.prepare_data()
    lmodel.setup(stage={'train':'fit', 'val':'validate', 'test':'test'}[opt.mode])
    lmodel.save_fault = opt.save_fault

    if opt.weights:
        assert os.path.exists(opt.weights), f'weights does not exist: {opt.weights}'
        csd = torch.load(opt.weights, map_location='cpu')['state_dict']
        csd = intersect_dicts(csd, lmodel.state_dict())  # intersect
        lmodel.load_state_dict(csd, strict=False)
        print(f'Transferred {len(csd)}/{len(lmodel.state_dict())} items from {opt.weights}')  # report

    # =============================================
    # callbacks
    bar = LitProgressBar()
    checkpoint = ModelCheckpoint(**cfg.ckpt, save_last=not opt.nosave)

    # =============================================
    # build trainer
    trainer = Trainer(
        accelerator=opt.accelerator,
        devices=devices,
        max_epochs=epochs,
        max_steps=(lmodel.trainset_size + batch_size_total - 1) // batch_size_total * epochs,
        logger=tb_logger,
        callbacks=[bar, checkpoint],
        sync_batchnorm=opt.sync_bn and devices > 1,
        precision=16 if opt.fp16 else 32,
        deterministic=False if opt.seed is None or cfg.model.loss.type=='CTCLoss' else True,
        benchmark=True if opt.seed is None else False,
        strategy='ddp' if devices > 1 else None,
        check_val_every_n_epoch= epochs+1 if opt.noval else opt.val_period,
    )

    if opt.mode.lower() == 'train':
        trainer.fit(lmodel)
    elif opt.mode.lower() == 'val':
        trainer.validate(lmodel)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()