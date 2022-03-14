import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks, loggers

from lightningOCR.common import Config, LitProgressBar
from lightningOCR.common import build_lightning_data, build_lightning_model
from lightningOCR import classifier


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',            type=str,  default='configs/resnet.py',
                        help='train config file path')
    parser.add_argument('--seed',           type=int,  default=77,
                        help='random seed')
    parser.add_argument('--project',        type=str,  default='../runs/train',
                        help='save to project/name')
    parser.add_argument('--name',           type=str,  default='cls',
                        help='save to project/name')
    parser.add_argument('--resume',         type=str,  default=None,
                        help='resume model from file')
    opt = parser.parse_known_args()[0]
    return opt


def main():

    # =============================================
    # parse args
    opt = parse_opt()
    opt.cfg = 'configs/resnet.py'
    cfg = Config.fromfile(opt.cfg)
    # cfg = update_cfg(opt, cfg)

    pl.seed_everything(opt.seed, workers=True)
    gpus = cfg['strategy']['gpus']
    epochs = cfg['strategy']['epochs']
    batch_size_per_gpu = cfg['data']['batch_size_per_gpu']
    batch_size = batch_size_per_gpu * gpus

    tb_logger = loggers.TensorBoardLogger(save_dir=opt.project, name=opt.name)

    # =============================================
    # build lightning data
    ldata = build_lightning_data(cfg.data)
    ldata.prepare_data()
    ldata.setup(stage='fit')


    # =============================================
    # build lightning model
    lmodel = build_lightning_model(cfg.model)


    # =============================================
    # callbacks
    bar = LitProgressBar()

    # =============================================
    # build trainer
    trainer = Trainer(
        gpus=gpus,
        max_epochs=epochs,
        max_steps=(ldata.trainset_size + batch_size - 1) // batch_size * epochs,
        logger=tb_logger,
        callbacks=[bar]
    )
    trainer.fit(lmodel, ldata)



if __name__ == '__main__':
    main()