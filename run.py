from argparse import Namespace
from typing import List, Optional
import sys, os
sys.path.append(os.path.dirname(__file__))

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from dataloader import ICDAR21
from pl_models import PlResNetTransformer

@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    datamodule = ICDAR21(**cfg.data)
    datamodule.setup()

    pl_model = PlResNetTransformer(**cfg.pl_model)

    callbacks: List[Callback] = []
    if cfg.callbacks.model_checkpoint:
        callbacks.append(ModelCheckpoint(**cfg.callbacks.model_checkpoint))
    if cfg.callbacks.early_stopping:
        callbacks.append(EarlyStopping(**cfg.callbacks.early_stopping))

    logger: Optional[WandbLogger] = None
    if cfg.logger:
        logger = WandbLogger(**cfg.logger)

    trainer = Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    if trainer.logger:
        trainer.logger.log_hyperparams(Namespace(**cfg))

    trainer.tune(pl_model, datamodule=datamodule)
    trainer.fit(pl_model, datamodule=datamodule)
    trainer.test(pl_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
