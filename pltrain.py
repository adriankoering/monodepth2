#!/usr/bin/env python
# coding: utf-8

import yaml
import argparse
from pathlib import Path

import pytorch_lightning as pl

from plmodels import TrainingModule as Model
import pldatasets as datasets

# import os
# os.environ["WANDB_API_KEY"] = Path("/workspace/wandb.key").read_text().strip()
#
# import wandb


def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", "--config", type=Path, required=True)
  parser.add_argument("-n", "--name", type=str, required=True)

  args = parser.parse_args()

  with args.config.open("r") as config:
    args.hparams = yaml.safe_load(config)

  return args


ARGS = parse_arguments()

Dataset = getattr(datasets, ARGS.hparams.pop("dataset") + "DaliModule")


def main():

  tb_logger = pl.loggers.TensorBoardLogger("lightning_logs", name=ARGS.name)
  # wb_logger = pl.loggers.WandbLogger(project="depth-estimation", name=ARGS.name)
  lr_logger = pl.callbacks.LearningRateLogger()
  # gpu_logger = pl.callbacks.GpuUsageLogger(temperature=True)

  ckpt_cb = pl.callbacks.ModelCheckpoint(
      monitor='val_loss',
      verbose=True,
      save_top_k=5,
      save_last=True,
  )

  trainer = pl.Trainer(
      gpus=1,
      max_epochs=ARGS.hparams["max_epochs"],
      logger=[tb_logger],  # wb_logger
      callbacks=[lr_logger],
      checkpoint_callback=ckpt_cb,
      weights_save_path="ckpts",
      terminate_on_nan=True,
      profiler=True,
      # resume_from_checkpoint="ckpts/rn18lowressmooth_depth-estimation/1_170hdqox/checkpoints/epoch=3.ckpt"
  )

  model = Model(**ARGS.hparams)
  ds = Dataset(**ARGS.hparams)
  trainer.fit(model, ds)

  print(ckpt_cb.best_model_path, ckpt_cb.best_model_score)


if __name__ == "__main__":
  main()
