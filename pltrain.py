#!/usr/bin/env python
# coding: utf-8

import yaml
import argparse
from pathlib import Path

import pytorch_lightning as pl

import plmodels as models
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

Model = getattr(models, ARGS.hparams.pop("model"))
Dataset = getattr(datasets, ARGS.hparams.pop("dataset") + "DaliModule")


def main():

  tb_logger = pl.loggers.TensorBoardLogger("lightning_logs", name=ARGS.name)
  # wb_logger = pl.loggers.WandbLogger(project="depth-estimation", name=ARGS.name)
  lr_logger = pl.callbacks.LearningRateLogger()
  # gpu_logger = pl.callbacks.GpuUsageLogger(temperature=True)
  trainer = pl.Trainer(
      gpus=1,
      max_epochs=ARGS.hparams["max_epochs"],
      logger=[tb_logger],  # wb_logger
      callbacks=[lr_logger],
      default_root_dir="ckpts",
      # checkpoint_callback=False,
      terminate_on_nan=True,
      profiler=True,
      # resume_from_checkpoint="ckpts/rn18lowressmooth_depth-estimation/1_170hdqox/checkpoints/epoch=3.ckpt"
  )

  model = Model(**ARGS.hparams)
  ds = Dataset(**ARGS.hparams)
  trainer.fit(model, ds)
  # trainer.test(model)


if __name__ == "__main__":
  main()
