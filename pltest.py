#!/usr/bin/env python
# coding: utf-8

import yaml
import argparse
from pathlib import Path

import pytorch_lightning as pl

from plmodels import TrainingModule as Model
from pldatasets import KittiTestModule as Dataset


def parse_arguments():
  parser = argparse.ArgumentParser()

  # model checkpoint to test
  parser.add_argument("-c", "--ckpt", type=str, required=True)
  # parser.add_argument("-n", "--name", type=str, required=True)

  args = parser.parse_args()

  p = Path(args.ckpt)
  args.name = p.parent.parent.parent.stem.split("_")[0]
  args.version = f"version_{p.parent.parent.stem[0]}"
  return args


ARGS = parse_arguments()


def main():

  # tb_logger = pl.loggers.TensorBoardLogger(
  #     "lightning_logs", name=ARGS.name, version=ARGS.version)
  trainer = pl.Trainer(gpus=1, logger=False)  # , logger=[tb_logger])

  model = Model.load_from_checkpoint(checkpoint_path=ARGS.ckpt)
  ds = Dataset(image_size=model.hparams["image_size"])
  ds.setup()
  test_dl = ds.test_dataloader()

  # # optionally: check if all data exists and is loadable
  # for image, depth in test_dl:
  #   pass

  test_dl = ds.test_dataloader()
  trainer.test(model, test_dataloaders=test_dl)


if __name__ == "__main__":
  main()
