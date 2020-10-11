#!/usr/bin/env python
# coding: utf-8

# Added by Adrian Köring

import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl

from plmodels import TrainingModule as Model
import pldatasets as datasets


def parse_arguments():
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", "--config", type=Path, required=True)

  args = parser.parse_args()

  with args.config.open("r") as config:
    args.hparams = yaml.safe_load(config)

  return args


ARGS = parse_arguments()

Dataset = getattr(datasets, ARGS.hparams.pop("dataset") + "DaliModule")


def main():

  model = Model(**ARGS.hparams)
  ds = Dataset(**ARGS.hparams)

  Iprev = Icenter = Inext = torch.randn(2, 3, 128, 416)
  # Iprev = Icenter = Inext = torch.randn(2, 3, 192, 640)
  disps, Tprev, Tcenter = model(Iprev, Icenter, Inext)
  print([disp.shape for disp in disps])

  # from plmodels.trainingmodule import RandomCrop
  # crop = RandomCrop([192, 640], [128, 416])
  #
  # Iprev, Ic, Inext, Ks = crop.forward(Iprev, Icenter, Inext)


if __name__ == "__main__":
  main()
