#!/usr/bin/env python
# coding: utf-8

import yaml
import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl

import plmodels as models
import pldatasets as datasets


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

  model = Model(**ARGS.hparams)
  ds = Dataset(**ARGS.hparams)

  Iprev = Icenter = Inext = torch.randn(12, 3, 128, 416)
  disps, Tprev, Tcenter = model(Iprev, Icenter, Inext)
  print([disp.shape for disp in disps])


if __name__ == "__main__":
  main()
