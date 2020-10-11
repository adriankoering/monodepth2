#!/usr/bin/env python
# coding: utf-8

# Added by Adrian Köring

import yaml
import argparse
from pathlib import Path

import kornia
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from plmodels import TrainingModule as Model
from pldatasets import KittiTestModule as Dataset


def parse_arguments():
  parser = argparse.ArgumentParser()

  # model checkpoint to visualize
  parser.add_argument("-c", "--ckpt", type=str, required=True)

  args = parser.parse_args()

  p = Path(args.ckpt)
  args.name = p.parent.parent.parent.stem.split("_")[0]
  args.version = f"version_{p.parent.parent.stem[0]}"
  return args


ARGS = parse_arguments()


def plot(image, idepth):

  fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))

  for ax in (ax0, ax1):
    ax.set_xticks([])
    ax.set_yticks([])

  ax0.set_title("Image")
  aximg = ax0.imshow(kornia.tensor_to_image(image))

  ax1.set_title("Inverse Depth")
  idepth = kornia.tensor_to_image(idepth)
  aximg = ax1.imshow(idepth, cmap="magma")  #  vmin=0, vmax=1,

  plt.savefig("testplot.jpg", dpi=100)


def main():

  model = Model.load_from_checkpoint(checkpoint_path=ARGS.ckpt)

  ds = Dataset(image_size=model.hparams["image_size"])
  ds.setup()
  vis_dl = ds.test_dataloader()

  # optionally: check if all data exists and is loadable
  for image, gt_depth in vis_dl:
    *_, pred = model.depth_model(image)
    depth, idepth = model.disentangle(pred)
    print(pred.shape, idepth.min(), idepth.max())
    plot(image, idepth)
    break


if __name__ == "__main__":
  main()
