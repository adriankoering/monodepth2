# Added by Adrian Köring

from .kitti_dataset import KittiDataset, KittiTestset
from .kitti_daliset import KittiTFRecordPipeline

import torch
from torch.utils import data
from torchvision import transforms

import pytorch_lightning as pl
from nonechucks import SafeDataset

from nvidia.dali.plugin.pytorch import DALIGenericIterator


def collate_fn(batch):
  prevs, centers, nexts = [], [], []
  for (prev, center, next) in batch:
    prevs.append(prev)
    centers.append(center)
    nexts.append(next)

  return [{
      "prev": torch.stack(prevs),
      "center": torch.stack(centers),
      "next": torch.stack(nexts)
  }]


def preprocessing(image_size):
  return transforms.Resize(image_size)


class KittiDataModule(pl.LightningDataModule):

  def __init__(self, split, image_size, batch_size, *args, **kwargs):
    super().__init__()

    self.split = split
    self.image_size = image_size
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_ds = SafeDataset(
        KittiDataset(
            self.split,
            train=True,
            transform=self.preprocessing(),
        ))

    self.val_ds = SafeDataset(
        KittiDataset(
            self.split,
            train=False,
            transform=self.preprocessing(),
        ))
    num_examples = len(self.train_ds), len(self.val_ds)
    print(f"Train/Val/Test Examples: {num_examples}")

  def train_dataloader(self):
    return data.DataLoader(
        self.train_ds,
        batch_size=self.batch_size,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
    )

  def val_dataloader(self):
    return data.DataLoader(
        self.val_ds,
        batch_size=self.batch_size,
        num_workers=16,
        collate_fn=collate_fn,
        pin_memory=True,
    )


class KittiTestModule(pl.LightningDataModule):

  def __init__(self, image_size):
    super().__init__()
    self.image_size = image_size

  def setup(self, stage=None):
    tfs = transforms.Compose([
        # pytorch 1.6 does not yet support resize for tensors (1.6-nightly does)
        transforms.ToPILImage(),
        transforms.Resize(self.image_size),
        transforms.ToTensor(),
    ])
    self.test_ds = KittiTestset(transform=tfs)  # SafeDataset()
    print(f"Test Examples: {len(self.test_ds)}")

  def test_dataloader(self):
    return data.DataLoader(
        self.test_ds,
        batch_size=1,
        num_workers=16,
        pin_memory=True,
    )


class IteratorLengthWrapper:

  def __init__(self, pipe, output_map=["prev", "center", "next"]):
    self.pipe = pipe
    self.outputs = output_map

  def __len__(self):
    return self.pipe.epoch_size(name="tfreader") // self.pipe.batch_size

  def __iter__(self):
    return DALIGenericIterator(self.pipe, self.outputs, reader_name="tfreader")


class KittiDaliModule(pl.LightningDataModule):

  def __init__(self, split, image_size, batch_size, *args, **kwargs):
    super().__init__()

    self.split = split
    self.image_size = image_size
    self.batch_size = batch_size

  def setup(self, stage=None):
    self.train_ds = KittiTFRecordPipeline(
        self.split,
        train=True,
        shuffle=True,
        image_size=self.image_size,
        batch_size=self.batch_size,
        num_threads=4)
    self.train_ds.build()

    self.val_ds = KittiTFRecordPipeline(
        self.split,
        train=False,
        shuffle=False,
        image_size=self.image_size,
        batch_size=self.batch_size,
        num_threads=4)
    self.val_ds.build()

    len_train = self.train_ds.epoch_size(name="tfreader")
    len_val = self.val_ds.epoch_size(name="tfreader")
    print(f"Train/Val Examples: {len_train}, {len_val}")

  def train_dataloader(self):
    return IteratorLengthWrapper(self.train_ds)

  def val_dataloader(self):
    return IteratorLengthWrapper(self.val_ds)
