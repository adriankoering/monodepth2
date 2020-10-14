# Added by Adrian Köring

import torch
from torch import nn
from kornia import augmentation


class NoCrop(nn.Module):

  def __init__(self, image_size):
    """ Add camera intrinsics to unaltered images """
    super().__init__()

    self.image_size = image_size
    H, W = image_size

    # canonical intrinsics as used in reference (to be scaled by image size)
    self.K = torch.tensor([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1.]])
    self.K[0] *= W
    self.K[1] *= H

  def stacked_K(self, B):
    """ Add batch dimension to intrinsics"""
    return torch.stack(B * [self.K])

  def forward(self, Iprev, Ic, Inext):
    B, C, H, W = Ic.shape
    return Iprev, Ic, Inext, self.stacked_K(B).to(Ic.device)


class RandomCrop(NoCrop):

  def __init__(self, image_size, crop_size, p=0.5):
    """ Crop images (uniform across sequence) and return matching intrinsics """
    super().__init__(image_size)
    self.crop_size = crop_size
    self.p = p

  def apply_crop(self, Ks, params):
    """ Apply crop to camera intrinsics
        (subtracts top-left crop corner from principal point)
    """
    src = params["src"]

    LT = src[:, 0]  # left, top corner
    Ks[..., 0, -1] = Ks[..., 0, -1] - LT[:, 0]
    Ks[..., 1, -1] = Ks[..., 1, -1] - LT[:, 1]

    return Ks

  def forward(self, Iprev, Ic, Inext):
    """ Apply crops and adjust intrinsics during training """

    if not self.training or torch.rand([]) < self.p:
      # Random crop only augments training
      return super().forward(Iprev, Ic, Inext)

    B, C, H, W = Ic.shape

    params = augmentation.random_generator.random_crop_generator(
        batch_size=B,
        input_size=self.image_size,
        size=self.crop_size,
    )

    Ic = augmentation.functional.apply_crop(Ic, params)
    Iprev = augmentation.functional.apply_crop(Iprev, params)
    Inext = augmentation.functional.apply_crop(Inext, params)

    Ks = self.stacked_K(B)
    Ks = self.apply_crop(Ks, params)

    return Iprev, Ic, Inext, Ks.to(Ic.device)
