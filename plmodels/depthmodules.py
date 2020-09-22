from torch import nn

from .encoders import ResnetEncoder
from .decoders import DepthDecoder


def UNet(*args, **kwargs):
  return nn.Sequential(
      ResnetEncoder(pretrained=True, *args, **kwargs),
      DepthDecoder(*args, **kwargs),
  )


class DeepLab(nn.Module):

  def __init__(self, regression, *args, **kwargs):
    pass

  def forward(self, x):
    return x
