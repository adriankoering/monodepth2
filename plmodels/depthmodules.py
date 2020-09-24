from torch import nn

from .encoders import ResnetEncoder
from .decoders import DepthDecoder


class UNet(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()

    self.encoder = ResnetEncoder(pretrained=True, *args, **kwargs)
    self.decoder = DepthDecoder(self.encoder.out_channels, *args, **kwargs)

  def forward(self, x):
    e = self.encoder(x)
    return self.decoder(e)


class DeepLab(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    pass

  def forward(self, x):
    return x
