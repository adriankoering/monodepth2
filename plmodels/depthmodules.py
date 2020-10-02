import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .encoders import ResnetEncoder as ResNet
from plmodels import contextmodules


class DecoderLayer(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               concat_channels=0,
               upsampling=True,
               output=True,
               regression=True):
    super().__init__()
    kwargs = {"kernel_size": 3, "padding": 1, "padding_mode": "reflect"}

    self.c0 = nn.Conv2d(in_channels, out_channels, **kwargs)
    self.b0 = nn.BatchNorm2d(out_channels)
    self.c1 = nn.Conv2d(out_channels + concat_channels, out_channels, **kwargs)
    self.b1 = nn.BatchNorm2d(out_channels)

    if upsampling:
      self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    else:
      self.upsample = nn.Identity()

    if output:
      regression_channels = 1 if regression else 80
      self.output = nn.Conv2d(out_channels, regression_channels, **kwargs)

    else:
      self.output = None

  def forward(self, x, skip=None):
    x = F.elu(self.b0(self.c0(x)))
    x = self.upsample(x)

    if skip is not None:
      x = torch.cat([x, skip], dim=1)

    x = F.elu(self.b1(self.c1(x)))
    o = self.output(x).sigmoid() if self.output else None

    return x, o


class MonoDepth2(nn.Module):

  def __init__(self, regression, *args, **kwargs):
    super().__init__()

    ds = 5 * [True]
    self.encoder = ResNet(pretrained=True, downsampling=ds, *args, **kwargs)
    f0, f1, f2, f3, f4 = self.encoder.out_channels

    u0, u1, u2, u3, u4 = ds
    self.upconv4 = DecoderLayer(f4, 256, f3, u4, output=False)
    self.upconv3 = DecoderLayer(256, 128, f2, u3, regression=regression)
    self.upconv2 = DecoderLayer(128, 64, f1, u2, regression=regression)
    self.upconv1 = DecoderLayer(64, 32, f0, u1, regression=regression)
    self.upconv0 = DecoderLayer(32, 16, 0, u0, regression=regression)

  def forward(self, images):
    f0, f1, f2, f3, f4 = self.encoder(images)

    x, _ = self.upconv4(f4, f3)
    x, o3 = self.upconv3(x, f2)
    x, o2 = self.upconv2(x, f1)
    x, o1 = self.upconv1(x, f0)
    _, o0 = self.upconv0(x)

    return o3, o2, o1, o0  # high-res last (for visualization)


class Dorn(nn.Module):

  def __init__(self, context, regression, *args, **kwargs):
    super().__init__()

    ds = [True, True, True, False, False]  # downsample layer0 through layer4
    self.encoder = ResNet(pretrained=True, downsampling=ds, *args, **kwargs)

    # skip upconvs without downsampling
    f0, f1, f2, _, _ = self.encoder.out_channels
    u0, u1, u2, _, _ = ds

    self.context = getattr(contextmodules, context)(f2, f2, *args, **kwargs)

    # self.upconv4 = DecoderLayer(f4, 256, f3, u4, output=False)
    # self.upconv3 = DecoderLayer(256, 128, f2, u3, regression=regression)
    self.upconv2 = DecoderLayer(f2, 64, f1, u2, regression=regression)
    self.upconv1 = DecoderLayer(64, 32, f0, u1, regression=regression)
    self.upconv0 = DecoderLayer(32, 16, 0, u0, regression=regression)

  def forward(self, images):
    f0, f1, f2, f3, f4 = self.encoder(images)

    f4 = self.context(f4)

    x, o2 = self.upconv2(f4, f1)
    x, o1 = self.upconv1(x, f0)
    _, o0 = self.upconv0(x)

    return o2, o1, o0
