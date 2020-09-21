from layers import ConvBlock, Conv3x3

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia import geometry


class DecoderLayer(nn.Module):

  def __init__(self, in_channels, out_channels, concat_channels=0, output=True):
    super().__init__()
    kwargs = {"kernel_size": 3, "padding": 1, "padding_mode": "reflect"}

    self.c0 = nn.Conv2d(in_channels, out_channels, **kwargs)
    self.b0 = nn.BatchNorm2d(out_channels)
    self.c1 = nn.Conv2d(out_channels + concat_channels, out_channels, **kwargs)
    self.b1 = nn.BatchNorm2d(out_channels)

    if output:
      self.output = nn.Conv2d(out_channels, 1, **kwargs)
    else:
      self.output = None

  def forward(self, x, skip=None):
    x = F.elu(self.b0(self.c0(x)))
    x = F.interpolate(x, scale_factor=2, mode="nearest")

    if skip is not None:
      x = torch.cat([x, skip], dim=1)

    x = F.elu(self.b1(self.c1(x)))
    o = self.output(x).sigmoid() if self.output else None

    return x, o


class DepthDecoder(nn.Module):

  def __init__(self):
    super().__init__()

    self.upconv4 = DecoderLayer(512, 256, 256, output=False)
    self.upconv3 = DecoderLayer(256, 128, 128)
    self.upconv2 = DecoderLayer(128, 64, 64)
    self.upconv1 = DecoderLayer(64, 32, 64)
    self.upconv0 = DecoderLayer(32, 16)

  def forward(self, features):
    f0, f1, f2, f3, f4 = features

    x, _ = self.upconv4(f4, f3)
    x, o3 = self.upconv3(x, f2)
    x, o2 = self.upconv2(x, f1)
    x, o1 = self.upconv1(x, f0)
    _, o0 = self.upconv0(x)

    return o3, o2, o1, o0


class PoseDecoder(nn.Module):

  def __init__(self):
    super().__init__()
    self.pose = nn.Sequential(
        nn.Conv2d(512, 256, 1),  # squeeze
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 6, 1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )

  def to_transform(self, predictions):
    aa, t = predictions.chunk(2, dim=-1)
    R = geometry.angle_axis_to_rotation_matrix(aa)
    # attaching translation creates 3x4 transformation matrix
    T = torch.cat([R, t.unsqueeze(-1)], dim=-1)
    # return 4x4 translation matrix (adds [0, 0, 0, 1] as last row)
    return geometry.convert_affinematrix_to_homography3d(T)

  def forward(self, features):
    *_, x = features
    x = 0.01 * self.pose(x)
    return self.to_transform(x)
