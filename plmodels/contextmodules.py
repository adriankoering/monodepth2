# Added by Adrian Köring

import torch
from torch import nn
from torch.nn import functional as F


class ASPP(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               pyramid=[2, 4, 8],
               *args,
               **kwargs):
    super().__init__()

    l0, l1 = pyramid
    out_channels //= 2

    self.cn0 = nn.Conv2d(in_channels, out_channels, 3, padding=l0, dilation=l0)
    self.bn0 = nn.BatchNorm2d(out_channels)

    self.cn1 = nn.Conv2d(in_channels, out_channels, 3, padding=l1, dilation=l1)
    self.bn1 = nn.BatchNorm2d(out_channels)

  def forward(self, x):

    o0 = self.bn0(self.cn0(x)).relu()
    o1 = self.bn1(self.cn1(x)).relu()

    return torch.cat([x, o0, o1], 1)


class PSP(nn.Module):

  def __init__(self, in_channels, out_channels, pyramid, *args, **kwargs):
    super().__init__()

    p2, p1 = pyramid
    out_channels //= 2

    self.pool2 = nn.Sequential(
        nn.AdaptiveAvgPool2d((p2, p2)),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

    self.pool1 = nn.Sequential(
        nn.AdaptiveAvgPool2d((p1, p1)),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    B, C, H, W = x.shape

    o2 = F.interpolate(
        self.pool2(x),
        size=(H, W),
        mode='bilinear',
        align_corners=True,
    )
    o1 = F.interpolate(
        self.pool1(x),
        size=(H, W),
        mode='bilinear',
        align_corners=True,
    )

    return torch.cat([x, o2, o1], dim=1)


class Dorn(nn.Module):

  def __init__(self, in_channels, out_channels, pyramid, *args, **kwargs):
    super().__init__()

    self.aspp = ASPP(in_channels, out_channels, pyramid)
    self.psp = PSP(in_channels, out_channels, [1, 4])

  def forward(self, x):

    a = self.aspp(x)
    p = self.psp(x)

    return torch.cat([a, p], 1)
