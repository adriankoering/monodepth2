import torch
from torch import nn


class md2(nn.Module):

  def __init__(self, in_channels, out_channels, pyramid, *args, **kwargs):
    super().__init__()

  def forward(self, x):
    return x


class aspp(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               pyramid=[6, 12, 18],
               *args,
               **kwargs):
    super().__init__()

    out_channels = out_channels // 4

    l0, l1, l2 = pyramid

    self.cn1 = nn.Conv2d(in_channels, out_channels, 1)
    self.bn1 = nn.BatchNorm2d(out_channels)

    self.cn6 = nn.Conv2d(in_channels, out_channels, 3, padding=l0, dilation=l0)
    self.bn6 = nn.BatchNorm2d(out_channels)

    self.cn12 = nn.Conv2d(in_channels, out_channels, 3, padding=l1, dilation=l1)
    self.bn12 = nn.BatchNorm2d(out_channels)

    self.cn18 = nn.Conv2d(in_channels, out_channels, 3, padding=l2, dilation=l2)
    self.bn18 = nn.BatchNorm2d(out_channels)

  def forward(self, x):

    o1 = self.bn1(self.cn1(x)).relu()
    o6 = self.bn6(self.cn6(x)).relu()
    o12 = self.bn12(self.cn12(x)).relu()
    o18 = self.bn18(self.cn18(x)).relu()

    return torch.cat([o1, o6, o12, o18], 1)


class psp(nn.Module):

  def __init__(self, in_channels, out_channels, pyramid, *args, **kwargs):

    out_channels = out_channels // 3

    p2, p1 = pyramid

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
    )

    self.pool2 = nn.Sequential(
        nn.AdaptiveAvgPool2d((p2, p2)),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
    )

    self.pool1 = nn.Sequential(
        nn.AdaptiveAvgPool2d((p1, p1)),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
    )

  def forward(self, x):
    B, C, H, W = x.shape

    oc = self.conv(x)
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

    return torch.cat([oc, o2, o1], dim=1)
