# Added by Adrian Köring

import torch
from torch import nn
import torch.nn.functional as F

from kornia import geometry

from .encoders import ResnetEncoder as ResNet


class PoseModule(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.encoder = ResNet(pretrained=True, num_input_images=2, *args, **kwargs)

    *_, f4 = self.encoder.out_channels

    self.pose = nn.Sequential(
        nn.Conv2d(f4, 256, 1),
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

  def forward(self, x):
    *_, x = self.encoder(x)
    x = 0.01 * self.pose(x)
    return self.to_transform(x)
