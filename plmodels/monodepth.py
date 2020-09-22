from .encoders import ResnetEncoder
from .decoders import DepthDecoder, PoseDecoder
from .trainingmodule import TestModule

import torch
from torch import nn

from kornia import linalg

from plmodels import depthmodules


class MonoDepth2(TestModule):

  def __init__(self, depthmodel, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.depth_model = getattr(depthmodules, depthmodel)(*args, **kwargs)

    self.pose_model = nn.Sequential(
        ResnetEncoder(pretrained=True, num_input_images=2, *args, **kwargs),
        PoseDecoder(),
    )

  def forward(self, Iprev, Icenter, Inext):
    inv_depths = self.depth_model(Icenter)

    # predict poses with consistent temporal order (cat along channels)
    Tprev = self.pose_model(torch.cat([Iprev, Icenter], dim=1))
    Tnext = self.pose_model(torch.cat([Icenter, Inext], dim=1))

    # invert Tprev, because we need pixels to go from center to adjacent
    Tprev = linalg.inverse_transformation(Tprev)

    return inv_depths, Tprev, Tnext
