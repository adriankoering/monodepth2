import torch
from torch import nn

from kornia import linalg

from .trainingmodule import TestModule

from plmodels import depthmodules, posemodules


class MonoDepth2(TestModule):

  def __init__(self, depthmodel, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.depth_model = getattr(depthmodules, depthmodel)(*args, **kwargs)
    self.pose_model = posemodules.PoseModule(*args, **kwargs)

  def forward(self, Iprev, Icenter, Inext):
    inv_depths = self.depth_model(Icenter)

    # predict poses with consistent temporal order (cat along channels)
    Tprev = self.pose_model(torch.cat([Iprev, Icenter], dim=1))
    Tnext = self.pose_model(torch.cat([Icenter, Inext], dim=1))

    # invert Tprev, because we need pixels to go from center to adjacent
    Tprev = linalg.inverse_transformation(Tprev)

    return inv_depths, Tprev, Tnext
