# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia import geometry, linalg


def disp_to_depth(disp, min_depth, max_depth):
  """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
  min_disp = 1 / max_depth
  max_disp = 1 / min_depth
  scaled_disp = min_disp + (max_disp - min_disp) * disp
  depth = 1 / scaled_disp
  return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
  """ Convert pose estimation (axisangle, translation) into a 4x4 matrix
  """
  R = geometry.angle_axis_to_rotation_matrix(axisangle)
  T = torch.cat([R, t.unsqueeze(-1)], dim=-1)
  # T ix 3x4, add [0, 0, 0, 1] as last row to form 4x4
  M = geometry.convert_affinematrix_to_homography3d(T)

  if invert:
    M = linalg.inverse_transformation(M)

  return M


class ConvBlock(nn.Module):
  """Layer to perform a convolution followed by ELU
    """

  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()

    self.conv = Conv3x3(in_channels, out_channels)
    self.nonlin = nn.ELU(inplace=True)

  def forward(self, x):
    out = self.conv(x)
    out = self.nonlin(out)
    return out


class Conv3x3(nn.Module):
  """Layer to pad and convolve input
    """

  def __init__(self, in_channels, out_channels, use_refl=True):
    super(Conv3x3, self).__init__()

    if use_refl:
      self.pad = nn.ReflectionPad2d(1)
    else:
      self.pad = nn.ZeroPad2d(1)
    self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

  def forward(self, x):
    out = self.pad(x)
    out = self.conv(out)
    return out


class BackprojectDepth(nn.Module):
  """Layer to transform a depth image into a point cloud
    """

  def __init__(self, batch_size, height, width):
    super(BackprojectDepth, self).__init__()

    self.batch_size = batch_size
    self.height = height
    self.width = width

    meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
    self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    self.id_coords = nn.Parameter(
        torch.from_numpy(self.id_coords), requires_grad=False)

    self.ones = nn.Parameter(
        torch.ones(self.batch_size, 1, self.height * self.width),
        requires_grad=False)

    self.pix_coords = torch.unsqueeze(
        torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)],
                    0), 0)
    self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
    self.pix_coords = nn.Parameter(
        torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

  def forward(self, depth, inv_K):
    cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
    cam_points = depth.view(self.batch_size, 1, -1) * cam_points
    cam_points = torch.cat([cam_points, self.ones], 1)

    return cam_points


class Project3D(nn.Module):
  """Layer which projects 3D points into a camera with intrinsics K and at position T
    """

  def __init__(self, batch_size, height, width, eps=1e-7):
    super(Project3D, self).__init__()

    self.batch_size = batch_size
    self.height = height
    self.width = width
    self.eps = eps

  def forward(self, points, K, T):
    P = torch.matmul(K, T)[:, :3, :]

    cam_points = torch.matmul(P, points)

    pix_coords = cam_points[:, :2, :] / (
        cam_points[:, 2, :].unsqueeze(1) + self.eps)
    pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[..., 0] /= self.width - 1
    pix_coords[..., 1] /= self.height - 1
    pix_coords = (pix_coords - 0.5) * 2
    return pix_coords


def upsample(x):
  """Upsample input tensor by a factor of 2
    """
  return F.interpolate(x, scale_factor=2, mode="nearest")


def compute_depth_errors(gt, pred):
  """Computation of error metrics between predicted and ground truth depths
    """
  thresh = torch.max((gt / pred), (pred / gt))
  a1 = (thresh < 1.25).float().mean()
  a2 = (thresh < 1.25**2).float().mean()
  a3 = (thresh < 1.25**3).float().mean()

  rmse = (gt - pred)**2
  rmse = torch.sqrt(rmse.mean())

  rmse_log = (torch.log(gt) - torch.log(pred))**2
  rmse_log = torch.sqrt(rmse_log.mean())

  abs_rel = torch.mean(torch.abs(gt - pred) / gt)

  sq_rel = torch.mean((gt - pred)**2 / gt)

  return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
