# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch


def inverse_depth_smoothness_loss(disp, img):
  """ Computes the smoothness loss for a disparity image
      The color image is used for edge-aware smoothness
  """
  grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
  grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

  grad_img_x = torch.mean(
      torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
  grad_img_y = torch.mean(
      torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

  grad_disp_x *= torch.exp(-grad_img_x)
  grad_disp_y *= torch.exp(-grad_img_y)

  return grad_disp_x.mean() + grad_disp_y.mean()
