# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):

  def __init__(self,
               num_ch_enc,
               num_input_features,
               num_frames_to_predict_for=None,
               stride=1):
    super(PoseDecoder, self).__init__()

    self.num_ch_enc = num_ch_enc
    self.num_input_features = num_input_features

    if num_frames_to_predict_for is None:
      num_frames_to_predict_for = num_input_features - 1
    self.num_frames_to_predict_for = num_frames_to_predict_for

    self.convs = OrderedDict()
    self.squeeze = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
    self.convs = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(num_input_features * 256, 256, 3, stride, 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, stride, 1),
        nn.ReLU(),
        nn.Conv2d(256, 6 * num_frames_to_predict_for, 1),
    )

  def forward(self, input_features):
    """ input features is a list with one element: this element has 5 tensors """
    # last features is a list with one element, where this element contains the
    # most low-res (most embedded) layer
    last_features = [f[-1] for f in input_features]

    # squeezes reduces channel dimensions across last_features
    cat_features = [self.squeeze(f) for f in last_features]
    cat_features = torch.cat(cat_features, 1)

    out = self.convs(cat_features)
    out = out.mean(3).mean(2)
    out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

    axisangle = out[..., :3]
    translation = out[..., 3:]

    return axisangle, translation
