# Added by Adrian Köring

import torch
from torch import nn
from torch import functional as F
from torchvision import models

from kornia.enhance import Normalize


class ResnetEncoder(nn.Module):
  """ Pytorch module for a resnet encoder """

  def __init__(self,
               backbone,
               pretrained,
               num_input_images=1,
               downsampling=[True, True, True, True, True],
               *args,
               **kwargs):
    super().__init__()

    normalize = Normalize(
        mean=torch.tensor(num_input_images * [0.485, 0.456, 0.406]),
        std=torch.tensor(num_input_images * [0.229, 0.224, 0.225]),
    )

    resnet = getattr(models, backbone)(pretrained=pretrained)

    input_conv = self._adapt_first_input(resnet.conv1, num_input_images)

    self.layer0 = nn.Sequential(normalize, input_conv, resnet.bn1, resnet.relu)
    self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3
    self.layer4 = resnet.layer4

    dilation = 1
    for i, downsample in enumerate(downsampling):
      # downsampling is default: refactor to remove
      # removing it is not implemented for .layer{0, 1}
      if i > 1 and not downsample:
        dilation *= 2
        self.remove_downsampling(getattr(self, f"layer{i}"), dilation)

    f0 = resnet.conv1.out_channels
    f1 = self._get_out_channels(resnet.layer1[-1])
    f2 = self._get_out_channels(resnet.layer2[-1])
    f3 = self._get_out_channels(resnet.layer3[-1])
    f4 = self._get_out_channels(resnet.layer4[-1])

    self.out_channels = [f0, f1, f2, f3, f4]

  def _adapt_first_input(self, input_conv, num_input_images):

    if num_input_images > 1:

      # recreate first conv layer with '3 * num_input_images' many channels
      new_conv = nn.Conv2d(
          num_input_images * input_conv.in_channels,
          input_conv.out_channels,
          kernel_size=input_conv.kernel_size,
          stride=input_conv.stride,
          padding=input_conv.padding,
          bias=input_conv.bias)

      # duplicate weights along input channels (dim = 1)
      #  scale down and re-assign weight back into layer and layer into model
      new_weight = torch.cat(num_input_images * [input_conv.weight.data], dim=1)
      new_conv.weight.data = new_weight / num_input_images
    else:
      # nothing to change
      new_conv = input_conv

    return new_conv

  def remove_downsampling(self, module, dilation):
    block0, *_ = module  # select first from any number of elements
    block0.conv1.stride = (1, 1)  # remove downsampling via total_stride
    block0.downsample[0].stride = (1, 1)

    # dilate all conv-layers by 'dilation' to compensate for removing stride
    def dilate(layer):
      layer.padding = (dilation, dilation)
      layer.dilation = (dilation, dilation)

    for m in module:
      for layer in m.children():
        if isinstance(layer, nn.Conv2d):
          dilate(layer)

  def _get_out_channels(self, module):
    out_channels = 0
    for m in module.children():
      try:
        out_channels = m.out_channels
      except AttributeError as e:  # not a conv-layer
        pass
    return out_channels

  def forward(self, x):
    f0 = self.layer0(x)
    f1 = self.layer1(f0)
    f2 = self.layer2(f1)
    f3 = self.layer3(f2)
    f4 = self.layer4(f3)

    # high to low-res features
    return [f0, f1, f2, f3, f4]
