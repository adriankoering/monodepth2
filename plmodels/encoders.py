import torch
from torch import nn
from torch import functional as F
from torchvision import models

from kornia.enhance import Normalize

from plmodels import contextmodules


class ResnetEncoder(nn.Module):
  """ Pytorch module for a resnet encoder """

  def __init__(self,
               backbone,
               context,
               pretrained,
               num_input_images=1,
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

    f0 = resnet.conv1.out_channels
    f1 = self._get_out_channels(resnet.layer1[-1])
    f2 = self._get_out_channels(resnet.layer2[-1])
    f3 = self._get_out_channels(resnet.layer3[-1])
    f4 = self._get_out_channels(resnet.layer4[-1])

    self.out_channels = [f0, f1, f2, f3, f4]

    self.context = getattr(contextmodules, context)(
        in_channels=f4, out_channels=f4, *args, **kwargs)

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
    f4 = self.context(self.layer4(f3))

    # high to low-res features
    return [f0, f1, f2, f3, f4]
