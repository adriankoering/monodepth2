from torch import nn
from torchvision import models

from .encoders import ResnetEncoder
from .decoders import DepthDecoder


class UNet(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()

    self.encoder = ResnetEncoder(pretrained=True, *args, **kwargs)
    self.decoder = DepthDecoder(self.encoder.out_channels, *args, **kwargs)

  def forward(self, x):
    e = self.encoder(x)
    return self.decoder(e)


class DeepLab(nn.Module):

  def __init__(self, regression, *args, **kwargs):
    super().__init__()
    num_classes = 1 if regression else 128  # TODO: 128?
    self.depth = models.segmentation.deeplabv3_resnet50(
        pretrained=False,
        progress=False,
        num_classes=num_classes,
        aux_loss=True,
    )

  def forward(self, x):
    out_dict = self.depth(x)
    return [out_dict["aux"], out_dict["out"]]
