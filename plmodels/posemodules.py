from torch import nn

from .encoders import ResnetEncoder
from .decoders import PoseDecoder


class PoseModule(nn.Module):

  def __init__(self, *args, **kwargs):
    super().__init__()
    self.encoder = ResnetEncoder(
        pretrained=True, num_input_images=2, *args, **kwargs)
    self.decoder = PoseDecoder(self.encoder.out_channels, *args, **kwargs)

  def forward(self, x):
    e = self.encoder(x)
    return self.decoder(e)
