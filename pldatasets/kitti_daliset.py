from subprocess import call
from pathlib import Path

from nvidia.dali.pipeline import Pipeline

from nvidia.dali import ops
from nvidia.dali import types
import nvidia.dali.tfrecord as tfrec

import numpy as np


class KittiTFRecordPipeline(Pipeline):

  def __init__(self,
               split,
               train,
               transform=None,
               data_dir="kitti_data",
               shuffle=False,
               image_size=[128, 416],
               batch_size=12,
               num_threads=4,
               device_id=0):
    super().__init__(batch_size, num_threads, device_id)

    mode = "train" if train else "val"
    tfrecordfile = Path(data_dir) / split / (mode + "_small.tfrecord")
    indexfile = tfrecordfile.with_suffix(".idx")

    if not indexfile.exists():
      call(["tfrecord2idx", str(tfrecordfile), str(indexfile)])

    self.input = ops.TFRecordReader(
        path=str(tfrecordfile),
        index_path=str(indexfile),
        features={
            "prev": tfrec.FixedLenFeature((), tfrec.string, ""),
            "center": tfrec.FixedLenFeature((), tfrec.string, ""),
            "next": tfrec.FixedLenFeature((), tfrec.string, "")
        },
        random_shuffle=shuffle,
        prefetch_queue_depth=3,
        initial_fill=1024,
        read_ahead=True)
    self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

    image_size = np.array(image_size).astype(np.float32).tolist()
    self.resize = ops.Resize(device="gpu", size=image_size)

    self.hsv = ops.Hsv(device="gpu", dtype=types.UINT8)
    self.hue_rng = ops.Uniform(range=[-0.2, 0.2])
    self.sat_rng = ops.Uniform(range=[0.8, 1.2])
    self.val_rng = ops.Uniform(range=[0.8, 1.2])

    self.cmnp = ops.CropMirrorNormalize(
        device="gpu", dtype=types.FLOAT, mean=3 * [0.], std=3 * [255.])
    self.coin = ops.CoinFlip()

  def define_graph(self):
    inputs = self.input(name="tfreader")
    images = inputs["prev"], inputs["center"], inputs["next"]

    images = [self.decode(x) for x in images]
    images = [self.resize(x) for x in images]

    h, s, v = self.hue_rng(), self.sat_rng(), self.val_rng()
    images = [self.hsv(x, hue=h, saturation=s, value=v) for x in images]

    flip = self.coin()
    images = [self.cmnp(x, mirror=flip) for x in images]

    return images
