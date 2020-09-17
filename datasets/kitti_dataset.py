# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
from PIL import Image

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
  """ Superclass for different types of KITTI dataset loaders
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size
    # (its later scaled by image resolution)
    # yapf: disable
    self.K = np.array([[0.58,    0, 0.5, 0],
                       [   0, 1.92, 0.5, 0],
                       [   0,    0,   1, 0],
                       [   0,    0,   0, 1]], dtype=np.float32)
    # yapf: enable

    self.full_res_shape = (1242, 375)
    self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

  def check_depth(self):
    line = self.filenames[0].split()
    scene_name = line[0]
    frame_index = int(line[1])

    velo_filename = os.path.join(
        self.data_path, scene_name,
        "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    return os.path.isfile(velo_filename)

  def get_color(self, folder, frame_index, side, do_flip):
    color = self.loader(self.get_image_path(folder, frame_index, side))

    if do_flip:
      color = color.transpose(Image.FLIP_LEFT_RIGHT)

    return color


class KITTIRAWDataset(KITTIDataset):
  """ KITTI dataset which loads the original velodyne depth maps for ground truth
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_image_path(self, folder, frame_index, side):
    fname = f"{frame_index:010d}{self.img_ext}"
    dir = self.data_path / folder / f"image_0{self.side_map[side]}/data"
    return dir / fname

  def get_depth(self, folder, frame_index, side, do_flip):
    calib_path = os.path.join(self.data_path, folder.split("/")[0])

    velo_dir = self.data_path / folder / "velodyne_points/data"
    velo_filename = velo_dir / f"{int(frame_index):010d}.bin"

    depth_gt = generate_depth_map(calib_path, velo_filename,
                                  self.side_map[side])
    depth_gt = skimage.transform.resize(
        depth_gt,
        self.full_res_shape[::-1],
        order=0,
        preserve_range=True,
        mode='constant')

    if do_flip:
      depth_gt = np.fliplr(depth_gt)

    return depth_gt


class KITTIOdomDataset(KITTIDataset):
  """ KITTI dataset for odometry training and testing
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_image_path(self, folder, frame_index, side):
    fname = f"{frame_index:06d}{self.img_ext}"
    dir = self.data_path / f"sequences/{int(folder):02d}" / f"image_{self.side_map[side]}"
    return dir / fname


class KITTIDepthDataset(KITTIDataset):
  """ KITTI dataset which uses the updated ground truth depth maps
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_image_path(self, folder, frame_index, side):
    fname = f"{frame_index:010d}{self.img_ext}"
    dir = self.data_path / folder / f"image_0{self.side_map[side]}/data"
    return dir / fname

  def get_depth(self, folder, frame_index, side, do_flip):
    fname = f"{frame_index:010d}.png"
    dir = self.data_path / folder / f"proj_depth/groundtruth/image_0{self.side_map[side]}"

    depth_path = dir / fname

    depth_gt = Image.open(depth_path)
    depth_gt = depth_gt.resize(self.full_res_shape, Image.NEAREST)
    depth_gt = np.array(depth_gt).astype(np.float32) / 256

    if do_flip:
      depth_gt = np.fliplr(depth_gt)

    return depth_gt
