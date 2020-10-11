# Added by Adrian Köring

from pathlib import Path
from functools import lru_cache

import torch
from torch.utils import data
from torchvision import transforms

from PIL import Image
from skimage.transform import resize

from kitti_utils import generate_depth_map


class KittiDataset(data.Dataset):

  def __init__(self,
               split,
               train=False,
               transform=None,
               data_dir="kitti_data",
               image_extension="png"):
    super().__init__()

    self.data_dir = Path(data_dir)
    mode = "train" if train else "val"
    filesfile = Path("splits") / split / f"{mode}_files.txt"
    self.files = filesfile.read_text().splitlines()
    self.img_ext = image_extension

    self.transform = transform
    self.side_map = {"r": 2, "l": 3}

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    images = self.load_sequence(*self.files[index].split())
    images = self.transform(images) if self.transform else images
    return images

  def load_sequence(self, folder, index, side):
    images = [self.load_image(folder, int(index) + o, side) for o in [-1, 0, 1]]
    return torch.stack(images, dim=0)

  @lru_cache(2048)
  def load_image(self, folder, frame_index, side):
    fname = f"{frame_index:010d}.{self.img_ext}"
    directoy = self.data_dir / folder / f"image_0{self.side_map[side]}/data"
    return transforms.functional.to_tensor(Image.open(directoy / fname))


class KittiTestset(data.Dataset):

  def __init__(self,
               transform=None,
               data_dir="kitti_data",
               image_extension="png"):
    super().__init__()

    self.data_dir = Path(data_dir)
    filesfile = Path("splits") / "eigen/test_files.txt"
    self.files = filesfile.read_text().splitlines()
    self.img_ext = image_extension

    self.transform = transform
    self.side_map = {"r": 2, "l": 3}

    self.target_shape = (375, 1242)

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    folder, frame_index, side = self.files[index].split()

    image = self.load_image(folder, frame_index, side)
    depth = self.load_depth(folder, frame_index, side)

    if self.transform:
      image = self.transform(image)

    return image, depth

  def load_image(self, folder, frame_index, side):
    fname = f"{int(frame_index):010d}.{self.img_ext}"
    directoy = self.data_dir / folder / f"image_0{self.side_map[side]}/data"
    try:
      return transforms.functional.to_tensor(Image.open(directoy / fname))
    except Exception as e:
      print(e)
      raise e

  def load_depth(self, folder, frame_index, side):
    calib_dir = self.data_dir / Path(folder).parent

    velo_dir = self.data_dir / folder / "velodyne_points/data"
    velo_fname = velo_dir / f"{int(frame_index):010d}.bin"

    try:
      depth = generate_depth_map(calib_dir, velo_fname, cam=self.side_map[side])
    except Exception as e:
      print(e)
      raise e

    depth = resize(
        depth, self.target_shape, order=0, mode='constant', preserve_range=True)

    # import numpy as np
    # Image.fromarray(depth.astype(np.uint8)).save("gtdepth.png")

    return torch.from_numpy(depth).unsqueeze(0).to(torch.float32)
