import pytorch_lightning as pl

import torch
from torch import nn
from torch import autograd
from torch.nn import functional as F
from torchvision import transforms

import kornia
from kornia import geometry, enhance, augmentation  #, losses

import matplotlib.pyplot as plt

import losses


class NoCrop(nn.Module):

  def __init__(self, image_size):
    super().__init__()

    self.image_size = image_size
    H, W = image_size

    self.K = torch.tensor([[0.58, 0, 0.5], [0, 1.92, 0.5], [0, 0, 1.]])
    self.K[0] *= W
    self.K[1] *= H

  def stacked_K(self, B):
    return torch.stack(B * [self.K])

  def forward(self, Iprev, Ic, Inext):
    B, C, H, W = Ic.shape
    return Iprev, Ic, Inext, self.stacked_K(B).to(Ic.device)


class RandomCrop(NoCrop):

  def __init__(self, image_size, crop_size):
    super().__init__(image_size)
    self.crop_size = crop_size

  def apply_crop(self, Ks, params):
    src = params["src"]

    LT = src[:, 0]  # left, top corner
    Ks[..., 0, -1] = Ks[..., 0, -1] - LT[:, 0]
    Ks[..., 1, -1] = Ks[..., 1, -1] - LT[:, 1]

    return Ks

  def forward(self, Iprev, Ic, Inext):

    if not self.training:
      # Random crop only augments training
      return super().forward(Iprev, Ic, Inext)

    B, C, H, W = Ic.shape

    params = augmentation.random_generator.random_crop_generator(
        batch_size=B,
        input_size=self.image_size,
        size=self.crop_size,
    )

    Ic = augmentation.functional.apply_crop(Ic, params)
    Iprev = augmentation.functional.apply_crop(Iprev, params)
    Inext = augmentation.functional.apply_crop(Inext, params)

    Ks = self.stacked_K(B)
    Ks = self.apply_crop(Ks, params)

    return Iprev, Ic, Inext, Ks.to(Ic.device)


class TrainingModule(pl.LightningModule):

  def __init__(self, image_size, crop_size, *args, **kwargs):
    super().__init__()
    self.save_hyperparameters()
    self.ssim = losses.SSIM(window_size=7)

    if crop_size:
      self.crop = RandomCrop(image_size, crop_size)
    else:
      self.crop = NoCrop(image_size)

  def configure_optimizers(self):
    opt = torch.optim.Adam(self.parameters(), lr=0.0001)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 15, 0.1)
    return [opt], [schedule]

  def unpack(self, x):
    """ Unpack training example as produced by DALIGenericIterator """
    x = x[0]
    return x["prev"], x["center"], x["next"]

  def upsample_like(self, idepth, image):
    B, C, *size = image.shape
    return F.interpolate(idepth, size=size, mode="nearest")

  def downsample(self, image, idepth):
    B, C, H, W = idepth.shape
    return F.interpolate(image, size=[H, W], mode="area")

  def denormalize(self, idepth, a=10, b=0.01):
    """ Scale predicted inverse depth into depth for projection """
    return 1 / (a * idepth + b)

  def photometric_error(self, img0, img1, alpha=0.85):
    """ Compute photometric error (ssim + l1) between images: returns BxHxW """
    ssim = self.ssim(img0, img1).mean(dim=1)
    l1 = F.l1_loss(img0, img1, reduction="none").mean(dim=1)
    return alpha * ssim + (1 - alpha) * l1

  def smoothness_loss(self, idepth, image, lam=1e-3):
    """ Compute image-edge-aware smoothness of inverse depth: returns BxHxW """
    norm_idepth = idepth / F.adaptive_avg_pool2d(idepth, (1, 1))
    image = self.downsample(image, idepth)
    Lsmooth = losses.inverse_depth_smoothness_loss(norm_idepth, image)

    (B, C, Hd, Wd), (_, _, Hi, Wi) = idepth.shape, image.shape
    scale = Hd // Hi

    return lam * Lsmooth / 2**scale

  def loss_fn(self, Rc, Ic, Ia, idepth):
    """ Compute reconstruction, baseline and smoothness loss:
        Ic, Rc: center image, reconstructed center image
        Ia: adjacent image (previous or next)
        idepth: inverse depth in [0, 1] as predicted by model

        returns:
        BxHxW: reconstruction loss between observed image and reconstruction
        BxHxW: baseline loss between observed center and adjacent image
        scalar: smoothness loss: depth gradients weighted by image edges
    """
    Lrecon = self.photometric_error(Rc, Ic)
    Lbase = self.photometric_error(Ia, Ic)
    Lsmooth = self.smoothness_loss(idepth, Ic)
    return Lrecon, Lbase, Lsmooth

  def reproject(self, depth, image, T, K):
    """ Reconstruct center image from given adjacent 'image', 'depth' and
        transformation 'T' from center to adjacent camera coordinates.
    """
    return geometry.warp_frame_depth(image, depth, T, K, padding_mode="border")

  def step(self, images):
    Iprev, Ic, Inext = self.unpack(images)

    Iprev, Ic, Inext, K = self.crop(Iprev, Ic, Inext)
    images = torch.stack([Iprev, Ic, Inext], dim=1)

    idepths, Tprev, Tnext = self(Iprev, Ic, Inext)

    scale_losses = []
    for idepth in idepths:

      recons, recon_losses, smooth_losses = [], [], []
      # iterate over adjacent images and transform
      for Ia, Ta in [[Iprev, Tprev], [Inext, Tnext]]:

        depth = self.upsample_like(self.denormalize(idepth), Ic)
        Rc = self.reproject(depth, Ia, Ta, K)
        Lrecon, Lbase, Lsmooth = self.loss_fn(Rc, Ic, Ia, idepth)
        recons.append(Rc)
        recon_losses.extend([Lrecon, Lbase])
        smooth_losses.append(Lsmooth)

      # between both reconstruction and baseline losses take the minimum. This
      #  implements equations 4 and 5 from the paper. Gradient will be backproped
      #  into the network, if prediction creates the lowest loss.
      # Otherwise, if baseline is lower than the model prediction, the gradient
      #  will 'vanish' between the oberseved images and not update any parameters.
      # This implements the masking heuristic, because model parameters are
      #  unaffected if the baseline photometric_error between center and adjacent
      #  image is lower than the reconstructed error.
      Lrecon = torch.stack(recon_losses, dim=1).min(dim=1)[0].mean()
      Lsmooth = torch.stack(smooth_losses).mean()

      scale_losses.append(Lrecon + Lsmooth)

    loss = torch.stack(scale_losses).mean()

    if self.log:
      Rprev, Rnext = recons
      self.log_images(idepth, Ic)
      self.log_sequence(torch.stack([Rprev, Ic, Rnext], dim=1), "recons")

      if self.log == "train":  # validation aggregates metrics
        self.logger.log_metrics({"loss/train": loss}, step=self.global_step)

    return loss

  def training_step(self, images, batch_idx):
    self.log = "train" if batch_idx % 420 == 0 else None
    loss = self.step(images)
    return {"loss": loss}

  def validation_step(self, images, batch_idx):
    self.log = "val" if batch_idx == 0 else None
    loss = self.step(images)
    return {"loss": loss}

  def validation_epoch_end(self, outputs):
    loss = torch.stack([o["loss"] for o in outputs]).mean()
    self.logger.log_metrics({"loss/val": loss}, self.global_step)
    return {"val_loss": loss}

  def log_images(self, idepths, images):
    *_, tensorboard_logger = self.logger.experiment

    for n in range(min(4, len(images))):
      idepth, image = idepths[n], images[n]

      key = f"{self.log}/depth_{n}"

      fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6))

      for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])

      # plot depth indivdually to scale things properly
      ax0.set_title("Inverse Depth")
      aximg = ax0.imshow(kornia.tensor_to_image(idepth))
      plt.colorbar(aximg, ax=ax0)

      ax1.set_title("Image")
      aximg = ax1.imshow(kornia.tensor_to_image(image))
      plt.colorbar(aximg, ax=ax1)

      tensorboard_logger.add_figure(key, fig, self.global_step)

  def log_sequence(self, batch, name="images"):
    *_, tensorboard_logger = self.logger.experiment

    for n in range(min(4, len(batch))):
      seq = batch[n].unsqueeze(0)
      key = f"{self.log}/{name}_{n}"
      tensorboard_logger.add_video(key, seq, self.global_step)


class TestModule(TrainingModule):

  def __init__(self,
               target_size=(375, 1242),
               min_depth=1e-3,
               max_depth=80.,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)

    self.target_size = target_size
    self.metric_names = [
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"
    ]
    self.min_depth = min_depth
    self.max_depth = max_depth

  def test_step(self, batch, batch_idx):
    image, gt_depth = batch

    # discard low-res depth estimates
    *_, pred_idepth = self.depth_model(image)
    assert pred_idepth.shape[-2:] == image.shape[-2:]

    pred_depth, gt_depth = self.prepare_depth(pred_idepth, gt_depth)
    metrics = self.compute_metrics(pred_depth, gt_depth)
    return dict(zip(self.metric_names, metrics))

  def test_epoch_end(self, outputs):
    """ outputs is list of batches, batch is list of dicts:
        outputs = [batch1 = [metric_b1_1 = {..}, metric_b1_2 = {..}, ..],
                   batch2 = [metric_b2_1 = {..}, metric_b2_2 = {..}, ..], ..]
    """
    mean = lambda k: torch.stack([o[k] for o in outputs]).mean()
    metrics = {"test/" + k: mean(k) for k in self.metric_names}
    self.logger.log_metrics(metrics, step=self.global_step)

    return metrics

  def eigen_mask(self, gt_depth):
    """ Return mask comparing prediction only at valid annotation pixels """

    B, C, H, W = gt_depth.shape

    # include valid gt pixel annotations
    valid = torch.logical_and(self.min_depth < gt_depth,
                              gt_depth < self.max_depth)

    crop = torch.zeros_like(gt_depth)
    top, bottom = int(0.40810811 * H), int(0.99189189 * H)
    left, right = int(0.03594771 * W), int(0.96405229 * W)
    crop[..., top:bottom, left:right] = 1

    return torch.logical_and(valid, crop)

  def prepare_depth(self, pred_idepth, gt_depth):
    """ Prepare predicted depth for evaluation with gt_depth """

    pred_depth = self.denormalize(pred_idepth)
    pred_depth = transforms.functional.resize(pred_depth, self.target_size)
    pred_depth.clamp_(self.min_depth, self.max_depth)

    mask = self.eigen_mask(gt_depth)

    pred_depth, gt_depth = pred_depth[mask], gt_depth[mask]

    assert len(pred_depth) and len(gt_depth)

    pred_depth *= torch.median(gt_depth) / torch.median(pred_depth)
    return pred_depth, gt_depth

  def compute_metrics(self, pred, gt):
    """ Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25**2).float().mean()
    a3 = (thresh < 1.25**3).float().mean()

    rmse = (gt - pred)**2
    rmse = rmse.mean().sqrt()

    rmse_log = (gt.log() - pred.log())**2
    rmse_log = rmse_log.mean().sqrt()

    abs_rel = ((gt - pred).abs() / gt).mean()

    sq_rel = ((gt - pred).pow(2) / gt).mean()

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
