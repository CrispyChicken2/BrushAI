"""``L_pcpt``: the VGG perceptual (feature reconstruction) loss of Johnson et al. (ECCV 2016).

The Paint Transformer paper (Liu et al., ICCV 2021) reports ``L_pcpt`` with this loss without
stating the layer or the normalization. Pinned definition used by the T-016 reproduction (see
``docs/experiments/pt_reproduction.md``): activations of ``relu3_3`` of the torchvision VGG16
(ImageNet weights, the backbone of LPIPS-VGG), images in [0, 1] normalized with the ImageNet mean
and standard deviation, squared error averaged over channels, rows and columns (Johnson et al. use
``relu3_3`` for their feature loss).
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torchvision

LAYER = "relu3_3"
_RELU3_3 = 16  # number of ``vgg16.features`` modules up to and including relu3_3
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


@lru_cache(maxsize=2)
def _features(device: str) -> torch.nn.Module:
    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    net: torch.nn.Module = vgg.features[:_RELU3_3].eval().requires_grad_(False)
    return net.to(device)


def vgg_features(img: torch.Tensor) -> torch.Tensor:
    """``relu3_3`` activations of ``(B, 3, H, W)`` images in [0, 1]: ``(B, 256, H / 4, W / 4)``."""
    mean = torch.tensor(_MEAN, device=img.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=img.device).view(1, 3, 1, 1)
    with torch.no_grad():
        out: torch.Tensor = _features(str(img.device))((img - mean) / std)
        return out


def pcpt_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    """``L_pcpt`` between two ``(3, H, W)`` (or ``(B, 3, H, W)``) images in [0, 1]: mean squared
    difference of their ``relu3_3`` features."""
    if pred.shape != target.shape:
        raise ValueError(f"shapes differ: {tuple(pred.shape)} and {tuple(target.shape)}")
    if pred.ndim == 3:
        pred, target = pred[None], target[None]
    return float((vgg_features(pred) - vgg_features(target)).square().mean())
