"""Final-image metrics shared by the demos and experiments: L1, PSNR, SSIM, LPIPS-VGG / LPIPS-Alex.

All images are ``(3, H, W)`` float tensors in [0, 1] on the same device. LPIPS-Alex is the held-out
metric (never used in a training loss, see ``docs/experiments/optim_sweep.md``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from brushstroke.baselines.optim.losses import lpips_distance


def _gaussian_window(
    size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean structural similarity (Wang et al., IEEE TIP 2004) over the colour channels.

    Gaussian window (11 x 11, sigma 1.5), no padding (valid region), ``C1 = 0.01^2``,
    ``C2 = 0.03^2`` (dynamic range 1). Returns a scalar in (-1, 1]; 1 for identical images.
    """
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(
            f"expected two images of shape (3, H, W), got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    w = _gaussian_window(window, sigma, x.device, x.dtype)[None, None].expand(x.shape[0], 1, -1, -1)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t[None], w, groups=t.shape[0])[0]

    mx, my = blur(x), blur(y)
    vxx, vyy, vxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vxx + vyy + c2))
    return s.mean()


def image_metrics(final: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """L1, PSNR (dB), SSIM, LPIPS-VGG and LPIPS-Alex between ``final`` and ``target``."""
    with torch.no_grad():
        mse = float((final - target).square().mean())
        return {
            "l1": float((final - target).abs().mean()),
            "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "ssim": float(ssim(final, target)),
            "lpips_vgg": float(lpips_distance(final[None], target[None], "vgg")),
            "lpips_alex": float(lpips_distance(final[None], target[None], "alex")),
        }
