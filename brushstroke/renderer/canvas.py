"""Procedural canvas / paper texture.

The texture is a height-like brightness field in ``[0, 1]`` (0.5 = neutral) with a plain-weave look:
alternating warp/weft thread cells (fine periodic structure at ``weave_period`` pixels), a slight
thread irregularity (smooth coordinate jitter) and low-frequency noise (uneven priming).
It is deterministic for a given seed and used by the animation (T-006) to shade the final image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brushstroke.renderer.brushes import gaussian_blur_1d


@dataclass
class CanvasConfig:
    """``canvas:`` config section."""

    enabled: bool = True
    seed: int = 0
    weave_period: float = 6.0  # thread width in pixels (fine periodic structure at 1/period)
    strength: float = 0.06  # shading depth used by apply_canvas (fraction of image brightness)
    low_freq_amount: float = 0.2  # amplitude of the low-frequency noise relative to the weave
    low_freq_scale: float = 48.0  # std (pixels) of the low-frequency noise blur


def _smooth_noise(h: int, w: int, sigma: float, gen: torch.Generator) -> torch.Tensor:
    n = gaussian_blur_1d(gaussian_blur_1d(torch.randn(h, w, generator=gen), sigma, 0), sigma, 1)
    return (n - n.mean()) / (n.std() + 1e-8)


def make_canvas_texture(
    height: int,
    width: int,
    seed: int,
    strength: float = 1.0,
    weave_period: float = 6.0,
    low_freq_amount: float = 0.2,
    low_freq_scale: float = 48.0,
) -> torch.Tensor:
    """Generate a woven-canvas texture.

    Args:
        height, width: output size in pixels (>= 8).
        seed: RNG seed (bitwise reproducible).
        strength: contrast of the field around the neutral value 0.5 (1 = full range [0, 1],
            0 = flat 0.5).
        weave_period: thread width in pixels.
        low_freq_amount: weight of the low-frequency noise relative to the weave.
        low_freq_scale: Gaussian std (pixels) of the low-frequency noise.

    Returns:
        float32 ``(1, H, W)`` in [0, 1] on CPU.
    """
    if height < 8 or width < 8:
        raise ValueError(f"canvas must be at least 8x8, got {(height, width)}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if weave_period <= 1.0:
        raise ValueError(f"weave_period must be > 1 pixel, got {weave_period}")
    gen = torch.Generator().manual_seed(seed)
    ys = torch.arange(height, dtype=torch.float32)[:, None].expand(height, width)
    xs = torch.arange(width, dtype=torch.float32)[None, :].expand(height, width)
    # Smooth sub-pixel jitter of the thread positions (hand-woven irregularity).
    jx = 0.35 * _smooth_noise(height, width, weave_period * 2.0, gen)
    jy = 0.35 * _smooth_noise(height, width, weave_period * 2.0, gen)
    x, y = xs + jx, ys + jy
    cell_x = torch.floor(x / weave_period)
    cell_y = torch.floor(y / weave_period)
    warp_on_top = ((cell_x + cell_y) % 2) == 0
    # Thread profile: round bump across the thread width, evaluated perpendicular to the thread.
    warp = torch.sin(math.pi * (x / weave_period - cell_x))  # warp threads run along y
    weft = torch.sin(math.pi * (y / weave_period - cell_y))  # weft threads run along x
    weave = torch.where(warp_on_top, warp, weft)  # in [0, 1]
    low = _smooth_noise(height, width, low_freq_scale, gen)
    field = (weave - 0.5) + low_freq_amount * 0.5 * low  # roughly in [-0.7, 0.7]
    field = field / (field.abs().max() + 1e-8)  # in [-1, 1]
    return (0.5 + 0.5 * strength * field).clamp(0.0, 1.0)[None]


def apply_canvas(img: torch.Tensor, tex: torch.Tensor, strength: float) -> torch.Tensor:
    """Shade an image with the canvas texture (multiplicative, neutral where ``tex == 0.5``).

    Args:
        img: ``(3, H, W)`` or ``(B, 3, H, W)`` in [0, 1].
        tex: ``(1, H, W)`` from :func:`make_canvas_texture`, in [0, 1], same ``H, W`` and device.
        strength: shading depth; the brightness is multiplied by ``1 + strength * (2 * tex - 1)``.

    Returns:
        Tensor like ``img``, clamped to [0, 1].
    """
    if tex.ndim != 3 or tex.shape[0] != 1 or img.shape[-2:] != tex.shape[-2:]:
        raise ValueError(f"texture {tuple(tex.shape)} incompatible with image {tuple(img.shape)}")
    shade = 1.0 + strength * (2.0 * tex - 1.0)
    return (img * shade).clamp(0.0, 1.0)
