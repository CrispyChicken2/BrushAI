"""Initialization of new strokes from the residual error (SNP-style error-map sampling).

Stylized Neural Painting (Zou et al., CVPR 2021; ``painter.py::stroke_sampler``) re-samples a stroke
position from the blurred absolute difference between target and canvas, raised to a power, and
takes the colour from the target. We do the same per patch: for every stroke assigned to a patch of
the ``g x g`` grid, the centre is sampled inside that patch with probability proportional to
``blur(|target - canvas|) ** power`` (uniform if the patch is already perfect), and the colour is
read from the (lightly blurred) target at that position.
"""

from __future__ import annotations

import math

import torch

from brushstroke.baselines.optim.config import OptimConfig
from brushstroke.renderer.brushes import gaussian_blur_1d


def blur2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of ``(H, W)`` or ``(C, H, W)``; ``sigma`` in pixels."""
    if x.ndim == 3:
        return torch.stack([blur2d(c, sigma) for c in x])
    return gaussian_blur_1d(gaussian_blur_1d(x, sigma, 0), sigma, 1)


def sample_strokes(
    target: torch.Tensor,
    canvas: torch.Tensor,
    grid: int,
    patch_ids: torch.Tensor,
    cfg: OptimConfig,
    gen: torch.Generator,
) -> torch.Tensor:
    """Draw fresh stroke parameters for strokes assigned to patches of a ``grid x grid`` division.

    Args:
        target: ``(3, H, W)`` target image in [0, 1].
        canvas: ``(3, H, W)`` current canvas (strokes drawn so far).
        grid: patches per side (``g``); patch ``p`` is row ``p // g``, column ``p % g``.
        patch_ids: ``(M,)`` long, the patch of each stroke to draw.
        cfg: initialization settings (``init_*``).
        gen: CPU generator (all randomness comes from it, so runs are reproducible).

    Returns:
        ``(M, 9)`` float32 CPU parameters in the renderer convention.
    """
    _, h, w = target.shape
    tgt, cur = target.detach().cpu(), canvas.detach().cpu()
    patch_px = min(h, w) / grid
    residual = blur2d((tgt - cur).abs().sum(0), max(0.5, cfg.init_blur_frac * patch_px))
    weights = residual.clamp_min(0).pow(cfg.init_error_power)
    colour_src = blur2d(tgt, max(0.5, patch_px / 8.0))
    m = patch_ids.numel()
    out = torch.zeros(m, 9)
    for p in torch.unique(patch_ids).tolist():
        rows = torch.nonzero(patch_ids == p).flatten()
        py, px = divmod(int(p), grid)
        y0, y1 = round(py * h / grid), round((py + 1) * h / grid)
        x0, x1 = round(px * w / grid), round((px + 1) * w / grid)
        wt = weights[y0:y1, x0:x1].reshape(-1)
        if not bool(wt.sum() > 1e-12):
            wt = torch.ones_like(wt)
        pick = torch.multinomial(wt, rows.numel(), replacement=True, generator=gen)
        ys, xs = y0 + pick // (x1 - x0), x0 + pick % (x1 - x0)
        out[rows, 0] = (xs.float() + 0.5) / w
        out[rows, 1] = (ys.float() + 0.5) / h
        out[rows, 5:8] = colour_src[:, ys, xs].T
    length = cfg.init_size_scale * (0.6 + 0.6 * torch.rand(m, generator=gen)) / grid
    out[:, 2] = length.clamp(max=1.0)
    lo, hi = cfg.init_aspect
    out[:, 3] = (out[:, 2] * (lo + (hi - lo) * torch.rand(m, generator=gen))).clamp(min=1e-3)
    out[:, 4] = torch.rand(m, generator=gen)
    out[:, 8] = cfg.init_opacity
    return out


def dead_strokes(params: torch.Tensor, cfg: OptimConfig) -> torch.Tensor:
    """Boolean ``(M,)`` mask of strokes that no longer contribute (near-zero opacity or area)."""
    area = params[:, 2] * params[:, 3]
    return (params[:, 8] < cfg.dead_opacity) | (area < cfg.dead_area) | ~torch.isfinite(area)


def cosine_lr(base: float, final_frac: float, step: int, total: int) -> float:
    """Cosine decay from ``base`` to ``base * final_frac`` over ``total`` steps."""
    t = step / max(1, total - 1)
    return base * (final_frac + (1.0 - final_frac) * 0.5 * (1.0 + math.cos(math.pi * t)))
