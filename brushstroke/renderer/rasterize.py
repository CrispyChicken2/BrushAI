"""Rasterization of stroke footprints: warp a brush mask with an affine transform.

Follows Paint Transformer's ``param2stroke`` (Liu et al., ICCV 2021): ``F.affine_grid`` +
``F.grid_sample`` (bilinear, zero padding, ``align_corners=False``) with the same warp matrix.
Derivation (output pixel -> brush coordinates), with normalized output coords ``X, Y`` in [-1, 1]
(x right, y down), stroke centre ``(cx, cy) = (2x - 1, 2y - 1)`` and angle ``t = pi * theta``::

    dx = (X - cx) * W / 2,  dy = (Y - cy) * H / 2          (offsets in pixels)
    u  =  cos(t) dx + sin(t) dy,   v = -sin(t) dx + cos(t) dy   (stroke-local pixels)
    brush_x = u / (w W / 2),       brush_y = v / (h H / 2)      (in [-1, 1] inside the box)

so the box is ``w W`` pixels long (along the rotated local x axis) and ``h H`` pixels thick, and
rotation is isotropic in pixel space even on non-square canvases.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from brushstroke.renderer.brushes import BrushBank
from brushstroke.renderer.params import THETA, A, H, W, X, Y, canonicalize, check_params

EPS_SIZE = 1e-6
SAMPLERS = ("grid_sample", "gather")


def sample_bilinear_gather(masks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling equal to ``F.grid_sample(padding_mode="zeros", align_corners=False)``.

    Built from ``floor`` / ``gather`` only, so its backward w.r.t. ``grid`` has no atomic adds:
    unlike ``grid_sample``'s CUDA backward it works under
    ``torch.use_deterministic_algorithms(True)``. Costs more memory (four gathers).

    Args:
        masks: ``(M, 1, h, w)``.
        grid: ``(M, H, W, 2)`` normalized coordinates in [-1, 1] (x first).

    Returns:
        ``(M, 1, H, W)``.
    """
    m, _, h, w = masks.shape
    ix = ((grid[..., 0] + 1.0) * w - 1.0) * 0.5
    iy = ((grid[..., 1] + 1.0) * h - 1.0) * 0.5
    x0, y0 = torch.floor(ix), torch.floor(iy)
    fx, fy = ix - x0, iy - y0
    flat = masks.reshape(m, h * w)
    out = torch.zeros_like(ix)
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).reshape(m, -1)
            vals = torch.gather(flat, 1, idx).reshape_as(ix)
            out = out + vals * valid.to(vals.dtype) * wx * wy
    return out[:, None]


def default_brush_index(
    batch: int, num_strokes: int, num_brushes: int, seed: int = 0
) -> torch.Tensor:
    """Deterministic random brush variant per stroke: long tensor ``(B, N)`` in ``[0, K)`` (CPU).

    The draw is prefix-stable in ``N``: the first ``k`` columns do not depend on ``N``, so rendering
    the first ``k`` strokes alone uses the same brushes as the full painting.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_brushes, (num_strokes, batch), generator=gen).T.contiguous()


def brush_masks(brush: torch.Tensor | BrushBank) -> torch.Tensor:
    masks = brush.masks if isinstance(brush, BrushBank) else brush
    if masks.ndim == 3:
        masks = masks[None]
    if masks.ndim != 4 or masks.shape[1] != 1:
        raise ValueError(f"expected brush masks (K, 1, h, w), got {tuple(masks.shape)}")
    return masks


def stroke_affine(params: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Affine matrices ``(M, 2, 3)`` mapping output grid coords to brush coords (see module doc).

    Args:
        params: ``(M, 9)`` canonicalized stroke parameters (``w >= h`` is *not* required here).
    """
    x0, y0, w, h, theta = (params[:, i] for i in (X, Y, W, H, THETA))
    w = w.clamp_min(EPS_SIZE)
    h = h.clamp_min(EPS_SIZE)
    t = math.pi * theta
    cos_t, sin_t = torch.cos(t), torch.sin(t)
    a00 = cos_t / w
    a01 = sin_t * height / (width * w)
    a02 = (1 - 2 * x0) * cos_t / w + (1 - 2 * y0) * sin_t * height / (width * w)
    a10 = -sin_t * width / (height * h)
    a11 = cos_t / h
    a12 = (1 - 2 * y0) * cos_t / h - (1 - 2 * x0) * sin_t * width / (height * h)
    return torch.stack([a00, a01, a02, a10, a11, a12], dim=-1).view(-1, 2, 3)


def rasterize_flat(
    params: torch.Tensor,
    masks: torch.Tensor,
    height: int,
    width: int,
    sampler: str = "grid_sample",
) -> torch.Tensor:
    """Brush coverage times opacity for a flat list of strokes.

    Args:
        params: ``(M, 9)`` canonicalized parameters.
        masks: ``(M, 1, h, w)`` brush mask of each stroke (same dtype as ``params``).
        sampler: ``"grid_sample"`` (fast; its CUDA backward is not allowed under deterministic
            mode, although it was bitwise reproducible in our tests) or ``"gather"``
            (:func:`sample_bilinear_gather`, deterministic backward).

    Returns:
        ``(M, 1, H, W)``.
    """
    grid = F.affine_grid(
        stroke_affine(params, height, width),
        [params.shape[0], 1, height, width],
        align_corners=False,
    )
    if sampler == "gather":
        cover = sample_bilinear_gather(masks, grid)
    elif sampler == "grid_sample":
        cover = F.grid_sample(
            masks, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
    else:
        raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")
    return cover * params[:, A].view(-1, 1, 1, 1)


def render_alphas(
    params: torch.Tensor,
    brush: torch.Tensor | BrushBank,
    height: int,
    width: int,
    brush_index: torch.Tensor | None = None,
    brush_seed: int = 0,
    sampler: str = "grid_sample",
) -> torch.Tensor:
    """Per-stroke alpha maps (brush coverage x opacity), before compositing.

    Strokes are canonicalized first (``w W >= h H`` in pixels) so the texture follows the long side.

    Args:
        params: ``(B, N, 9)`` stroke parameters (see :mod:`brushstroke.renderer.params`).
        brush: ``(K, 1, h, w)`` masks or a :class:`BrushBank` (main axis = local x).
        height, width: canvas size in pixels.
        brush_index: optional ``(B, N)`` long tensor selecting the variant of each stroke;
            default is a deterministic random choice from ``brush_seed``.
        sampler: ``"grid_sample"`` or ``"gather"`` (see :func:`rasterize_flat`).

    Returns:
        ``(B, N, 1, H, W)`` in [0, 1], same dtype/device as ``params``.
    """
    check_params(params)
    b, n, _ = params.shape
    masks = brush_masks(brush).to(device=params.device, dtype=params.dtype)
    if brush_index is None:
        brush_index = default_brush_index(b, n, masks.shape[0], brush_seed)
    flat = canonicalize(params, height, width).reshape(b * n, -1)
    sel = masks[brush_index.to(params.device).reshape(-1)]
    return rasterize_flat(flat, sel, height, width, sampler).reshape(b, n, 1, height, width)
