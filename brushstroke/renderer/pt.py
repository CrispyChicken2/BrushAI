"""Paint Transformer compatible rendering mode (T-011).

Reproduces the stroke renderer of the official Paint Transformer code (Liu et al., ICCV 2021;
https://github.com/Huage001/PaintTransformer at commit 7960b8e) as specified in
``docs/design/paint_transformer.md`` section 1:

* two brush images (vertical = index 0, horizontal = index 1) chosen by ``h > w`` (no
  canonicalization);
* ``pt_infer`` (``inference/inference.py:param2stroke``): the brush is nearest-resized to the
  render size, warped,
  alpha = ``warped > 0``; ``pt_train`` (``train/models/painter_model.py:param2stroke``): alpha =
  ``brush > 0`` is
  binarized *before* warping, the 394 px images are warped directly;
* foreground = 3 x 3 dilation of ``brush_gray * colour``, alpha = 3 x 3 erosion of the (binary or
  warped) alpha;
* compositing ``canvas = fg * alpha * decision + canvas * (1 - alpha * decision)`` on a **black**
  canvas;
* no opacity: stroke ``k`` is painted iff its decision is 1.

The brush images are Apache-2.0 assets of the official repository, fetched (not committed) by
``python -m scripts.fetch_pt_brushes``; loading them without the files raises an error (no
fallback).
Parameter layout of a PT stroke: ``(x, y, w, h, theta, r, g, b)`` with the same normalization as
:mod:`brushstroke.renderer.params` (``theta in [0, 1]`` means ``theta * pi``).
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PT_BRUSH_DIR = Path("data/raw/pt_brush")
PT_MODES = ("pt_infer", "pt_train")
NUM_PT_PARAMS = 8
_FILES = {
    "inference": ("inference_brush_large_vertical.png", "inference_brush_large_horizontal.png"),
    "train": ("train_brush_large_vertical.png", "train_brush_large_horizontal.png"),
}


def load_pt_brushes(
    variant: str = "inference",
    directory: str | Path = PT_BRUSH_DIR,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load the two official brush images as ``(2, 1, 394, 394)`` float in [0, 1] (0 = vertical, 1
    = horizontal).

    Args:
        variant: ``"inference"`` (images of ``inference/brush``) or ``"train"`` (images of
        ``train/brush``);
            the official repository ships different files for the two stages, both converted with
            ``convert('L')``.
        directory: folder written by ``python -m scripts.fetch_pt_brushes``.

    Raises:
        FileNotFoundError: the assets are missing (never silently replaced by another brush).
    """
    if variant not in _FILES:
        raise ValueError(f"variant must be one of {sorted(_FILES)}, got {variant!r}")
    folder = Path(directory)
    images = []
    for name in _FILES[variant]:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(
                f"Paint Transformer brush image {path} not found. Fetch the official assets with "
                "`python -m scripts.fetch_pt_brushes` (add --insecure behind a TLS proxy)."
            )
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        images.append(torch.from_numpy(arr)[None])
    return torch.stack(images).to(device)


@lru_cache(maxsize=8)
def cached_pt_brushes(variant: str, directory: str, device: str) -> torch.Tensor:
    """:func:`load_pt_brushes`, memoized (do not mutate the result)."""
    return load_pt_brushes(variant, directory, device)


def dilation(x: torch.Tensor) -> torch.Tensor:
    """3 x 3 max filter (official ``morphology.dilation``, border value ``-1e9`` == ignoring the
    border)."""
    return F.max_pool2d(x, 3, stride=1, padding=1)


def erosion(x: torch.Tensor) -> torch.Tensor:
    """3 x 3 min filter (official ``morphology.erosion``, border value ``+1e9`` == ignoring the
    border)."""
    return -F.max_pool2d(-x, 3, stride=1, padding=1)


def pt_warp_grid(param: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Sampling grid ``(M, H, W, 2)`` of the official warp (same operations and order as the
    official code).

    Args:
        param: ``(M, >= 5)`` rows ``x, y, w, h, theta`` (normalized, ``theta * pi`` radians).
    """
    x0, y0, w, h, theta = (param[:, i] for i in range(5))
    pi = torch.acos(torch.tensor(-1.0, device=param.device, dtype=param.dtype))
    sin_theta = torch.sin(pi * theta)
    cos_theta = torch.cos(pi * theta)
    warp_00 = cos_theta / w
    warp_01 = sin_theta * height / (width * w)
    warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * height / (width * w)
    warp_10 = -sin_theta * width / (height * h)
    warp_11 = cos_theta / h
    warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * width / (height * h)
    warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
    warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
    warp = torch.stack([warp_0, warp_1], dim=1)
    return F.affine_grid(warp, [param.shape[0], 3, height, width], align_corners=False)


def pt_param2stroke(
    param: torch.Tensor,
    height: int,
    width: int,
    brushes: torch.Tensor,
    mode: str = "pt_infer",
    morphology: bool = True,
    brush_size: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Foreground colours and alpha maps of strokes (the official ``param2stroke``).

    Args:
        param: ``(M, 8)`` strokes ``x, y, w, h, theta, r, g, b``.
        height, width: output size in pixels.
        brushes: ``(2, 1, h_b, w_b)`` from :func:`load_pt_brushes` (use the variant that matches
        ``mode``:
            ``"inference"`` for ``pt_infer``, ``"train"`` for ``pt_train``).
        mode: ``pt_infer`` or ``pt_train`` (see the module docstring).
        morphology: apply the 3 x 3 dilation of the foreground and erosion of the alpha (the
        official code applies
            them right after ``param2stroke``).
        brush_size: ``pt_infer`` only: size ``(h, w)`` the brush images are nearest-resized to
            before the warp (default: the render size, as the official code, which renders in
            windows: the multi-scale pipeline resizes the brush to the *window* it paints in).

    Returns:
        ``(foreground, alpha)``, both ``(M, 3, H, W)``.
    """
    if mode not in PT_MODES:
        raise ValueError(f"mode must be one of {PT_MODES}, got {mode!r}")
    if param.ndim != 2 or param.shape[1] != NUM_PT_PARAMS:
        raise ValueError(f"expected params of shape (M, {NUM_PT_PARAMS}), got {tuple(param.shape)}")
    b = param.shape[0]
    w, h = param[:, 2], param[:, 3]
    colour = param[:, 5:8].unsqueeze(-1).unsqueeze(-1)  # (M, 3, 1, 1)
    index = torch.full((b,), -1, device=param.device, dtype=torch.long)
    index[h > w] = 0  # vertical image
    index[h <= w] = 1  # horizontal image
    grid = pt_warp_grid(param, height, width)
    if mode == "pt_infer":
        # nearest, as the official default
        resized = F.interpolate(brushes, brush_size or (height, width))
        warped = F.grid_sample(resized[index], grid, align_corners=False)
        alpha = (warped > 0).float().repeat(1, 3, 1, 1)
        foreground = warped.repeat(1, 3, 1, 1) * colour
    else:
        # (2, 2, h_b, w_b): gray image and its binarization, warped together
        source = torch.cat([brushes, (brushes > 0).float()], dim=1)
        both = torch.empty(b, 2, height, width, dtype=param.dtype, device=param.device)
        for k in (
            0,
            1,
        ):  # only two distinct brushes: expand (no copy) instead of gathering b copies
            sel = torch.nonzero(index == k).flatten()
            if sel.numel() > 0:
                src = source[k : k + 1].expand(sel.numel(), -1, -1, -1)
                both[sel] = F.grid_sample(src, grid[sel], align_corners=False)
        alpha = both[:, 1:2].repeat(1, 3, 1, 1)
        foreground = both[:, 0:1].repeat(1, 3, 1, 1) * colour
    if morphology:
        foreground, alpha = dilation(foreground), erosion(alpha)
    return foreground, alpha


def pt_composite(
    canvas: torch.Tensor,
    foregrounds: torch.Tensor,
    alphas: torch.Tensor,
    decisions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Paint strokes in order: ``canvas = fg * a * d + canvas * (1 - a * d)``.

    Args:
        canvas: ``(B, 3, H, W)``.
        foregrounds, alphas: ``(B, N, 3, H, W)``.
        decisions: optional ``(B, N)`` 0/1 (or bool); ``None`` = all strokes painted.
    """
    for k in range(foregrounds.shape[1]):
        a = alphas[:, k]
        if decisions is not None:
            a = a * decisions[:, k].to(a.dtype).view(-1, 1, 1, 1)
        canvas = foregrounds[:, k] * a + canvas * (1 - a)
    return canvas


def _black(batch: int, height: int, width: int, like: torch.Tensor) -> torch.Tensor:
    return torch.zeros(batch, 3, height, width, dtype=like.dtype, device=like.device)


def render_pt(
    params: torch.Tensor,
    height: int,
    width: int,
    brushes: torch.Tensor,
    mode: str = "pt_infer",
    decisions: torch.Tensor | None = None,
    canvas: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Render ``(B, N, 8)`` PT strokes in order on ``canvas`` (default black): ``(B, 3, H, W)``."""
    return render_pt_progressive(
        params, height, width, brushes, [params.shape[1]], mode, decisions, canvas, chunk_size
    )[:, 0]


def render_pt_progressive(
    params: torch.Tensor,
    height: int,
    width: int,
    brushes: torch.Tensor,
    checkpoints: Sequence[int],
    mode: str = "pt_infer",
    decisions: torch.Tensor | None = None,
    canvas: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> torch.Tensor:
    """Canvases after the first ``k`` strokes for each ``k`` in ``checkpoints``: ``(B,
    len(checkpoints), 3, H, W)``."""
    if params.ndim != 3 or params.shape[-1] != NUM_PT_PARAMS:
        raise ValueError(
            f"expected params of shape (B, N, {NUM_PT_PARAMS}), got {tuple(params.shape)}"
        )
    b, n, _ = params.shape
    if any(not 0 <= k <= n for k in checkpoints) or chunk_size < 1:
        raise ValueError(f"checkpoints must lie in [0, {n}] and chunk_size >= 1")
    current = _black(b, height, width, params) if canvas is None else canvas.clone()
    snapshots: dict[int, torch.Tensor] = {}
    done = 0
    for target in sorted(set(checkpoints)):
        while done < target:
            end = min(done + chunk_size, target)
            fg, al = pt_param2stroke(
                params[:, done:end].reshape(-1, NUM_PT_PARAMS), height, width, brushes, mode
            )
            fg = fg.view(b, end - done, 3, height, width)
            al = al.view(b, end - done, 3, height, width)
            dec = None if decisions is None else decisions[:, done:end]
            current = pt_composite(current, fg, al, dec)
            done = end
        snapshots[target] = current
    return torch.stack([snapshots[k] for k in checkpoints], dim=1)


# ----------------------------------------------------------------------------- parameter adapters


def ours_to_pt(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Our ``(..., 9)`` strokes -> PT ``(..., 8)`` strokes and a 0/1 decision.

    PT has no opacity: the ninth parameter ``a`` is interpreted as the stroke decision (``a > 0``
    painted).
    """
    if params.shape[-1] != 9:
        raise ValueError(f"expected (..., 9) params, got {tuple(params.shape)}")
    return params[..., :8], (params[..., 8] > 0).to(params.dtype)


def pt_to_ours(params: torch.Tensor, decisions: torch.Tensor | None = None) -> torch.Tensor:
    """PT ``(..., 8)`` strokes (+ optional 0/1 decisions) -> our ``(..., 9)`` strokes with ``a =
    decision`` (default 1)."""
    if params.shape[-1] != NUM_PT_PARAMS:
        raise ValueError(f"expected (..., 8) params, got {tuple(params.shape)}")
    a = (
        torch.ones_like(params[..., :1])
        if decisions is None
        else decisions.to(params.dtype).unsqueeze(-1)
    )
    return torch.cat([params, a], dim=-1)


def patch_to_canvas(
    params: torch.Tensor,
    patch_row: torch.Tensor | int,
    patch_col: torch.Tensor | int,
    canvas_size: int,
    patch_size: int = 32,
    shifted: bool = False,
) -> torch.Tensor:
    """Patch-level PT strokes -> canvas-normalized strokes (study note section 1, "Patch
    coordinates to canvas coordinates").

    The network predicts ``x, y, w, h`` in units of its ``patch_size`` px patch. For the patch
    ``(row, col)`` of a layer whose
    canvas is ``canvas_size`` px: ``x = (patch_size * col + patch_size * x_p) / canvas_size`` (``-
    patch_size / 2`` for the
    shifted final layer, whose patch grid is offset by half a patch), likewise ``y``; ``w =
    patch_size * w_p / canvas_size``,
    ``h`` likewise. ``theta`` and any colour channels are copied.

    Args:
        params: ``(..., >= 5)`` strokes in patch units.
        patch_row, patch_col: integer patch indices, broadcastable against ``params[..., 0]``.
    """
    out = params.clone()
    offset = -patch_size / 2 if shifted else 0.0
    col = torch.as_tensor(patch_col, dtype=params.dtype, device=params.device)
    row = torch.as_tensor(patch_row, dtype=params.dtype, device=params.device)
    out[..., 0] = (patch_size * col + offset + patch_size * params[..., 0]) / canvas_size
    out[..., 1] = (patch_size * row + offset + patch_size * params[..., 1]) / canvas_size
    out[..., 2] = patch_size * params[..., 2] / canvas_size
    out[..., 3] = patch_size * params[..., 3] / canvas_size
    return out
