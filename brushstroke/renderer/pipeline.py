"""Chunked, differentiable rendering of ordered strokes onto a canvas.

Strokes are processed in chunks of ``chunk_size``: for each chunk the alphas are rasterized,
composited in closed form over the running canvas, and freed, so peak memory is
``O(chunk_size * H * W)`` instead of ``O(N * H * W)``. With ``use_checkpoint`` the chunk's
activations are recomputed in the backward pass (``torch.utils.checkpoint``), so the autograd graph
keeps only one ``(B, 3, H, W)`` canvas per chunk boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from brushstroke.renderer.brushes import BrushBank
from brushstroke.renderer.compositing import composite_parallel
from brushstroke.renderer.params import B as BLUE
from brushstroke.renderer.params import R as RED
from brushstroke.renderer.params import canonicalize, check_params
from brushstroke.renderer.pt import (
    PT_MODES,
    cached_pt_brushes,
    ours_to_pt,
    render_pt_progressive,
)
from brushstroke.renderer.rasterize import (
    brush_masks,
    default_brush_index,
    rasterize_window,
    render_alphas,
    stroke_boxes,
)

_DTYPES: dict[str, torch.dtype | None] = {
    "auto": None,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


@dataclass
class RenderConfig:
    """``render:`` config section."""

    chunk_size: int = 64  # strokes composited at once (memory ~ chunk_size * H * W)
    use_checkpoint: bool = False  # recompute chunk activations in backward (slower, less memory)
    brush_seed: int = 0  # seed of the default per-stroke brush variant choice
    dtype: str = "auto"  # compositing dtype: auto (= params dtype), float32, float64, bfloat16
    sampler: str = "grid_sample"  # grid_sample | gather (gather: deterministic CUDA backward)
    min_size: float = 1e-3  # smallest w / h used by the parameter helpers
    mode: str = (
        "default"  # default: our procedural brushes | pt: Paint Transformer compatible (T-011)
    )
    pt_alpha: str = (
        "pt_infer"  # pt mode: pt_infer | pt_train (alpha/brush pipeline of the official code)
    )
    pt_brush_dir: str = (
        "data/raw/pt_brush"  # pt mode: official brush PNGs (scripts/fetch_pt_brushes.py)
    )
    bbox: bool = (
        False  # composite each chunk only inside the union bounding box of its strokes (T-027)
    )
    bbox_chunk: int = (
        0  # chunk size of the bounding-box path (0 = chunk_size); smaller = tighter boxes
    )
    bbox_max_fraction: float = (
        0.75  # a chunk whose box covers more of the canvas takes the full path
    )


def make_canvas(
    canvas: torch.Tensor | Sequence[float] | None,
    batch: int,
    height: int,
    width: int,
    like: torch.Tensor,
) -> torch.Tensor:
    """Initial canvas ``(B, 3, H, W)`` from None (white), RGB triple, ``(3,H,W)`` or batched."""
    if canvas is None:
        return torch.ones(batch, 3, height, width, dtype=like.dtype, device=like.device)
    t = torch.as_tensor(canvas, dtype=like.dtype, device=like.device)
    if t.ndim == 1 and t.shape[0] == 3:
        return t.view(1, 3, 1, 1).expand(batch, 3, height, width).clone()
    if t.ndim == 3:
        t = t[None]
    if t.shape[0] == 1:
        t = t.expand(batch, -1, -1, -1)
    if t.shape != (batch, 3, height, width):
        raise ValueError(
            f"canvas {tuple(t.shape)} incompatible with (B={batch}, 3, {height}, {width})"
        )
    return t.clone()


def _composite_chunk(
    canvas: torch.Tensor,
    chunk: torch.Tensor,
    masks: torch.Tensor,
    index: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
    sampler: str,
) -> torch.Tensor:
    alphas = render_alphas(chunk, masks, height, width, brush_index=index, sampler=sampler)
    colors = chunk[..., RED : BLUE + 1]
    out = composite_parallel(canvas.to(dtype), alphas.to(dtype), colors.to(dtype))
    return out.to(canvas.dtype)


def _composite_chunk_bbox(
    canvas: torch.Tensor,
    chunk: torch.Tensor,
    masks: torch.Tensor,
    index: torch.Tensor,
    box: tuple[int, int, int, int],
    height: int,
    width: int,
    dtype: torch.dtype,
    sampler: str,
) -> torch.Tensor:
    """:func:`_composite_chunk` restricted to the window ``box``: the strokes are rasterized and composited on the crop
    only and the result is pasted back (outside the box every alpha is exactly 0, so the canvas is unchanged there)."""
    y0, y1, x0, x1 = box
    b, n = chunk.shape[:2]
    flat = canonicalize(chunk, height, width).reshape(b * n, -1)
    sel = masks[index.reshape(-1)]
    alphas = rasterize_window(flat, sel, height, width, box, sampler).reshape(
        b, n, 1, y1 - y0, x1 - x0
    )
    colors = chunk[..., RED : BLUE + 1]
    crop = canvas[:, :, y0:y1, x0:x1].to(dtype)
    out = composite_parallel(crop, alphas.to(dtype), colors.to(dtype)).to(canvas.dtype)
    result = canvas.clone()
    result[:, :, y0:y1, x0:x1] = out
    return result


def _chunk_box(
    boxes: torch.Tensor, done: int, end: int, height: int, width: int, max_fraction: float
) -> tuple[int, int, int, int] | None | bool:
    """Union box of strokes ``done:end`` over the batch: ``None`` if every stroke is off the canvas (nothing to paint),
    ``False`` if it covers more than ``max_fraction`` of the canvas (the full path is then cheaper), else the box."""
    part = boxes[:, done:end].reshape(-1, 4)
    live = part[(part[:, 1] > part[:, 0]) & (part[:, 3] > part[:, 2])]
    if len(live) == 0:
        return None
    y0, y1 = int(live[:, 0].min()), int(live[:, 1].max())
    x0, x1 = int(live[:, 2].min()), int(live[:, 3].max())
    if (y1 - y0) * (x1 - x0) > max_fraction * height * width:
        return False
    return (y0, y1, x0, x1)


def _render_pt_mode(
    params: torch.Tensor,
    height: int,
    width: int,
    checkpoints: Sequence[int],
    canvas: torch.Tensor | Sequence[float] | None,
    cfg: RenderConfig,
) -> torch.Tensor:
    """Paint Transformer compatible rendering of our ``(B, N, 9)`` strokes (opacity = 0/1 decision,
    black canvas)."""
    if cfg.pt_alpha not in PT_MODES:
        raise ValueError(f"render.pt_alpha must be one of {PT_MODES}, got {cfg.pt_alpha!r}")
    p8, decisions = ours_to_pt(params)
    variant = "train" if cfg.pt_alpha == "pt_train" else "inference"
    brushes = cached_pt_brushes(variant, cfg.pt_brush_dir, str(params.device))
    start = None if canvas is None else make_canvas(canvas, params.shape[0], height, width, params)
    return render_pt_progressive(
        p8,
        height,
        width,
        brushes,
        checkpoints,
        cfg.pt_alpha,
        decisions,
        start,
        min(cfg.chunk_size, 16),
    )


def render_progressive(
    params: torch.Tensor,
    brush: torch.Tensor | BrushBank,
    height: int,
    width: int,
    checkpoints: Sequence[int],
    canvas: torch.Tensor | Sequence[float] | None = None,
    cfg: RenderConfig | None = None,
    brush_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Canvases after the first ``k`` strokes for every ``k`` in ``checkpoints`` (single pass).

    Strokes are rendered once, in order; the running canvas is recorded when it reaches each
    checkpoint, so the cost is that of rendering ``max(checkpoints)`` strokes.

    Args:
        params: ``(B, N, 9)`` stroke parameters.
        brush: brush masks ``(K, 1, h, w)`` or a :class:`BrushBank`.
        checkpoints: stroke counts in ``[0, N]`` (any order, duplicates ok); 0 = initial canvas.
        canvas: initial canvas (see :func:`make_canvas`), default white.
        cfg: :class:`RenderConfig`.
        brush_index: optional ``(B, N)`` brush variant per stroke (default: seeded random).

    Returns:
        ``(B, len(checkpoints), 3, H, W)`` in the order of ``checkpoints``.
    """
    check_params(params)
    cfg = cfg or RenderConfig()
    b, n, _ = params.shape
    if cfg.mode == "pt":
        return _render_pt_mode(params, height, width, checkpoints, canvas, cfg)
    if cfg.mode != "default":
        raise ValueError(f"render.mode must be 'default' or 'pt', got {cfg.mode!r}")
    if cfg.chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {cfg.chunk_size}")
    if any(not 0 <= k <= n for k in checkpoints):
        raise ValueError(f"checkpoints must lie in [0, {n}], got {list(checkpoints)}")
    dtype = _DTYPES[cfg.dtype] or params.dtype
    masks = brush_masks(brush).to(device=params.device, dtype=params.dtype)
    if brush_index is None:
        brush_index = default_brush_index(b, n, masks.shape[0], cfg.brush_seed)
    brush_index = brush_index.to(params.device)

    current = make_canvas(canvas, b, height, width, params)
    wanted = sorted(set(checkpoints))
    snapshots: dict[int, torch.Tensor] = {}
    done = 0
    step = cfg.bbox_chunk if cfg.bbox and cfg.bbox_chunk > 0 else cfg.chunk_size
    boxes = None
    if cfg.bbox:
        boxes = stroke_boxes(params, (masks.shape[-2], masks.shape[-1]), height, width)
    for target in wanted:
        while done < target:
            end = min(done + step, target)
            box = (
                _chunk_box(boxes, done, end, height, width, cfg.bbox_max_fraction)
                if boxes is not None
                else False
            )
            if box is None:  # every stroke of the chunk lies outside the canvas
                done = end
                continue
            fn: Callable[..., torch.Tensor]
            if box is False:
                fn = _composite_chunk
                args: tuple[Any, ...] = (
                    current,
                    params[:, done:end],
                    masks,
                    brush_index[:, done:end],
                    height,
                    width,
                    dtype,
                    cfg.sampler,
                )
            else:
                fn = _composite_chunk_bbox
                args = (
                    current,
                    params[:, done:end],
                    masks,
                    brush_index[:, done:end],
                    box,
                    height,
                    width,
                    dtype,
                    cfg.sampler,
                )
            if cfg.use_checkpoint and torch.is_grad_enabled():
                current = checkpoint(fn, *args, use_reentrant=False)
            else:
                current = fn(*args)
            done = end
        snapshots[target] = current
    return torch.stack([snapshots[k] for k in checkpoints], dim=1)


def render(
    params: torch.Tensor,
    brush: torch.Tensor | BrushBank,
    height: int,
    width: int,
    canvas: torch.Tensor | Sequence[float] | None = None,
    cfg: RenderConfig | None = None,
    brush_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render all N strokes of each painting: ``(B, 3, H, W)`` (see :func:`render_progressive`)."""
    n = params.shape[1]
    return render_progressive(params, brush, height, width, [n], canvas, cfg, brush_index)[:, 0]
