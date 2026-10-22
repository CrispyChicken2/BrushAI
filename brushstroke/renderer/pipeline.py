"""Chunked, differentiable rendering of ordered strokes onto a canvas.

Strokes are processed in chunks of ``chunk_size``: for each chunk the alphas are rasterized,
composited in closed form over the running canvas, and freed, so peak memory is
``O(chunk_size * H * W)`` instead of ``O(N * H * W)``. With ``use_checkpoint`` the chunk's
activations are recomputed in the backward pass (``torch.utils.checkpoint``), so the autograd graph
keeps only one ``(B, 3, H, W)`` canvas per chunk boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from brushstroke.renderer.brushes import BrushBank
from brushstroke.renderer.compositing import composite_parallel
from brushstroke.renderer.params import B as BLUE
from brushstroke.renderer.params import R as RED
from brushstroke.renderer.params import check_params
from brushstroke.renderer.rasterize import brush_masks, default_brush_index, render_alphas

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
    for target in wanted:
        while done < target:
            end = min(done + cfg.chunk_size, target)
            args = (
                current,
                params[:, done:end],
                masks,
                brush_index[:, done:end],
                height,
                width,
                dtype,
                cfg.sampler,
            )
            if cfg.use_checkpoint and torch.is_grad_enabled():
                current = checkpoint(_composite_chunk, *args, use_reentrant=False)
            else:
                current = _composite_chunk(*args)
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
