"""Paint Transformer strokes as a :class:`~brushstroke.strokes.StrokeSequence` (T-015).

Two things live here:

* :func:`layers_to_sequence` converts the strokes predicted layer by layer (patch units, see
  :class:`LayerStrokes`) to a sequence in the coordinates of the input image, ordered like the
  official renderer paints them: (layer, phase, stroke index, patch raster order inside the phase);
* :func:`replay_sequence` re-renders such a sequence the way the multi-scale pipeline painted it
  (brush resized to the stroke's window, strokes clipped to their window), which reproduces the
  pipeline canvas. The generic renderer (``StrokeSequence.render`` with ``RenderConfig(mode="pt")``)
  draws every stroke at the final resolution with the brush resized to the whole canvas and no
  clipping: same strokes, slightly different texture sampling (measured in
  ``docs/design/paint_transformer.md`` section 14).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from brushstroke.renderer.pt import (
    NUM_PT_PARAMS,
    dilation,
    erosion,
    patch_to_canvas,
    pt_param2stroke,
)
from brushstroke.strokes import BrushSpec, StrokeSequence

# Window phases (row parity, column parity) in the order of the official renderer:
# even/even, odd/odd, odd/even, even/odd window indices. Windows of one phase do not overlap.
PHASES = ((0, 0), (1, 1), (1, 0), (0, 1))
FINAL_LAYER_VARIANTS = ("official", "gated")
SOURCE_CONFIDENT = "paint_transformer"
_MARGIN = 1  # replay: extra pixels around the canvas (radius of the 3 x 3 morphology)
SOURCE_FORCED = "paint_transformer_forced"  # final-layer stroke painted by the official code only


@dataclass
class LayerStrokes:
    """The strokes predicted for one layer, in patch units (what the network outputs).

    Attributes:
        level: 0..K for the regular layers, K + 1 for the shifted final layer.
        grid: patches per side ``n`` (``n x n`` patches; ``2^K + 1`` for the shifted layer).
        canvas_size: side in pixels of the canvas the patches tile: ``32 * n`` for a regular layer
            (the network sees the canvas resized to it), the padded square ``32 * 2^K`` for the
            shifted layer (whose patch grid is offset by half a patch).
        shifted: True for the final layer.
        params: ``(n, n, N, 8)`` strokes ``x, y, w, h, theta, r, g, b``; ``x, y, w, h`` in units of
            the patch size (after ``|h|`` if enabled).
        logits: ``(n, n, N)`` confidence logits.
        painted: ``(n, n, N)`` bool, painted in the ``official`` variant.
        confident: ``(n, n, N)`` bool, ``logit > 0``.
    """

    level: int
    grid: int
    canvas_size: int
    shifted: bool
    params: torch.Tensor
    logits: torch.Tensor
    painted: torch.Tensor
    confident: torch.Tensor


def layers_to_sequence(
    layers: list[LayerStrokes],
    image_size: tuple[int, int],
    padded_size: int,
    variant: str = "official",
    drop_invisible: bool = True,
) -> tuple[StrokeSequence, int]:
    """Convert the predicted layers to a :class:`StrokeSequence` of the (unpadded) image.

    ``scale_level`` is the layer (``K + 1`` = shifted final layer), ``patch_id`` the raster index
    ``row * n + col`` of the patch in its layer grid. Parameters are in the renderer convention
    (``x, y, w, h`` fractions of the image width and height, ``a = 1``) and the background is
    black; render with ``RenderConfig(mode="pt")`` or :func:`replay_sequence`.

    Args:
        variant: ``official`` keeps every painted stroke (final-layer strokes with ``logit <= 0``
            are tagged ``paint_transformer_forced``), ``gated`` drops those.
        drop_invisible: drop strokes whose bounding disc lies outside the image (possible only
            when the image was padded).

    Returns:
        ``(sequence, number of dropped invisible strokes)``.
    """
    if variant not in FINAL_LAYER_VARIANTS:
        raise ValueError(f"variant must be one of {FINAL_LAYER_VARIANTS}, got {variant!r}")
    height, width = image_size
    pad_y, pad_x = (padded_size - height) // 2, (padded_size - width) // 2
    chunks: list[torch.Tensor] = []
    level_tags: list[torch.Tensor] = []
    patch_tags: list[torch.Tensor] = []
    sources: list[str] = []
    dropped = 0
    for layer in layers:
        n, n_stroke = layer.grid, layer.params.shape[2]
        keep = (layer.painted if variant == "official" else layer.confident).cpu()
        confident = layer.confident.cpu()
        params = layer.params.cpu()
        for phase_y, phase_x in PHASES:
            ii = torch.arange(phase_y, n, 2)
            jj = torch.arange(phase_x, n, 2)
            if ii.numel() == 0 or jj.numel() == 0:
                continue
            ss, ri, cj = torch.meshgrid(torch.arange(n_stroke), ii, jj, indexing="ij")
            ss, ri, cj = ss.flatten(), ri.flatten(), cj.flatten()
            sel = keep[ri, cj, ss]
            ss, ri, cj = ss[sel], ri[sel], cj[sel]
            if ss.numel() == 0:
                continue
            canvas = patch_to_canvas(
                params[ri, cj, ss], ri, cj, layer.canvas_size, shifted=layer.shifted
            )  # fractions of the padded square
            out = _to_image_fractions(canvas, padded_size, pad_y, pad_x, height, width)
            if drop_invisible:
                visible = _visible(out, height, width)
                dropped += int((~visible).sum())
                out, ri, cj, ss = out[visible], ri[visible], cj[visible], ss[visible]
            chunks.append(out)
            level_tags.append(torch.full((out.shape[0],), layer.level, dtype=torch.int32))
            patch_tags.append((ri * n + cj).to(torch.int32))
            forced = layer.shifted & ~confident[ri, cj, ss]
            sources.extend(SOURCE_FORCED if f else SOURCE_CONFIDENT for f in forced.tolist())
    rows = torch.cat(chunks) if chunks else torch.zeros(0, NUM_PT_PARAMS)
    seq = StrokeSequence(
        params=torch.cat([rows, torch.ones(rows.shape[0], 1)], dim=1),
        canvas_size=(height, width),
        brush=BrushSpec(),
        background=(0.0, 0.0, 0.0),
        scale_level=torch.cat(level_tags) if level_tags else None,
        patch_id=torch.cat(patch_tags) if patch_tags else None,
        source=sources,
    )
    return seq, dropped


def _to_image_fractions(
    canvas: torch.Tensor, size: int, pad_y: int, pad_x: int, height: int, width: int
) -> torch.Tensor:
    """Strokes in fractions of the padded square -> fractions of the unpadded image."""
    out = canvas.clone()
    out[:, 0] = (canvas[:, 0] * size - pad_x) / width
    out[:, 1] = (canvas[:, 1] * size - pad_y) / height
    out[:, 2] = canvas[:, 2] * size / width
    out[:, 3] = canvas[:, 3] * size / height
    return out


def _visible(params: torch.Tensor, height: int, width: int, margin: float = 2.0) -> torch.Tensor:
    """Mask of strokes whose bounding disc (half diagonal of the box + ``margin`` px, which covers
    the 3 x 3 morphology) meets the image."""
    cx, cy = params[:, 0] * width, params[:, 1] * height
    radius = 0.5 * torch.hypot(params[:, 2] * width, params[:, 3] * height) + margin
    return (cx + radius > 0) & (cx - radius < width) & (cy + radius > 0) & (cy - radius < height)


# ----------------------------------------------------------------------------- replay


def window_geometry(
    level: torch.Tensor, patch_id: torch.Tensor, padded_size: int, levels: int, patch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window of every stroke in the padded square: ``(rect (M, 4), brush size (M,))``.

    ``rect`` is ``(top, bottom, left, right)`` in pixels of the padded square (bottom/right
    exclusive); the brush size is the side the official code resizes the brush to (the window
    side). A regular layer ``l`` has windows of ``2 * S / 2^l`` px centred on their patch; the
    shifted final layer has 64 px (``2 * patch``) windows on the half-patch-shifted grid.
    """
    final = level == levels
    grid = torch.where(final, torch.full_like(level, 2 ** (levels - 1) + 1), 2**level)
    row, col = patch_id // grid, patch_id % grid
    side = torch.where(final, torch.full_like(level, patch), padded_size // 2**level)  # patch side
    # patch origin (the shifted grid starts half a patch before the padded square), minus the margin
    origin_offset = torch.where(final, torch.full_like(level, patch // 2), torch.zeros_like(level))
    top = side * row - origin_offset - side // 2
    left = side * col - origin_offset - side // 2
    window = 2 * side
    rect = torch.stack([top, top + window, left, left + window], dim=1)
    return rect, window


@torch.no_grad()
def replay_sequence(
    seq: StrokeSequence, brushes: torch.Tensor, chunk_size: int = 16
) -> torch.Tensor:
    """Re-render a Paint Transformer sequence like the pipeline painted it: ``(3, H, W)``.

    Every stroke is drawn on the canvas of the sequence (native resolution) with the brush
    nearest-resized to the side of the window the pipeline painted it in and clipped to that
    window; this needs the ``scale_level`` / ``patch_id`` tags and the ``padded_size`` /
    ``levels`` / ``patch_size`` entries of ``seq.meta`` written by :func:`paint`. The result
    equals the pipeline canvas up to float rounding of the parameter round trip.

    Args:
        seq: sequence produced by :func:`paint` (black background, ``a = 1``).
        brushes: ``(2, 1, h_b, w_b)`` official inference brush images.
        chunk_size: strokes rendered at once (each chunk shares one brush size).
    """
    meta = seq.meta
    for key in ("padded_size", "levels", "patch_size"):
        if key not in meta:
            raise ValueError(f"replay needs meta['{key}'] (sequence not made by paint())")
    assert seq.scale_level is not None and seq.patch_id is not None
    device = seq.params.device
    height, width = seq.canvas_size
    size, levels, patch = int(meta["padded_size"]), int(meta["levels"]), int(meta["patch_size"])
    pad_y, pad_x = (size - height) // 2, (size - width) // 2
    rect, window_t = window_geometry(
        seq.scale_level.long(), seq.patch_id.long(), size, levels, patch
    )
    window = window_t.tolist()
    # the strokes are rasterized on a canvas one pixel larger on every side so that the 3 x 3
    # morphology of the border pixels sees the stroke beyond the image, as in the padded pipeline
    m = _MARGIN
    ext_h, ext_w = height + 2 * m, width + 2 * m
    shift = torch.tensor([pad_y - m, pad_y - m, pad_x - m, pad_x - m], device=device)
    rect = rect.to(device) - shift
    ys = torch.arange(ext_h, device=device).view(1, ext_h, 1)
    xs = torch.arange(ext_w, device=device).view(1, 1, ext_w)
    canvas = torch.tensor(seq.background, device=device).view(1, 3, 1, 1)
    canvas = canvas.expand(1, 3, ext_h, ext_w).clone()
    params = seq.params[:, :NUM_PT_PARAMS].to(device).clone()
    params[:, 0] = (params[:, 0] * width + m) / ext_w
    params[:, 1] = (params[:, 1] * height + m) / ext_h
    params[:, 2] = params[:, 2] * width / ext_w
    params[:, 3] = params[:, 3] * height / ext_h
    start, n = 0, params.shape[0]
    while start < n:
        # a chunk ends at a change of window size (the brush resize is shared by the chunk)
        end = start + 1
        while end < n and end - start < chunk_size and window[end] == window[start]:
            end += 1
        fg, alpha = pt_param2stroke(
            params[start:end],
            ext_h,
            ext_w,
            brushes,
            "pt_infer",
            morphology=False,
            brush_size=(window[start], window[start]),
            select_by_pixels=True,
        )
        r = rect[start:end]
        inside = (
            (ys >= r[:, 0].view(-1, 1, 1))
            & (ys < r[:, 1].view(-1, 1, 1))
            & (xs >= r[:, 2].view(-1, 1, 1))
            & (xs < r[:, 3].view(-1, 1, 1))
        ).unsqueeze(1)
        # morphology as inside the window: the official filters see the window only (border value
        # +1e9 for the erosion, -1e9 for the dilation), the pixels outside must not count
        alpha = erosion(torch.where(inside, alpha, torch.ones_like(alpha))) * inside
        fg = dilation(fg * inside)
        for k in range(end - start):
            canvas = fg[k : k + 1] * alpha[k : k + 1] + canvas * (1 - alpha[k : k + 1])
        start = end
    return canvas[0, :, m:-m, m:-m]
