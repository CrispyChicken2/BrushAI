"""Paint Transformer multi-scale inference (T-015).

Reproduces the inference procedure of the official code (Liu et al., ICCV 2021, Algorithm 1;
``inference/inference.py:main`` and ``param2img_parallel`` at commit 7960b8e, see
``docs/design/paint_transformer.md`` section 6 and section 14):

1. the image is zero-padded (black, centred) to a ``32 * 2^K`` square, ``K = max(ceil(log2(max(H,
   W) / 32)), 0)``;
2. for each layer ``l = 0..K`` the target and the current canvas are resized (nearest) to ``32 *
   2^l``, cut into ``2^l x 2^l`` patches of 32 px, the network predicts 8 strokes per patch, the
   colour is sampled from the target at each predicted centre and the confident strokes
   (``logit > 0``) are painted by :func:`paint_layer` in four non-overlapping window phases;
3. a last layer at full resolution, shifted by half a patch (17 x 17 patches for ``K = 4``), is
   predicted from the final canvas. **Quirk of the official code:** its strokes are painted
   whatever their confidence. Both variants are computed here: ``official`` (all strokes) and
   ``gated`` (only ``logit > 0``).

Stroke parameters are rewritten in the layer canvas and then in the coordinates of the input image
(:func:`layers_to_sequence`), ordered by (layer, phase, stroke index, patch raster order), which is
the order in which the official renderer paints them.

``|h|``: the training losses are invariant to the sign of the stroke height (study note section 12,
T-014 finding 2), so a trained checkpoint may predict ``h < 0``. With ``abs_h`` (default) the
height is replaced by ``|h|`` right after the prediction, before the colour sampling, the painting
and the export, so that the canvas the network sees, the canvas that is painted and the exported
strokes are consistent. For checkpoints with ``h > 0`` (the official weights, the T-014 seeds 1 and
2) this is the identity.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F

from brushstroke.renderer.pipeline import RenderConfig
from brushstroke.renderer.pt import NUM_PT_PARAMS, pt_param2stroke
from brushstroke.stroke_model.model import PaintTransformer, PTModelConfig
from brushstroke.stroke_model.pt_sequence import (
    FINAL_LAYER_VARIANTS,
    PHASES,
    LayerStrokes,
    layers_to_sequence,
)
from brushstroke.stroke_model.refine import RefineConfig, refine_patch_strokes
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.device import peak_vram_mb, reset_peak_vram


class StrokePredictor(Protocol):
    """What the pipeline needs from the network (:class:`PaintTransformer` satisfies it)."""

    def strokes(
        self, img: torch.Tensor, canvas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(shape (B, N, 5), colour (B, N, 3), logits (B, N))`` for ``(B, 3, P, P)`` patches."""


@dataclass
class PTInferenceConfig:
    """``inference:`` config section."""

    patch_size: int = 32  # P: input size of the network (and size of the patches)
    final_layer: str = "official"  # variant exported to the sequence: official | gated
    abs_h: bool = True  # replace the predicted height by its absolute value (see module doc)
    brush_dir: str = "data/raw/pt_brush"  # official brush PNGs (scripts/fetch_pt_brushes.py)
    max_patches_per_batch: int = 0  # network batch size (0 = all the patches of a layer)
    drop_invisible: bool = True  # drop strokes entirely outside the image (only with padding)


@dataclass
class PaintPTConfig:
    """Root config of ``scripts/paint_pt.py`` (``configs/paint_pt.yaml``)."""

    seed: int = 0
    device: str = "auto"
    output_dir: str = "outputs/pt"
    # T-014 selected checkpoint (lowest held-out loss after |h| canonicalization)
    checkpoint: str = "outputs/pt_long/seed2/latest.pt"
    height: int = 512  # the input is centre-cropped and resized to height x width (0 = native)
    width: int = 512
    inference: PTInferenceConfig = field(default_factory=PTInferenceConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    render: RenderConfig = field(default_factory=lambda: RenderConfig(mode="pt"))


@dataclass
class PTResult:
    """Output of :func:`paint`.

    Attributes:
        canvas: ``(3, H, W)`` painting of the ``official`` variant (all final-layer strokes).
        canvas_gated: ``(3, H, W)`` painting with only the confident final-layer strokes.
        layer_canvases: canvas after each regular layer ``0..K`` then after the final layer
            (``official`` variant), each ``(3, H, W)``.
        layers: the predicted strokes of every layer.
        sequence: :class:`StrokeSequence` of the variant chosen in the config.
        stats: stroke counts, timings, peak VRAM, refinement losses.
    """

    canvas: torch.Tensor
    canvas_gated: torch.Tensor
    layer_canvases: list[torch.Tensor]
    layers: list[LayerStrokes]
    sequence: StrokeSequence
    stats: dict[str, object] = field(default_factory=dict)


# ----------------------------------------------------------------------------- geometry helpers


def num_layers(height: int, width: int, patch_size: int = 32) -> int:
    """``K = max(ceil(log2(max(H, W) / P)), 0)`` (official, paper Eq. 14)."""
    return max(math.ceil(math.log2(max(height, width) / patch_size)), 0)


def pad_square(img: torch.Tensor, size: int) -> torch.Tensor:
    """Black zero padding of ``(B, C, h, w)`` to ``size x size``, centred (extra row/column of odd
    padding at the bottom/right, as the official ``pad``)."""
    _, _, h, w = img.shape
    if h > size or w > size:
        raise ValueError(f"image {h}x{w} does not fit in a {size}x{size} square")
    top, left = (size - h) // 2, (size - w) // 2
    return F.pad(img, [left, size - w - left, top, size - h - top])


def crop_square(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of :func:`pad_square`."""
    size_h, size_w = img.shape[-2:]
    top, left = (size_h - height) // 2, (size_w - width) // 2
    return img[..., top : top + height, left : left + width]


def extract_patches(x: torch.Tensor, patch: int) -> torch.Tensor:
    """``(1, 3, n*P, m*P)`` -> ``(n*m, 3, P, P)`` non-overlapping patches in raster order."""
    cols = F.unfold(x, (patch, patch), stride=(patch, patch))
    return cols.permute(0, 2, 1).contiguous().view(-1, 3, patch, patch)


# ----------------------------------------------------------------------------- painting


def paint_layer(
    canvas: torch.Tensor,
    params: torch.Tensor,
    decision: torch.Tensor,
    brushes: torch.Tensor,
    mode: str = "pt_infer",
) -> torch.Tensor:
    """Paint the strokes of one layer on ``canvas``, window by window (official
    ``param2img_parallel``).

    The canvas is cut into windows of twice the patch size, one per patch, overlapping their
    neighbours by half a window. The windows are painted in four phases (:data:`PHASES`) so that
    the windows of a phase never overlap; inside a window the strokes are painted in index order
    and clipped to the window.

    Args:
        canvas: ``(B, 3, H, W)``, ``H = n_y * patch``, ``W = n_x * patch``.
        params: ``(B, n_y, n_x, N, 8)`` strokes in **window units**: ``x, y, w, h`` are fractions
            of the 2P x 2P window (the network output ``x / 2 + 0.25``, ``w / 2``), ``theta``
            and the colour unchanged.
        decision: ``(B, n_y, n_x, N)`` bool, stroke painted or not.
        brushes: ``(2, 1, h_b, w_b)`` official brush images (variant matching ``mode``).

    Returns:
        The new canvas ``(B, 3, H, W)``.
    """
    b, n_y, n_x, s, _ = params.shape
    height, width = canvas.shape[-2:]
    py, px = height // n_y, width // n_x
    wy, wx = 2 * py, 2 * px
    padded = F.pad(canvas, [px // 2, px // 2, py // 2, py // 2])
    ys = torch.arange(wy, device=canvas.device)
    xs = torch.arange(wx, device=canvas.device)
    for phase_y, phase_x in PHASES:
        ii = torch.arange(phase_y, n_y, 2, device=canvas.device)
        jj = torch.arange(phase_x, n_x, 2, device=canvas.device)
        if ii.numel() == 0 or jj.numel() == 0:
            continue
        rows = (py * ii)[:, None, None, None] + ys[None, None, :, None]  # (ni, 1, wy, 1)
        cols = (px * jj)[None, :, None, None] + xs[None, None, None, :]  # (1, nj, 1, wx)
        windows = padded[:, :, rows, cols].permute(0, 2, 3, 1, 4, 5)  # (B, ni, nj, 3, wy, wx)
        phase_params = params[:, ii][:, :, jj]
        phase_decision = decision[:, ii][:, :, jj]
        flat_decision = phase_decision.reshape(-1)
        foreground = torch.zeros(flat_decision.numel(), 3, wy, wx, device=canvas.device)
        alpha = torch.zeros_like(foreground)
        if bool(flat_decision.any()):
            fg, al = pt_param2stroke(
                phase_params.reshape(-1, NUM_PT_PARAMS)[flat_decision], wy, wx, brushes, mode
            )
            foreground[flat_decision], alpha[flat_decision] = fg, al
        foreground = foreground.view(b, ii.numel(), jj.numel(), s, 3, wy, wx)
        alpha = alpha.view(b, ii.numel(), jj.numel(), s, 3, wy, wx)
        gate = phase_decision.view(b, ii.numel(), jj.numel(), s, 1, 1, 1).to(canvas.dtype)
        for k in range(s):
            a = alpha[:, :, :, k] * gate[:, :, :, k]
            windows = foreground[:, :, :, k] * a + windows * (1 - a)
        padded[:, :, rows, cols] = windows.permute(0, 3, 1, 2, 4, 5)
    return padded[:, :, py // 2 : -(py // 2), px // 2 : -(px // 2)]


def to_window_units(params: torch.Tensor) -> torch.Tensor:
    """Patch units -> window units (``x / 2 + 0.25``, ``w / 2``; official ``main``)."""
    out = params.clone()
    out[..., :2] = out[..., :2] / 2 + 0.25
    out[..., 2:4] = out[..., 2:4] / 2
    return out


# ----------------------------------------------------------------------------- prediction


@torch.no_grad()
def predict_layer(
    net: StrokePredictor,
    img_patches: torch.Tensor,
    canvas_patches: torch.Tensor,
    batch: int = 0,
    abs_h: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stroke parameters ``(n, N, 8)`` (patch units, colour sampled from the target, ``|h|`` if
    ``abs_h``), logits ``(n, N)`` and the raw predicted heights ``(n, N)`` of every patch of a
    layer."""
    n = img_patches.shape[0]
    step = n if batch <= 0 else batch
    params, logits, heights = [], [], []
    for start in range(0, n, step):
        sl = slice(start, start + step)
        shape, colour, logit = net.strokes(img_patches[sl], canvas_patches[sl])
        heights.append(shape[..., 3].clone())
        if abs_h:
            shape = shape.clone()
            shape[..., 3] = shape[..., 3].abs()
        params.append(torch.cat([shape, colour], dim=-1))
        logits.append(logit)
    return torch.cat(params), torch.cat(logits), torch.cat(heights)


def _layer_decisions(logits: torch.Tensor, final: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``(painted, confident)``: confident = ``logit > 0``; the official final layer paints every
    stroke whose logit is non zero (``decision.bool()`` on the raw logits)."""
    confident = logits > 0
    return (logits != 0 if final else confident), confident


# ----------------------------------------------------------------------------- the pipeline


@torch.no_grad()
def paint(
    net: StrokePredictor,
    image: torch.Tensor,
    brushes: torch.Tensor,
    cfg: PTInferenceConfig | None = None,
    refine: RefineConfig | None = None,
) -> PTResult:
    """Multi-scale Paint Transformer painting of ``image``.

    Args:
        net: trained predictor in eval mode.
        image: ``(3, H, W)`` float in [0, 1] (any size; padded to a square internally).
        brushes: ``(2, 1, h_b, w_b)`` official inference brush images
            (:func:`brushstroke.renderer.pt.load_pt_brushes`).
        cfg: inference options.
        refine: optional per-patch refinement (off by default), see :mod:`.refine`.

    Returns:
        A :class:`PTResult`: paintings of both final-layer variants, per-layer strokes and the
        :class:`StrokeSequence` of the variant ``cfg.final_layer``.
    """
    cfg = cfg or PTInferenceConfig()
    refine = refine or RefineConfig()
    if cfg.final_layer not in FINAL_LAYER_VARIANTS:
        raise ValueError(
            f"final_layer must be one of {FINAL_LAYER_VARIANTS}, got {cfg.final_layer}"
        )
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected an image of shape (3, H, W), got {tuple(image.shape)}")
    device = image.device
    patch = cfg.patch_size
    height, width = image.shape[-2:]
    k_max = num_layers(height, width, patch)
    size = patch * 2**k_max
    reset_peak_vram()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    target = pad_square(image[None], size)
    canvas = torch.zeros_like(target)
    layers: list[LayerStrokes] = []
    layer_canvases: list[torch.Tensor] = []
    refine_stats: list[dict[str, float]] = []
    negative_h = [0, 0]  # painted strokes with a negative predicted height, painted strokes

    def predict(
        img_l: torch.Tensor, canvas_l: torch.Tensor, grid: int, level: int, final: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_p, canvas_p = extract_patches(img_l, patch), extract_patches(canvas_l, patch)
        params, logits, raw_h = predict_layer(
            net, img_p, canvas_p, cfg.max_patches_per_batch, cfg.abs_h
        )
        painted, confident = _layer_decisions(logits, final)
        negative_h[0] += int((raw_h[painted] < 0).sum())
        negative_h[1] += int(painted.sum())
        if refine.enabled and level in refine_levels(refine, k_max + 2):
            params, stats = refine_patch_strokes(img_p, canvas_p, params, painted, brushes, refine)
            refine_stats.append({"level": float(level), **stats})
        n_stroke = params.shape[1]
        return (
            params.view(grid, grid, n_stroke, NUM_PT_PARAMS),
            logits.view(grid, grid, n_stroke),
            painted.view(grid, grid, n_stroke),
            confident.view(grid, grid, n_stroke),
        )

    for level in range(k_max + 1):
        side = patch * 2**level
        grid = 2**level
        img_l = F.interpolate(target, (side, side))  # nearest, as the official default
        canvas_l = F.interpolate(canvas, (side, side))
        params, logits, painted, confident = predict(img_l, canvas_l, grid, level, False)
        layers.append(LayerStrokes(level, grid, side, False, params, logits, painted, confident))
        # the strokes are painted on the full-resolution canvas: the layer only sets the window size
        canvas = paint_layer(canvas, to_window_units(params)[None], painted[None], brushes)
        layer_canvases.append(crop_square(canvas, height, width)[0])

    # shifted final layer: image and canvas padded by half a patch, 17 x 17 patches for K = 4
    border = patch // 2
    img_s = F.pad(F.interpolate(target, (size, size)), [border] * 4)
    canvas_s = F.pad(canvas, [border] * 4)
    grid = img_s.shape[-1] // patch
    params, logits, painted, confident = predict(img_s, canvas_s, grid, k_max + 1, True)
    layers.append(LayerStrokes(k_max + 1, grid, size, True, params, logits, painted, confident))
    window_params = to_window_units(params)[None]
    official = paint_layer(canvas_s, window_params, painted[None], brushes)
    gated = paint_layer(canvas_s, window_params, confident[None], brushes)
    final_official = crop_square(official[:, :, border:-border, border:-border], height, width)[0]
    final_gated = crop_square(gated[:, :, border:-border, border:-border], height, width)[0]
    layer_canvases.append(final_official)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    sequence, dropped = layers_to_sequence(
        layers, (height, width), size, cfg.final_layer, cfg.drop_invisible
    )
    painted_counts = [int(layer.painted.sum()) for layer in layers]
    confident_final = int(layers[-1].confident.sum())
    sequence.meta.update(
        {
            "producer": "paint_transformer",
            "levels": k_max + 1,
            "padded_size": size,
            "patch_size": patch,
            "final_layer": cfg.final_layer,
            "abs_h": cfg.abs_h,
            "refined": bool(refine.enabled),
            "dropped_invisible": dropped,
            "runtime_seconds": seconds,
            "peak_vram_mb": peak_vram_mb(),
        }
    )
    stats: dict[str, object] = {
        "layers": k_max + 1,
        "painted_per_layer": painted_counts,
        "strokes_official": sum(painted_counts),
        "strokes_gated": sum(painted_counts[:-1]) + confident_final,
        "strokes_exported": len(sequence),
        "dropped_invisible": dropped,
        "negative_h_fraction": negative_h[0] / max(negative_h[1], 1),
        "runtime_seconds": seconds,
        "peak_vram_mb": peak_vram_mb(),
        "refine_layers": refine_stats,
    }
    return PTResult(final_official, final_gated, layer_canvases, layers, sequence, stats)


def refine_levels(refine: RefineConfig, count: int) -> list[int]:
    """Layers refined: ``refine.levels`` or all of them when empty."""
    return list(refine.levels) if refine.levels else list(range(count))


# ----------------------------------------------------------------------------- checkpoints


def load_pt_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[PaintTransformer, dict[str, object]]:
    """Load a Paint Transformer in eval mode from a training checkpoint (``latest.pt`` of
    :mod:`brushstroke.stroke_model.train`) or from a plain ``state_dict`` such as the released
    ``model.pth``.

    Returns:
        ``(network, info)``; ``info`` has the checkpoint ``step`` and ``kind`` (``"training"`` or
        ``"state_dict"``).

    Raises:
        FileNotFoundError: ``path`` does not exist.
    """
    ckpt = Path(path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"checkpoint {ckpt} not found (see docs/experiments/pt_training.md)"
        )
    try:  # plain state_dicts (the released weights) load without unpickling arbitrary objects
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - our own training checkpoints hold the config and RNG states
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
    info: dict[str, object] = {"kind": "state_dict", "step": None}
    model_cfg = PTModelConfig()
    if isinstance(state, dict) and "model" in state and "config" in state:
        info = {"kind": "training", "step": int(state.get("step", -1))}
        raw_cfg = state["config"].get("model", {})
        model_cfg = PTModelConfig(**raw_cfg)
        state = state["model"]
    net = PaintTransformer(model_cfg)
    net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    net.eval().requires_grad_(False)
    return net.to(device), info
