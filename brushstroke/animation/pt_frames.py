"""Frames of a Paint Transformer sequence (T-018): the official brush images with partial reveal.

A :class:`~brushstroke.strokes.StrokeSequence` produced by the Paint Transformer pipeline
(:mod:`brushstroke.stroke_model.pt_sequence`: strokes ``x, y, w, h, theta, r, g, b`` + decision,
black background) is drawn with the PT renderer (:func:`~brushstroke.renderer.pt.pt_param2stroke`,
``pt_infer`` mode: brush images resized to the canvas, strokes at the final resolution), not with
the procedural brush bank. The reveal front runs along the long axis of the brush image in the
stroke frame and multiplies the stroke alpha, so a revealed stroke is a part of the full one and
the last frame is the generic PT render of the sequence. Disable the canvas texture for these
frames (``canvas_enabled: false``): the PT canvas is black and the texture would only shade strokes.
"""

from __future__ import annotations

import numpy as np
import torch

from brushstroke.animation.config import AnimationConfig
from brushstroke.animation.frames import FrameRenderer
from brushstroke.animation.timeline import Timeline
from brushstroke.renderer.pt import load_pt_brushes, pt_composite, pt_param2stroke, pt_warp_grid
from brushstroke.renderer.rasterize import reveal_front
from brushstroke.strokes import StrokeSequence

PT_SOURCES = ("paint_transformer", "paint_transformer_forced")


def is_pt_sequence(seq: StrokeSequence) -> bool:
    """True if the strokes come from the Paint Transformer pipeline (``source`` tags)."""
    source = seq.source or []
    return len(source) > 0 and all(s in PT_SOURCES for s in source)


class PTFrameRenderer(FrameRenderer):
    """:class:`FrameRenderer` drawing PT strokes (same timeline, prefix and window logic)."""

    def _setup(self, seq: StrokeSequence) -> None:
        self.brushes = load_pt_brushes("inference", self.cfg.pt_brush_dir, self.device)
        self.pt_params = seq.params[:, :8].to(self.device, torch.float32)
        self.decision = (seq.params[:, 8] > 0).to(self.device, torch.float32)

    def _composite(
        self, canvas: torch.Tensor, lo: int, hi: int, progress: np.ndarray | None
    ) -> torch.Tensor:
        for a in range(lo, hi, self.cfg.chunk_size):
            b = min(a + self.cfg.chunk_size, hi)
            params = self.pt_params[a:b]
            fg, alpha = pt_param2stroke(
                params, self.height, self.width, self.brushes, select_by_pixels=True
            )
            if progress is not None:
                reveal = torch.from_numpy(progress[a - lo : b - lo]).to(self.device, torch.float32)
                grid = pt_warp_grid(params, self.height, self.width)
                tall = (params[:, 3] * self.height > params[:, 2] * self.width).view(-1, 1, 1)
                axis = torch.where(tall, grid[..., 1], grid[..., 0])  # long axis of the brush image
                front = reveal_front(axis, reveal, self.cfg.reveal_edge, self.reverse[a:b])
                alpha = alpha * front[:, None]
            dec = self.decision[a:b][None]
            canvas = pt_composite(canvas[None], fg[None], alpha[None], dec)[0]
        return canvas


def make_frame_renderer(
    seq: StrokeSequence, cfg: AnimationConfig, timeline: Timeline | None = None
) -> FrameRenderer:
    """The frame renderer of ``cfg.renderer`` (``auto``: PT strokes -> PT, else procedural)."""
    use_pt = cfg.renderer == "pt" or (cfg.renderer == "auto" and is_pt_sequence(seq))
    if cfg.renderer not in ("auto", "pt", "procedural"):
        raise ValueError(f"renderer must be auto | pt | procedural, got {cfg.renderer!r}")
    return (PTFrameRenderer if use_pt else FrameRenderer)(seq, cfg, timeline)
