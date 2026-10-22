"""Frame rendering: incremental compositing of completed strokes + partial reveal of in-flight ones.

Strokes are painted in index order. At every frame the finished *prefix* of strokes (all strokes
before the first unfinished one) is kept as a composited canvas, and only the strokes of the window
``[prefix, started)`` are rasterized, with their current reveal progress, and composited over it.
This is exactly the full composite of all strokes at their progress, at a cost proportional to the
number of strokes in flight. The geometry comes from the T-003 renderer (``rasterize_flat`` with its
``reveal`` argument); the canvas texture of T-002 modulates every frame.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from brushstroke.animation.config import AnimationConfig
from brushstroke.animation.timeline import Timeline, build_timeline
from brushstroke.renderer.brushes import get_brush_bank
from brushstroke.renderer.canvas import apply_canvas, make_canvas_texture
from brushstroke.renderer.compositing import composite_parallel
from brushstroke.renderer.params import B as BLUE
from brushstroke.renderer.params import R as RED
from brushstroke.renderer.params import canonicalize
from brushstroke.renderer.rasterize import rasterize_flat
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.device import get_device


def output_size(seq: StrokeSequence, cfg: AnimationConfig) -> tuple[int, int]:
    """Output ``(H, W)``: the config override or the sequence's canvas size."""
    h, w = seq.canvas_size
    return (cfg.height or h, cfg.width or w)


class FrameRenderer:
    """Renders the frames of one sequence; use :meth:`frames` to iterate over all of them."""

    def __init__(
        self, seq: StrokeSequence, cfg: AnimationConfig, timeline: Timeline | None = None
    ) -> None:
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.height, self.width = output_size(seq, cfg)
        self.timeline = timeline or build_timeline(len(seq), cfg)
        self.n = len(seq)
        spec = seq.brush
        bank = get_brush_bank(
            spec.kind, spec.bank_size, (spec.mask_height, spec.mask_width), spec.seed
        )
        self.masks = bank.masks.to(self.device)
        assert seq.brush_index is not None
        self.brush_index = seq.brush_index.to(self.device)
        params = seq.params.to(self.device)
        self.params = canonicalize(params[None], self.height, self.width)[0]
        self.colors = params[:, RED : BLUE + 1]
        self.reverse = torch.from_numpy(self.timeline.reverse).to(self.device)
        self.background = torch.tensor(seq.background, dtype=torch.float32, device=self.device)
        self.tex = (
            make_canvas_texture(
                self.height,
                self.width,
                cfg.canvas_seed,
                weave_period=cfg.weave_period,
                low_freq_amount=cfg.low_freq_amount,
                low_freq_scale=cfg.low_freq_scale,
            ).to(self.device)
            if cfg.canvas_enabled
            else None
        )
        self.reset()

    def reset(self) -> None:
        """Back to the empty canvas (frames must then be requested in increasing order)."""
        self._base = self.background.view(3, 1, 1).expand(3, self.height, self.width).contiguous()
        self._prefix = 0  # strokes [0, prefix) are finished and composited into _base

    def _composite(
        self, canvas: torch.Tensor, lo: int, hi: int, progress: np.ndarray | None
    ) -> torch.Tensor:
        """Composite strokes ``[lo, hi)`` over ``canvas`` (``(3, H, W)``) in index order."""
        for a in range(lo, hi, self.cfg.chunk_size):
            b = min(a + self.cfg.chunk_size, hi)
            reveal = None
            if progress is not None:
                reveal = torch.from_numpy(progress[a - lo : b - lo]).to(self.device, torch.float32)
            alphas = rasterize_flat(
                self.params[a:b],
                self.masks[self.brush_index[a:b]],
                self.height,
                self.width,
                reveal=reveal,
                reveal_edge=self.cfg.reveal_edge,
                reveal_reverse=self.reverse[a:b],
            )
            canvas = composite_parallel(canvas[None], alphas[None], self.colors[a:b][None])[0]
        return canvas

    @torch.no_grad()
    def render_frame(self, frame: int) -> torch.Tensor:
        """Float frame ``(3, H, W)`` in [0, 1], canvas texture included. Ask in increasing order."""
        tl = self.timeline
        progress = tl.progress(frame)
        finished = (
            int(np.argmin(progress >= 1.0))
            if self.n and not bool((progress >= 1.0).all())
            else self.n
        )
        if finished > self._prefix:
            self._base = self._composite(self._base, self._prefix, finished, None)
            self._prefix = finished
        started = int(np.searchsorted(tl.start, tl.frame_time(frame), side="left"))
        started = max(started, self._prefix)
        canvas = self._base
        if started > self._prefix:
            canvas = self._composite(
                canvas, self._prefix, started, progress[self._prefix : started]
            )
        if self.tex is not None:
            canvas = apply_canvas(canvas, self.tex, self.cfg.canvas_strength)
        return canvas

    def frames(self) -> Iterator[np.ndarray]:
        """Yield every frame as a contiguous ``(H, W, 3)`` uint8 array (hold frames repeat)."""
        self.reset()
        last: np.ndarray | None = None
        for f in range(self.timeline.num_frames):
            if f >= self.timeline.paint_frames and last is not None:
                yield last
                continue
            img = self.render_frame(f)
            last = np.ascontiguousarray(
                (img.clamp(0, 1) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            )
            yield last
