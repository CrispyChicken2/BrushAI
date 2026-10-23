"""Deterministic stroke timeline: when each stroke starts and how long its reveal lasts.

Time runs over ``duration`` seconds of painting followed by ``hold`` seconds of the finished
painting; ``num_frames = round(fps * duration) + round(fps * hold)``. Strokes are painted in index
order (start times are non-decreasing), several can be in flight at once (``overlap``), and the
every stroke has ended when the painting phase ends.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brushstroke.animation.config import AnimationConfig

DIRECTIONS = ("forward", "reverse", "random")
EASINGS = ("uniform", "decelerate", "by_level")
STROKE_EASES = ("linear", "smoothstep")


@dataclass(frozen=True)
class Timeline:
    """Start time, reveal duration (seconds) and direction of every stroke.

    Attributes:
        start: ``(N,)`` non-decreasing start times in ``[0, paint_seconds]``.
        duration: ``(N,)`` reveal durations; ``start + duration <= paint_seconds``.
        reverse: ``(N,)`` bool, reveal from the far end of the main axis.
        fps: frames per second.
        paint_frames: number of frames of the painting phase.
        hold_frames: number of frames of the final hold.
        stroke_ease: ``"linear"`` or ``"smoothstep"`` progress curve inside a stroke.
    """

    start: np.ndarray
    duration: np.ndarray
    reverse: np.ndarray
    fps: int
    paint_frames: int
    hold_frames: int
    stroke_ease: str = "smoothstep"

    @property
    def num_frames(self) -> int:
        return self.paint_frames + self.hold_frames

    @property
    def paint_seconds(self) -> float:
        return self.paint_frames / self.fps

    def frame_time(self, frame: int) -> float:
        """Time of ``frame``: the painting phase is spread so that its last frame is at ``T``."""
        if frame >= self.paint_frames:
            return self.paint_seconds
        return self.paint_seconds * frame / max(1, self.paint_frames - 1)

    def progress(self, frame: int) -> np.ndarray:
        """``(N,)`` reveal progress in [0, 1] of every stroke at ``frame`` (monotone in time)."""
        t = self.frame_time(frame)
        raw = np.clip((t - self.start) / self.duration, 0.0, 1.0)
        if self.stroke_ease == "smoothstep":
            return np.asarray(raw * raw * (3.0 - 2.0 * raw))
        return np.asarray(raw)


def num_frames(cfg: AnimationConfig) -> int:
    """``round(fps * duration) + round(fps * hold)``."""
    return round(cfg.fps * cfg.duration) + round(cfg.fps * cfg.hold)


def build_timeline(
    num_strokes: int, cfg: AnimationConfig, levels: np.ndarray | None = None
) -> Timeline:
    """Assign every stroke a start time, a reveal duration and a direction (deterministic).

    Each stroke gets a time slot (equal for ``easing="uniform"``; growing linearly up to
    ``decel_ratio`` times for ``"decelerate"``, i.e. fast background strokes and slower details).
    With ``easing="by_level"`` the time of a whole coarse-to-fine level (``levels``: ``(N,)`` int
    tags, e.g. ``scale_level``) is proportional to ``count ** level_time_power``: power 1 is the
    uniform pacing, 0 gives every level the same time, so the few coarse strokes that lay down the
    big masses are painted slowly enough to be seen.
    Its reveal lasts ``overlap x slot`` but at least ``min_reveal_frames`` frames (so that it does
    not pop in), and the starts are spread over the remaining time so that every stroke ends by the
    end of the painting phase (the last one exactly then when it has the longest reveal).
    """
    if cfg.easing not in EASINGS or cfg.direction not in DIRECTIONS:
        raise ValueError(f"easing must be in {EASINGS} and direction in {DIRECTIONS}")
    if cfg.stroke_ease not in STROKE_EASES:
        raise ValueError(f"stroke_ease must be in {STROKE_EASES}")
    if cfg.fps < 1 or cfg.duration <= 0 or cfg.hold < 0 or cfg.overlap <= 0:
        raise ValueError("fps >= 1, duration > 0, hold >= 0 and overlap > 0 are required")
    paint_frames = round(cfg.fps * cfg.duration)
    if paint_frames < 1:
        raise ValueError("duration is shorter than one frame")
    hold_frames = round(cfg.fps * cfg.hold)
    total = paint_frames / cfg.fps
    n = num_strokes
    if n == 0:
        empty = np.zeros(0)
        return Timeline(
            empty, empty, empty.astype(bool), cfg.fps, paint_frames, hold_frames, cfg.stroke_ease
        )

    if cfg.easing == "by_level":
        if levels is None or len(levels) != n:
            raise ValueError("easing 'by_level' needs one level tag per stroke")
        counts = {lv: int((levels == lv).sum()) for lv in np.unique(levels)}
        weights = np.array(
            [counts[lv] ** (cfg.level_time_power - 1.0) for lv in levels], dtype=float
        )
    elif cfg.easing == "decelerate" and n > 1:
        weights = np.linspace(1.0, cfg.decel_ratio, n)
    else:
        weights = np.ones(n)
    slot = total * weights / weights.sum()
    duration = np.maximum(cfg.overlap * slot, cfg.min_reveal_frames / cfg.fps)
    duration = np.minimum(duration, total)
    cum = np.concatenate([[0.0], np.cumsum(weights)[:-1]])
    frac = cum / cum[-1] if n > 1 else np.zeros(1)
    start = frac * (total - duration.max())  # non-decreasing; start + duration <= total

    if cfg.direction == "forward":
        reverse = np.zeros(n, dtype=bool)
    elif cfg.direction == "reverse":
        reverse = np.ones(n, dtype=bool)
    else:
        reverse = np.random.default_rng(cfg.seed).random(n) < 0.5
    return Timeline(start, duration, reverse, cfg.fps, paint_frames, hold_frames, cfg.stroke_ease)
