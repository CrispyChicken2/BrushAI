"""Typed configuration of the animation (``configs/animation.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnimationConfig:
    """Timeline, reveal, canvas and export settings."""

    # timeline
    fps: int = 30
    duration: float = 10.0  # seconds of painting; the last stroke ends exactly at the end of it
    hold: float = 1.5  # seconds the finished painting stays on screen
    overlap: float = 3.0  # average number of strokes in flight (reveal time = overlap x time slot)
    min_reveal_frames: float = 2.0  # a stroke is revealed over at least this many frames (0 = off)
    easing: str = "uniform"  # time slots per stroke: uniform | decelerate | by_level
    level_time_power: float = 0.5  # by_level: time of a level ~ strokes_in_level ** power
    decel_ratio: float = (
        3.0  # decelerate: the last stroke gets this many times the first one's slot
    )
    stroke_ease: str = "smoothstep"  # reveal progress inside a stroke: linear | smoothstep
    # partial reveal
    reveal_edge: float = 0.15  # soft front width, fraction of the stroke length
    direction: str = "random"  # forward | reverse | random (seeded per stroke)
    seed: int = 0
    # rendering
    height: int = 0  # output size in pixels; 0 = the sequence's canvas size
    width: int = 0
    device: str = "auto"
    chunk_size: int = 32  # strokes composited at once per frame
    canvas_enabled: bool = True
    canvas_strength: float = 0.06  # shading depth of the canvas texture (apply_canvas)
    canvas_seed: int = 0
    weave_period: float = 6.0
    low_freq_amount: float = 0.2
    low_freq_scale: float = 48.0
    # export
    formats: list[str] = field(default_factory=lambda: ["mp4"])  # mp4 | webm | gif
    ffmpeg: str = "ffmpeg"
    mp4_crf: int = 18
    webm_crf: int = 32
    gif_width: int = 480  # max width of the GIF (never upscaled)
    gif_fps: int = 15
    gif_dither: str = "bayer"  # none | bayer | sierra2_4a | floyd_steinberg
