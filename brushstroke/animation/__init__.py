"""Stroke-by-stroke animation: timeline, partial reveal, frame rendering and ffmpeg export."""

from brushstroke.animation.config import AnimationConfig
from brushstroke.animation.export import export_frames, find_ffmpeg
from brushstroke.animation.frames import FrameRenderer
from brushstroke.animation.pt_frames import PTFrameRenderer, is_pt_sequence, make_frame_renderer
from brushstroke.animation.timeline import Timeline, build_timeline, num_frames

__all__ = [
    "AnimationConfig",
    "FrameRenderer",
    "PTFrameRenderer",
    "Timeline",
    "build_timeline",
    "export_frames",
    "find_ffmpeg",
    "is_pt_sequence",
    "make_frame_renderer",
    "num_frames",
]
