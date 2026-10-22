"""Root run configuration (``configs/default.yaml``). Sections are added by later tickets."""

from __future__ import annotations

from dataclasses import dataclass, field

from brushstroke.renderer.brushes import BrushConfig
from brushstroke.renderer.canvas import CanvasConfig


@dataclass
class RunConfig:
    """Top-level run settings."""

    seed: int = 0
    deterministic: bool = False
    device: str = "auto"
    output_dir: str = "outputs/default"
    brush: BrushConfig = field(default_factory=BrushConfig)
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
