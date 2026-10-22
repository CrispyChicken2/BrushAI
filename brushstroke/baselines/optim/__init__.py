"""Baseline (a): optimization-only progressive stroke decomposition (SNP style)."""

from brushstroke.baselines.optim.config import LossConfig, OptimConfig, PaintOptimConfig
from brushstroke.baselines.optim.optimize import paint_optim

__all__ = ["LossConfig", "OptimConfig", "PaintOptimConfig", "paint_optim"]
