"""Typed configuration of the optimization-only baseline (``configs/baseline_optim.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from brushstroke.renderer.brushes import BrushConfig
from brushstroke.renderer.pipeline import RenderConfig


@dataclass
class LossConfig:
    """Loss terms and weights (a weight of 0 disables the term)."""

    l1: float = 1.0  # mean absolute pixel error
    lpips: float = 0.1  # LPIPS-VGG (Zhang et al., CVPR 2018); 0 = off
    ot: float = 0.0  # Sinkhorn divergence on downsampled intensities (SNP uses 0.1 with L1 = 1)
    ot_size: int = 24  # the images are average-pooled to ot_size x ot_size
    ot_eps: float = 0.02  # entropic regularization (cost = squared distance in the unit square)
    ot_iters: int = 30  # Sinkhorn iterations


@dataclass
class OptimConfig:
    """Progressive coarse-to-fine optimization settings."""

    height: int = 512
    width: int = 512
    # grid m x m per level, and the new strokes per patch at each level (2037 strokes in total)
    levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 20])
    strokes_per_patch: list[int] = field(default_factory=lambda: [1, 1, 1, 3, 4, 2])
    iters: int = 100  # optimization steps per level when iters_per_level is empty
    # steps per level (one entry per level; empty = `iters` everywhere): most steps at fine levels
    iters_per_level: list[int] = field(default_factory=lambda: [100, 100, 100, 300, 300, 150])
    optimizer: str = "adam"  # adam | rmsprop (as in SNP)
    lr: float = 0.02  # on the unconstrained (logit) parameters
    lr_final_frac: float = 0.1  # lr decays (cosine) to lr * lr_final_frac within each level
    finetune_all: bool = True  # also optimize the strokes of earlier levels at every level
    finetune_lr_factor: float = 1.0  # lr multiplier of the earlier strokes when finetune_all
    background: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])  # initial canvas RGB
    # initialization of the new strokes of a level
    init_error_power: float = 2.0  # position sampling weights = blurred |target - canvas| ** power
    init_blur_frac: float = 0.25  # blur sigma of the error map, as a fraction of the patch size
    init_size_scale: float = 1.0  # stroke length ~ U(0.6, 1.2) * scale / grid
    init_aspect: list[float] = field(
        default_factory=lambda: [0.4, 1.0]
    )  # thickness / length ~ U(lo, hi)
    init_opacity: float = 0.9
    init_mode: str = "error"  # error: positions from the residual, colours from the target | random
    init_size_mode: str = "patch"  # patch: length ~ 1/g | density: ~ 1/(g sqrt(strokes per patch))
    # re-initialize dead strokes from the current residual every `resample_every` steps (0 = off)
    resample_every: int = 0
    dead_opacity: float = 0.05  # a stroke is dead if its opacity is below this ...
    dead_area: float = 1e-4  # ... or its box area (w * h, canvas fractions) is below this
    final_pass_iters: int = 0  # joint fine-tuning of all strokes after the last level (0 = off)
    final_pass_lr_factor: float = 0.3  # lr of the final pass relative to `lr`
    checkpoint_bytes: float = 6e9  # use gradient checkpointing above this estimated activation size


@dataclass
class PaintOptimConfig:
    """Root config of ``scripts/paint_optim.py``."""

    seed: int = 0
    deterministic: bool = False
    device: str = "auto"
    output_dir: str = "outputs/optim"
    brush: BrushConfig = field(default_factory=BrushConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
