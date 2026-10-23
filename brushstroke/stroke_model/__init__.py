"""Stroke predictor components (Paint Transformer, M2)."""

from brushstroke.stroke_model.losses import (
    LossOutput,
    PaintTransformerLoss,
    PTLossConfig,
    gaussian_w_distance,
    resolve_loss_config,
)
from brushstroke.stroke_model.model import (
    PaintTransformer,
    PTModelConfig,
    count_parameters,
    init_official_weights,
)
from brushstroke.stroke_model.synthetic import (
    PatchBatch,
    SyntheticConfig,
    SyntheticStrokeGenerator,
)

__all__ = [
    "LossOutput",
    "PTLossConfig",
    "PTModelConfig",
    "PaintTransformer",
    "PaintTransformerLoss",
    "PatchBatch",
    "SyntheticConfig",
    "SyntheticStrokeGenerator",
    "count_parameters",
    "gaussian_w_distance",
    "init_official_weights",
    "resolve_loss_config",
]
