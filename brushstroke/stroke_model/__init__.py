"""Stroke predictor components (Paint Transformer, M2)."""

from brushstroke.stroke_model.inference import (
    PaintPTConfig,
    PTInferenceConfig,
    PTResult,
    load_pt_model,
    paint,
    paint_layer,
)
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
from brushstroke.stroke_model.pt_sequence import (
    LayerStrokes,
    layers_to_sequence,
    replay_sequence,
)
from brushstroke.stroke_model.refine import RefineConfig, refine_patch_strokes
from brushstroke.stroke_model.synthetic import (
    PatchBatch,
    SyntheticConfig,
    SyntheticStrokeGenerator,
)

__all__ = [
    "LayerStrokes",
    "LossOutput",
    "PTInferenceConfig",
    "PTLossConfig",
    "PTModelConfig",
    "PTResult",
    "PaintTransformer",
    "PaintTransformerLoss",
    "PaintPTConfig",
    "PatchBatch",
    "RefineConfig",
    "SyntheticConfig",
    "SyntheticStrokeGenerator",
    "count_parameters",
    "gaussian_w_distance",
    "init_official_weights",
    "layers_to_sequence",
    "load_pt_model",
    "paint",
    "paint_layer",
    "refine_patch_strokes",
    "replay_sequence",
    "resolve_loss_config",
]
