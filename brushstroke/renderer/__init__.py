"""Differentiable textured stroke renderer (brushes, canvas, parameters, compositing)."""

from brushstroke.renderer.brushes import BRUSH_KINDS, BrushBank, BrushConfig, make_brush
from brushstroke.renderer.canvas import CanvasConfig, apply_canvas, make_canvas_texture
from brushstroke.renderer.compositing import composite_parallel, composite_sequential
from brushstroke.renderer.params import (
    NUM_PARAMS,
    canonicalize,
    clamp_params,
    params_from_unconstrained,
    unconstrained_from_params,
)
from brushstroke.renderer.pipeline import RenderConfig, make_canvas, render, render_progressive
from brushstroke.renderer.rasterize import render_alphas

__all__ = [
    "BRUSH_KINDS",
    "NUM_PARAMS",
    "BrushBank",
    "BrushConfig",
    "CanvasConfig",
    "RenderConfig",
    "apply_canvas",
    "canonicalize",
    "clamp_params",
    "composite_parallel",
    "composite_sequential",
    "make_brush",
    "make_canvas",
    "make_canvas_texture",
    "params_from_unconstrained",
    "render",
    "render_alphas",
    "render_progressive",
    "unconstrained_from_params",
]
