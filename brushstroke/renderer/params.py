"""Stroke parameter convention and re-parameterization helpers.

A set of strokes is a float tensor ``(B, N, 9)`` (batch of B paintings, N strokes each, in painting
order: stroke ``k`` is composited over strokes ``0..k-1``). The last dimension is, in order::

    index  name   range     meaning
    0      x      [0, 1]    stroke centre, horizontal, fraction of the canvas width (0 = left edge)
    1      y      [0, 1]    stroke centre, vertical, fraction of the canvas height (0 = top edge)
    2      w      (0, 1]    box size along the stroke's local x axis, fraction of the canvas width
    3      h      (0, 1]    box size along the stroke's local y axis, fraction of the canvas height
    4      theta  [0, 1]    rotation, mapped to [0, pi) radians (periodic: 0 and 1 are the same)
    5-7    r,g,b  [0, 1]    stroke colour
    8      a      [0, 1]    stroke opacity (Paint Transformer strokes have no such parameter: a = 1)

This is the layout of the Paint Transformer renderer (Liu et al., ICCV 2021,
``inference/inference.py::param2stroke``: ``x0, y0, w, h, theta, R, G, B``) plus a trailing opacity.
``w`` and ``h`` are relative to the canvas width and height respectively, exactly as there.
"""

from __future__ import annotations

import torch

NUM_PARAMS = 9
X, Y, W, H, THETA, R, G, B, A = range(NUM_PARAMS)
PARAM_NAMES = ("x", "y", "w", "h", "theta", "r", "g", "b", "a")


def check_params(params: torch.Tensor) -> None:
    if params.ndim != 3 or params.shape[-1] != NUM_PARAMS:
        raise ValueError(
            f"expected stroke params of shape (B, N, {NUM_PARAMS}), got {tuple(params.shape)}"
        )


def clamp_params(params: torch.Tensor, min_size: float = 1e-3) -> torch.Tensor:
    """Clamp stroke parameters to their valid ranges (``w, h`` to ``[min_size, 1]``). Differentiable
    (gradient is zero for clamped entries)."""
    check_params(params)
    lo = torch.zeros(NUM_PARAMS, dtype=params.dtype, device=params.device)
    lo[[W, H]] = min_size
    hi = torch.ones(NUM_PARAMS, dtype=params.dtype, device=params.device)
    return torch.clamp(params, min=lo, max=hi)


def params_from_unconstrained(raw: torch.Tensor, min_size: float = 1e-3) -> torch.Tensor:
    """Map unconstrained values ``(B, N, 9)`` to valid parameters with a sigmoid.

    Every entry goes through ``sigmoid``; ``w, h`` are then rescaled to ``(min_size, 1)``. Use for
    gradient-based optimization (T-005). Inverse: :func:`unconstrained_from_params`.
    """
    check_params(raw)
    p = torch.sigmoid(raw)
    scale = torch.ones(NUM_PARAMS, dtype=raw.dtype, device=raw.device)
    offset = torch.zeros(NUM_PARAMS, dtype=raw.dtype, device=raw.device)
    scale[[W, H]] = 1.0 - min_size
    offset[[W, H]] = min_size
    return p * scale + offset


def unconstrained_from_params(
    params: torch.Tensor, min_size: float = 1e-3, eps: float = 1e-4
) -> torch.Tensor:
    """Inverse of :func:`params_from_unconstrained` (logit); inputs are pulled ``eps`` inside."""
    check_params(params)
    scale = torch.ones(NUM_PARAMS, dtype=params.dtype, device=params.device)
    offset = torch.zeros(NUM_PARAMS, dtype=params.dtype, device=params.device)
    scale[[W, H]] = 1.0 - min_size
    offset[[W, H]] = min_size
    p = ((params - offset) / scale).clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def canonicalize(params: torch.Tensor, height: int = 1, width: int = 1) -> torch.Tensor:
    """Return params where every stroke is at least as long as thick, in pixels (``w W >= h H``).

    If ``h H > w W``, the pixel sizes are swapped (``w' = h H / W``, ``h' = w W / H``) and half a
    turn (theta + 0.5, i.e. 90 degrees) is added modulo 1. The geometric footprint is unchanged,
    but the brush texture (main axis = local x) then always follows the longer side, like the
    vertical/horizontal brush selection of Paint Transformer and Stylized Neural Painting. On a
    square canvas this is the plain ``h > w`` swap; on other canvases the sizes are compared and
    swapped in pixels because ``w`` and ``h`` are fractions of different canvas sides. The returned
    ``w'`` may exceed 1 (it is only used internally for rasterization).
    """
    check_params(params)
    w, h, theta = params[..., W], params[..., H], params[..., THETA]
    swap = h * height > w * width
    new_w = torch.where(swap, h * height / width, w)
    new_h = torch.where(swap, w * width / height, h)
    new_theta = torch.where(swap, torch.remainder(theta + 0.5, 1.0), theta)
    out = params.clone()
    out[..., W], out[..., H], out[..., THETA] = new_w, new_h, new_theta
    return out
