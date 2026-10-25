"""Equivalence classes of the Paint Transformer stroke parameters ``(w, h, theta)`` (T-019).

The renderer of :mod:`brushstroke.renderer.pt` (official code, ``docs/design/paint_transformer.md``
section 1) paints the brush image ``B`` at ``B(u, v)`` with
``(u, v) = diag(1/w, 1/h) R(-pi theta) (p - c)`` (up to the aspect-ratio factors of the canvas), and
the network is free to predict any sign for ``w`` and ``h`` and any ``theta``. Several parameter
triples therefore paint (almost) the same stroke. Derivation (full note:
``docs/design/pt_convention.md``):

* ``theta`` enters through ``cos(pi theta)`` and ``sin(pi theta)`` only: **exact** period 2.
* ``(w, h, theta + 1)`` and ``(-w, -h, theta)`` are the same warp (``R(-pi theta - pi) =
  -R(-pi theta)`` and ``diag(1/w, 1/h)`` is odd in both): **exact** for the alpha map. The
  foreground differs by the brush texture only when the choice of brush image changes (``h > w``
  flips with the signs: 0.003 mean absolute difference); the two brush masks of the training images
  are identical. This is the *exact class*.
* The brush mask is *nearly* symmetric about its horizontal axis: ``h -> -h`` (hence ``w -> -w``
  with ``theta + 1``) renders a stroke that differs on 14 % of its area, half the difference of a
  rotation by pi. This is the *mirror class*: **approximate**, not an exact symmetry of the renderer
  (the ``|h|`` / ``|w|, |h|`` fold of T-016 is this class without the ``theta`` shift).
* Larger approximate classes (rotation by pi of the stroke, then exchange of ``w`` and ``h`` with a
  rotation by pi/2) differ on about 30 % of the area (two unrelated strokes of the same distribution
  differ by 157 % of it, since they barely overlap): they are only used as diagnostics.

``theta`` is in units of pi, as everywhere in :mod:`brushstroke.renderer.pt`.
"""

from __future__ import annotations

import itertools

import torch

LEVELS = ("none", "exact", "mirror", "rot180", "swap")
THETA_PERIOD = 2.0  # exact period of theta (units of pi)


def _signed(
    shape: torch.Tensor, sign_w: float, sign_h: float, shift: float, swap: bool
) -> torch.Tensor:
    """One variant of ``(..., >= 5)`` strokes: ``(x, y, +-w, +-h, theta + shift)`` (``w`` and ``h``
    exchanged first if ``swap``); the result is ``(..., 5)``."""
    x, y, w, h, theta = (shape[..., i] for i in range(5))
    if swap:
        w, h = h, w
    return torch.stack([x, y, sign_w * w, sign_h * h, theta + shift], dim=-1)


def variant_specs(level: str) -> list[tuple[float, float, float, bool]]:
    """``(sign_w, sign_h, theta_shift, swap)`` of the parameter variants of ``level`` (the identity
    first).

    ``none``: identity only; ``exact``: identity and ``(-w, -h, theta + 1)``; ``mirror``: ``exact``
    and ``(w, -h, theta)``, ``(-w, h, theta + 1)``; ``rot180``: every sign pair with ``theta`` and
    ``theta + 1``; ``swap``: ``rot180`` and the exchange of ``w`` and ``h`` with a rotation of the
    stroke by +-pi/2 (``theta +- 1/2``). Each level is closed under composition (a group acting on
    the strokes).
    """
    if level == "none":
        return [(1.0, 1.0, 0.0, False)]
    if level == "exact":
        return [(1.0, 1.0, 0.0, False), (-1.0, -1.0, 1.0, False)]
    if level == "mirror":
        return [*variant_specs("exact"), (1.0, -1.0, 0.0, False), (-1.0, 1.0, 1.0, False)]
    if level == "rot180":
        pairs = list(itertools.product((1.0, -1.0), (1.0, -1.0), (0.0, 1.0)))
        pairs.sort(key=lambda t: t != (1.0, 1.0, 0.0))  # identity first
        return [(sw, sh, k, False) for sw, sh, k in pairs]
    if level == "swap":
        specs = variant_specs("rot180")
        specs += [
            (sw, sh, 0.5 * k, True)
            for sw, sh, k in itertools.product((1.0, -1.0), (1.0, -1.0), (1, 3))
        ]
        return specs
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


def variants(shape: torch.Tensor, level: str) -> torch.Tensor:
    """The parameter variants of ``level`` of ``(..., >= 5)`` strokes: ``(V, ..., 5)``."""
    return torch.stack([_signed(shape, *spec) for spec in variant_specs(level)])


def wrap_theta(delta: torch.Tensor) -> torch.Tensor:
    """A ``theta`` difference reduced modulo the exact period 2 into ``[-1, 1)``."""
    return torch.remainder(delta + THETA_PERIOD / 2, THETA_PERIOD) - THETA_PERIOD / 2


def _class_deltas(shape: torch.Tensor, gt: torch.Tensor, level: str) -> torch.Tensor:
    """``(V, ..., 5)`` differences of the variants of ``shape`` to ``gt``, ``theta`` modulo 2."""
    delta = variants(shape, level) - gt[..., :5]
    return torch.cat([delta[..., :4], wrap_theta(delta[..., 4:5])], dim=-1)


def stroke_l1_sum(pred: torch.Tensor, gt: torch.Tensor, level: str = "none") -> torch.Tensor:
    """Parameter L1 summed over the parameters, minimum over the equivalence class of ``pred``.

    ``pred`` and ``gt``: broadcastable ``(..., >= 5)``. The first five parameters are compared as
    ``(x, y, w, h, theta)`` and any further parameter (colour) with a plain L1 (it is not part of
    the convention). With ``level="none"`` this is exactly ``(pred - gt).abs().sum(-1)`` (the
    official distance, no wrap of ``theta``); otherwise ``theta`` differences are taken modulo its
    exact period.

    Returns:
        ``(...)`` distances, ``>= 0``, equal for the strokes of one class of ``pred``.
    """
    if level == "none":
        return (pred - gt).abs().sum(dim=-1)
    best = _class_deltas(pred, gt, level).abs().sum(dim=-1).min(dim=0).values
    if pred.shape[-1] > 5:
        best = best + (pred[..., 5:] - gt[..., 5:]).abs().sum(dim=-1)
    return best


def class_indicator(shape: torch.Tensor, gt: torch.Tensor, level: str) -> torch.Tensor:
    """Index of the variant of ``level`` of ``shape`` nearest to ``gt`` (L1 on the five shape
    parameters, ``theta`` modulo 2): ``(...)`` long, 0 = the stroke as predicted."""
    return _class_deltas(shape, gt, level).abs().sum(dim=-1).argmin(dim=0)
