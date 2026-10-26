"""Anytime curves and strokes-to-target (T-026): ``docs/design/evaluation.md`` sections 3 and 5.

A curve is a pair of arrays ``(n, q)``: ``n`` the stroke counts of the checkpoints (ascending, starting at 0) and
``q`` the metric (lower is better) of the canvas after the first ``n`` strokes. Between checkpoints the curve is linear in
``(n, q)``; nothing is smoothed or fitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _arrays(ns: Sequence[float], q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(ns, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("a curve needs two or more checkpoints and one value per checkpoint")
    if np.any(np.diff(x) <= 0):
        raise ValueError("checkpoints must be strictly ascending")
    return x, y


def reaches(ns: Sequence[float], budget: float) -> bool:
    """True if the producer has a canvas at ``budget`` strokes (it emitted at least that many)."""
    return budget in {float(n) for n in ns}


def _up_to(ns: Sequence[float], q: Sequence[float], budget: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = _arrays(ns, q)
    if budget not in set(x.tolist()):
        raise ValueError(f"budget {budget} is not a checkpoint of the curve")
    keep = x <= budget
    return x[keep], y[keep]


def auc_raw(ns: Sequence[float], q: Sequence[float], budget: float) -> float:
    """``integral_0^1 q(B u) du`` by the trapezoid rule on the checkpoints ``n <= B`` (section 3.2, raw).

    The x-axis is the stroke count normalized by the budget, ``u = n / B``.
    """
    x, y = _up_to(ns, q, budget)
    return float(np.trapezoid(y, x / budget))


def auc_normalized(
    ns: Sequence[float], q: Sequence[float], budget: float, reference: float
) -> float:
    """``AUC@B`` of section 3.2: the raw integral divided by the image's blank-canvas reference value."""
    if reference <= 0:
        raise ValueError("the reference value must be positive")
    return auc_raw(ns, q, budget) / reference


def auc_log(
    ns: Sequence[float], q: Sequence[float], budget: float, start: float, reference: float
) -> float:
    """Log-spaced variant: the mean of ``q`` over ``log n`` for ``n`` in ``[start, B]``, divided by the reference.

    Trapezoid in ``log n``, divided by ``log B - log start`` so that it is an average (comparable to the linear AUC).
    Weights the early strokes, which the linear AUC hides behind the long plateau at the end of the curve.
    """
    x, y = _up_to(ns, q, budget)
    keep = x >= start
    if keep.sum() < 2:
        raise ValueError(f"fewer than two checkpoints in [{start}, {budget}]")
    lx = np.log(x[keep])
    return float(np.trapezoid(y[keep], lx) / (math.log(budget) - math.log(start)) / reference)


def auc_late(
    ns: Sequence[float], q: Sequence[float], budget: float, start: float, reference: float
) -> float:
    """Robustness variant of section 3.3: the mean of ``q`` over ``n`` in ``[start, B]``, divided by the reference.

    Trapezoid in ``n`` divided by ``B - start``: the part of the curve after the blank canvas has mostly been covered.
    """
    x, y = _up_to(ns, q, budget)
    keep = x >= start
    if keep.sum() < 2:
        raise ValueError(f"fewer than two checkpoints in [{start}, {budget}]")
    return float(np.trapezoid(y[keep], x[keep]) / (budget - start) / reference)


@dataclass(frozen=True)
class StrokesToTarget:
    """Result of :func:`strokes_to_target`.

    ``n_star`` is ``max_n`` for a censored image (the curve never reaches the target within the budget); ``censored`` tells
    it apart from a genuine crossing exactly at ``max_n``.
    """

    n_star: float
    censored: bool


def strokes_to_target(
    ns: Sequence[float], q: Sequence[float], tau: float, max_n: float
) -> StrokesToTarget:
    """The smallest stroke count at which the curve reaches ``tau`` (section 5).

    The first crossing counts (the curve is not assumed monotone): linear interpolation between the two bracketing
    checkpoints. A curve that starts at or below ``tau`` gives 0. A curve that never reaches ``tau`` for ``n <= max_n`` is
    right-censored at ``max_n`` (a conservative lower bound; paired tests treat it as a tie at ``max_n``).
    """
    x, y = _arrays(ns, q)
    keep = x <= max_n
    x, y = x[keep], y[keep]
    if y[0] <= tau:
        return StrokesToTarget(float(x[0]), False)
    for i in range(1, len(x)):
        if y[i] <= tau:
            frac = (y[i - 1] - tau) / (y[i - 1] - y[i])
            return StrokesToTarget(float(x[i - 1] + frac * (x[i] - x[i - 1])), False)
    return StrokesToTarget(float(max_n), True)


def value_at(ns: Sequence[float], q: Sequence[float], n: float) -> float:
    """The curve at a checkpoint ``n`` (exact; the metrics at the reporting budgets are checkpoints)."""
    x, y = _arrays(ns, q)
    hit = np.nonzero(x == n)[0]
    if not len(hit):
        raise ValueError(f"{n} is not a checkpoint of the curve")
    return float(y[hit[0]])
