"""Statistics of the evaluation protocol (T-026): ``docs/design/evaluation.md`` section 6.

* the unit of analysis is the image: per-image values are means over replicates, tests and intervals are over images;
* descriptive: mean, std over images, mean of the per-image std over replicates (the replicate noise), a 95 % bootstrap
  interval of the mean (10,000 resamples of the images, percentile method, seed 22; Efron, 1979);
* "A beats B": a two-sided Wilcoxon signed-rank test (Wilcoxon, 1945) on the paired per-image differences, with the
  matched-pairs rank-biserial correlation (Kerby, 2014), the median paired difference and its bootstrap interval;
* Holm's step-down correction (Holm, 1979) inside the fixed families.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    statistic: str = "mean",
    resamples: int = 10_000,
    seed: int = 22,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval of the ``mean`` or the ``median`` over resamples of the images (rows are resampled
    together, so a paired difference is resampled as one unit)."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("bootstrap needs a non-empty 1-d sample")
    if statistic not in ("mean", "median"):
        raise ValueError(f"statistic must be 'mean' or 'median', got {statistic!r}")
    idx = np.random.default_rng(seed).integers(0, len(x), size=(resamples, len(x)))
    stat = (np.mean if statistic == "mean" else np.median)(x[idx], axis=1)
    lo, hi = np.quantile(stat, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(lo), float(hi)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the order given (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p[i])))
        adjusted[i] = running
    return [float(a) for a in adjusted]


def rank_biserial(differences: Sequence[float] | np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014): ``(W+ - W-) / (W+ + W-)`` over the non-zero differences.

    +1 if every difference is positive, -1 if every one is negative, 0 if the rank sums are equal (or all differences are zero).
    """
    d = np.asarray(differences, dtype=np.float64)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    return (w_pos - w_neg) / (w_pos + w_neg)


@dataclass(frozen=True)
class PairedResult:
    """``a - b`` over the paired images (a lower metric value is better)."""

    n: int
    n_nonzero: int
    p: float
    r: float  # rank-biserial correlation; negative = a lower than b
    median_diff: float
    ci_low: float
    ci_high: float

    def effect_ok(self, threshold: float = 0.3) -> bool:
        """The interval of the median difference excludes 0 and ``|r| >= threshold`` (the p-value is separate, Holm-adjusted)."""
        return (self.ci_low > 0 or self.ci_high < 0) and abs(self.r) >= threshold


def paired_test(
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10_000,
    seed: int = 22,
    confidence: float = 0.95,
) -> PairedResult:
    """Two-sided Wilcoxon signed-rank test on ``a - b`` (zero differences dropped), effect size and bootstrap interval."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) == 0:
        raise ValueError("paired samples must be non-empty 1-d arrays of equal length")
    d = x - y
    nonzero = int((d != 0).sum())
    p = (
        1.0
        if nonzero == 0
        else float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    )
    lo, hi = bootstrap_ci(d, "median", resamples, seed, confidence)
    return PairedResult(len(d), nonzero, p, rank_biserial(d), float(np.median(d)), lo, hi)


@dataclass(frozen=True)
class Description:
    mean: float
    std: float  # over images
    ci_low: float
    ci_high: float  # bootstrap interval of the mean
    replicate_noise: float  # mean over images of the per-image std over replicates
    n: int


def describe(
    per_image_mean: Sequence[float],
    per_image_replicate_std: Sequence[float] | None = None,
    resamples: int = 10_000,
    seed: int = 22,
    confidence: float = 0.95,
) -> Description:
    """Descriptive statistics over images (section 6.3). The seed spread is reported separately, never as the uncertainty of the mean."""
    x = np.asarray(per_image_mean, dtype=np.float64)
    lo, hi = bootstrap_ci(x, "mean", resamples, seed, confidence)
    noise = (
        float(np.mean(per_image_replicate_std))
        if per_image_replicate_std is not None and len(per_image_replicate_std)
        else 0.0
    )
    return Description(
        float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, lo, hi, noise, len(x)
    )


def per_image_summary(values_by_replicate: Sequence[float]) -> tuple[float, float]:
    """``(mean over replicates, std over replicates)`` of one image; the std is 0 for a single replicate."""
    v = np.asarray(values_by_replicate, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
