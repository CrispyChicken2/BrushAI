"""Tests of the protocol statistics (T-026): Holm, Wilcoxon, effect size, bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from brushstroke.eval import stats as st


def test_holm_on_known_p_values() -> None:
    # sorted 0.005, 0.01, 0.03, 0.04 with m = 4 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    adjusted = st.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert st.holm([0.5]) == [0.5]
    assert st.holm([0.6, 0.7, 0.9]) == pytest.approx([1.0, 1.0, 1.0])  # capped at 1
    assert st.holm([0.001, 0.001]) == pytest.approx([0.002, 0.002])
    assert st.holm([]) == []


def test_holm_never_lowers_a_p_value_and_keeps_the_order() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, 9).tolist()
    adj = st.holm(p)
    assert all(a >= q for a, q in zip(adj, p, strict=True))
    order = np.argsort(p)
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(8))


def test_rank_biserial_extremes_and_a_known_value() -> None:
    assert st.rank_biserial([0.1, 0.2, 0.3]) == 1.0
    assert st.rank_biserial([-0.1, -0.2]) == -1.0
    assert st.rank_biserial([0.0, 0.0]) == 0.0
    assert st.rank_biserial([1.0, -1.0]) == 0.0  # equal ranks
    # ranks of |d| = 1, 2, 3, 4; positives {3, 4}: W+ = 7, W- = 3 -> r = 0.4
    assert st.rank_biserial([-1.0, -2.0, 3.0, 4.0]) == pytest.approx(0.4)


def test_the_paired_test_agrees_with_scipy_and_reports_the_effect() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.05, 40)
    b = a + rng.normal(0.03, 0.02, 40)  # b is higher: a is better (lower)
    res = st.paired_test(a, b)
    ref = stats.wilcoxon(a - b, zero_method="wilcox", alternative="two-sided")
    assert res.p == pytest.approx(ref.pvalue) and res.p < 1e-4
    assert res.r < -0.9 and res.median_diff < 0 and res.ci_high < 0
    assert res.effect_ok(0.3) and res.n == res.n_nonzero == 40


def test_ties_and_identical_samples_are_handled_conservatively() -> None:
    same = st.paired_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert same.p == 1.0 and same.n_nonzero == 0 and same.r == 0.0 and same.effect_ok() is False
    with pytest.raises(ValueError):
        st.paired_test([1.0], [1.0, 2.0])
    noise = st.paired_test(np.zeros(30), np.tile([0.1, -0.1], 15))
    assert noise.effect_ok() is False  # symmetric differences: no effect


def test_the_bootstrap_interval_is_deterministic_and_covers_the_mean() -> None:
    x = np.random.default_rng(3).normal(10.0, 1.0, 90)
    lo, hi = st.bootstrap_ci(x, "mean")
    assert (lo, hi) == st.bootstrap_ci(x, "mean") and lo < x.mean() < hi
    assert hi - lo < 1.0
    assert st.bootstrap_ci(x, "mean", seed=1) != (lo, hi)
    mlo, mhi = st.bootstrap_ci(x, "median")
    assert mlo < np.median(x) < mhi
    with pytest.raises(ValueError):
        st.bootstrap_ci(x, "mode")
    with pytest.raises(ValueError):
        st.bootstrap_ci([], "mean")


def test_describe_keeps_the_replicate_noise_apart_from_the_uncertainty_of_the_mean() -> None:
    means = [0.40, 0.42, 0.38, 0.45, 0.41]
    d = st.describe(means, [0.01, 0.02, 0.0, 0.03, 0.01])
    assert d.mean == pytest.approx(0.412) and d.replicate_noise == pytest.approx(0.014) and d.n == 5
    assert d.ci_low < d.mean < d.ci_high and d.std == pytest.approx(np.std(means, ddof=1))
    assert st.describe([0.5]).std == 0.0
    assert st.per_image_summary([0.2, 0.4]) == pytest.approx((0.3, np.std([0.2, 0.4], ddof=1)))
    assert st.per_image_summary([0.2]) == (0.2, 0.0)
