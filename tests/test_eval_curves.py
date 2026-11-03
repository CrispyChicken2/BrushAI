"""Tests of the anytime curves and strokes-to-target (T-026): known synthetic curves."""

from __future__ import annotations

import pytest

from brushstroke.eval import curves as cv

GRID = [0, 25, 50, 100, 150, 200, 250, 350, 500, 750, 1000, 1250, 1500, 1750, 2000]


def test_the_auc_of_constant_and_linear_curves_is_known() -> None:
    assert cv.auc_raw(GRID, [0.8] * 15, 2000) == pytest.approx(0.8)
    linear = [1.0 - n / 2000 for n in GRID]  # q goes 1 -> 0
    assert cv.auc_raw(GRID, linear, 2000) == pytest.approx(0.5)
    assert cv.auc_raw(GRID, linear, 1000) == pytest.approx(
        0.75
    )  # checkpoints up to 1000, u = n / 1000
    assert cv.auc_normalized(GRID, linear, 2000, reference=2.0) == pytest.approx(0.25)


def test_a_method_that_never_improves_on_the_blank_canvas_scores_one() -> None:
    assert cv.auc_normalized(GRID, [1.1] * 15, 2000, reference=1.1) == pytest.approx(1.0)
    decreasing = [1.1 * (1 - n / 2000) for n in GRID]
    assert 0.0 <= cv.auc_normalized(GRID, decreasing, 500, 1.1) <= 1.0


def test_the_budget_must_be_a_checkpoint_and_curves_must_be_valid() -> None:
    with pytest.raises(ValueError, match="not a checkpoint"):
        cv.auc_raw(GRID, [0.5] * 15, 300)
    with pytest.raises(ValueError, match="ascending"):
        cv.auc_raw([0, 50, 25], [1, 1, 1], 50)
    with pytest.raises(ValueError):
        cv.auc_raw([0], [1.0], 0)
    with pytest.raises(ValueError, match="positive"):
        cv.auc_normalized(GRID, [0.5] * 15, 2000, reference=0.0)


def test_the_log_and_late_variants_are_averages_with_the_defined_weights() -> None:
    # a constant curve gives that constant (divided by the reference) whatever the weighting
    assert cv.auc_log(GRID, [0.6] * 15, 2000, 25, reference=1.2) == pytest.approx(0.5)
    assert cv.auc_late(GRID, [0.6] * 15, 2000, 250, reference=1.2) == pytest.approx(0.5)
    # high early, low late: the log average weights the early checkpoints more
    early_high = [1.0 if n <= 100 else 0.1 for n in GRID]
    assert cv.auc_log(GRID, early_high, 2000, 25, 1.0) > cv.auc_late(
        GRID, early_high, 2000, 250, 1.0
    )
    # the late variant ignores everything before its start
    assert cv.auc_late(GRID, early_high, 2000, 250, 1.0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="fewer than two"):
        cv.auc_late(GRID, [0.5] * 15, 250, 250, 1.0)


def test_strokes_to_target_interpolates_the_first_crossing() -> None:
    q = [1.0, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    r = cv.strokes_to_target(
        GRID, q, tau=0.7, max_n=2000
    )  # between n=50 (0.8) and n=100 (0.6): halfway
    assert r.censored is False and r.n_star == pytest.approx(75.0)
    assert cv.strokes_to_target(GRID, q, tau=0.9, max_n=2000).n_star == pytest.approx(
        25.0
    )  # at a checkpoint
    assert cv.strokes_to_target(GRID, q, tau=1.5, max_n=2000) == cv.StrokesToTarget(
        0.0, False
    )  # already below


def test_a_curve_that_never_reaches_tau_is_right_censored_at_the_budget() -> None:
    q = [1.0 - 0.0001 * n for n in GRID]  # best value 0.8
    r = cv.strokes_to_target(GRID, q, tau=0.5, max_n=2000)
    assert r.censored is True and r.n_star == 2000.0
    # a crossing exactly at the last checkpoint is a crossing, not a censoring
    last = cv.strokes_to_target([0, 1000, 2000], [1.0, 0.9, 0.5], tau=0.5, max_n=2000)
    assert last.censored is False and last.n_star == pytest.approx(2000.0)
    # only checkpoints within max_n are used
    assert cv.strokes_to_target(GRID, [0.9] * 14 + [0.1], tau=0.5, max_n=1750).censored is True


def test_the_first_crossing_counts_when_the_curve_is_not_monotone() -> None:
    q = [1.0, 0.4, 0.8, 0.3]
    r = cv.strokes_to_target([0, 100, 200, 300], q, tau=0.5, max_n=300)
    assert r.n_star == pytest.approx(100 * (1.0 - 0.5) / (1.0 - 0.4))


def test_reaching_a_budget_and_reading_a_checkpoint() -> None:
    assert cv.reaches([0, 25, 250], 250)
    assert not cv.reaches([0, 25, 100], 250) and not cv.reaches([], 250)
    assert cv.value_at(GRID, list(range(15)), 250) == 6.0
    with pytest.raises(ValueError):
        cv.value_at(GRID, [0.0] * 15, 300)
