"""Tests of the report logic of the convention experiment (T-019): rules on a synthetic store."""

from __future__ import annotations

from typing import Any

import pytest

from scripts import exp_pt_convention as exp
from scripts import pt_convention_report as rep


def model(
    total_inv: float, h_neg: float = 0.0, d_l1: float = 0.1, w_neg: float = 0.0
) -> dict[str, Any]:
    raw = {
        "l_pixel": 0.045,
        "d_l1": d_l1,
        "d_w": 0.023,
        "h_negative": h_neg,
        "w_negative": w_neg,
        **{f"d_l1_{k}": d_l1 for k in ("exact", "mirror", "rot180", "swap")},
    }
    return {
        "step": 56_320,
        "raw": raw,
        "abs_h": raw,
        "totals": {"raw": total_inv + 0.1, "abs_h": total_inv, "inv": total_inv},
    }


def group(*totals: float, h_neg: tuple[float, ...] = (0.0, 0.0, 0.0)) -> dict[str, Any]:
    return {f"seed{i}": model(t, h) for i, (t, h) in enumerate(zip(totals, h_neg, strict=True))}


def test_outliers_and_cv() -> None:
    store = {"g": group(3.7, 3.8, 3.9, h_neg=(1.0, 0.2, 0.0))}
    assert rep._outliers(store, "g") == 1
    assert rep._cv([1.0, 1.0, 1.0]) == 0.0
    assert rep._cv([0.1, 0.1, 0.4]) == pytest.approx(0.866, abs=1e-3)


def test_best_option_prefers_h1_then_the_lowest_total() -> None:
    store = {
        "ours": group(3.77, 3.72, 3.82),
        "optA_positive": group(3.75, 3.73, 3.80),  # no outlier, not worse: H1 met
        "optB_invariant_l1": group(3.70, 3.71, 3.72, h_neg=(1.0, 0.0, 0.0)),  # an outlier seed
        "optC_sign_penalty": group(4.40, 4.50, 4.45),  # clearly worse
    }
    rows, verdicts = rep.option_rows(store, "")
    assert len(rows) == 3
    assert verdicts["optA_positive"]["h1"] is True
    assert (
        verdicts["optB_invariant_l1"]["h1"] is False
        and verdicts["optB_invariant_l1"]["outliers"] == 1
    )
    assert verdicts["optC_sign_penalty"]["worse"] is True
    assert rep.best_option(store) == "optA_positive"
    assert rep.best_option({"ours": store["ours"]}) is None


def test_train_commands_carry_the_option_and_recipe_overrides() -> None:
    cmd = exp.command("optA_positive", 2, "official")
    assert "model.convention=positive" in cmd and "seed=2" in cmd
    assert (
        "output_dir=outputs\\pt_conv\\optA_positive\\seed2" in cmd
        or "output_dir=outputs/pt_conv/optA_positive/seed2" in cmd
    )
    long_cmd = exp.command("optC_sign_penalty", 0, "long")
    assert "loss.lambda_sign=10" in long_cmd and "schedule.n_epochs=400" in long_cmd
    assert "resume=auto" in long_cmd


def test_clean_bias_ignores_models_with_negative_signs() -> None:
    store = {"g": {"a": model(3.7, d_l1=0.1), "b": model(3.7, h_neg=1.0, d_l1=0.25)}}
    text = rep.clean_bias(store)
    assert "1 models" in text and "`mirror` 0.0 %" in text


def test_clean_bias_ignores_the_t021_groups() -> None:
    store = {"g": {"a": model(3.7, d_l1=0.1)}}
    before = rep.clean_bias(store)
    store["pw_full"] = {"a": model(3.7, d_l1=0.5), "b": model(3.7, d_l1=0.3)}
    assert rep.clean_bias(store) == before
