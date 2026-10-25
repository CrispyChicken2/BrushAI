"""Tests of the paper-weights ablation runner and of its report rules (T-021)."""

from __future__ import annotations

from typing import Any

from scripts import exp_pt_paper_weights as pw
from scripts.pt_repro_common import PW_TERMS


def test_commands_carry_the_paper_weights_and_the_removed_term() -> None:
    full = pw.command("full", 1)
    assert "seed=1" in full and "loss.preset=custom" in full and "loss.matching_cost=bce" in full
    assert "loss.pos_weight=8" in full and "loss.lambda_decision=1" in full
    assert not any(a.endswith("=0") for a in full)
    for name, override in pw.REMOVE.items():
        cmd = pw.command(name, 0)
        assert (override in cmd) if override else True
        assert f"output_dir=outputs/pt_pw/{name}/seed0" in cmd


def test_direction_rounds_to_two_decimals_like_t016() -> None:
    assert not pw.direction(1.004) and pw.direction(1.006) and not pw.direction(0.99)
    assert pw.hypothesis(7, 10) == "confirmed"
    assert pw.hypothesis(7, 8) == "weak (within noise)"
    assert pw.hypothesis(7, 7) == "refuted" and pw.hypothesis(7, 3) == "refuted"


def group_values(scale: float) -> dict[str, Any]:
    """Three seeds whose metrics are all ``scale`` times a base value (std 1 %)."""
    return {f"seed{i}": scale * (1 + 0.01 * (i - 1)) for i in range(3)}


def synthetic_stores(worse: dict[str, float]) -> tuple[dict, dict, dict]:
    """Stores with a full model and the ablations of both weights; ``worse[group]`` multiplies
    every metric of the ablated groups of that side."""
    real, strokes, conv = {}, {}, {}
    for full, prefix in (("ours", "abl_"), ("pw_full", "pw_abl_")):
        side = "code" if full == "ours" else "paper"
        for group, scale in [(full, 1.0)] + [(f"{prefix}{t}", worse[side]) for t in PW_TERMS]:
            vals = group_values(scale)
            real[group] = {
                k: {
                    "img": {
                        "l_pixel": v,
                        "l_pcpt": v,
                        "lpips_alex": v,
                        "lpips_vgg": v,
                        "strokes": 1,
                    }
                }
                for k, v in vals.items()
            }
            row = lambda v: {  # noqa: E731
                "l_pixel": v,
                "d_l1": v,
                "d_w": v,
                "h_negative": 0.0,
                "w_negative": 0.0,
                "d_l1_mirror": v,
            }
            strokes[group] = {
                k: {"raw": row(v), "abs_h": row(v), "abs_wh": row(v)} for k, v in vals.items()
            }
            conv[group] = {k: {"raw": row(v)} for k, v in vals.items()}
    return real, strokes, conv


def test_report_counts_cells_and_states_the_verdict() -> None:
    real, strokes, conv = synthetic_stores({"code": 0.9, "paper": 1.3})
    text = pw.build_report(real, strokes, conv)
    assert "code weights 0 of 20, paper weights 20 of 20 -> **confirmed**" in text
    real, strokes, conv = synthetic_stores({"code": 1.3, "paper": 0.9})
    assert "**refuted**" in pw.build_report(real, strokes, conv)
    table = pw.ratios(real, strokes, conv, "code", "fold")
    assert set(table) == set(PW_TERMS) and pw.counts(table) == (20, 20)


def test_report_marks_an_incomplete_evaluation() -> None:
    real, strokes, conv = synthetic_stores({"code": 1.3, "paper": 1.3})
    for store in (real, strokes, conv):
        del store["pw_abl_no_dw"]
    assert "Incomplete" in pw.build_report(real, strokes, conv)
