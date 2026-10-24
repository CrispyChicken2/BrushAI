"""Tests of the T-016 reproduction tools: L_pcpt, stroke metrics, Optim, ablation setup, reports."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brushstroke.eval.pcpt import LAYER, pcpt_loss, vgg_features
from brushstroke.eval.pt_strokes import (
    StrokeScores,
    network_scores,
    optim_strokes,
    unweighted_loss,
)
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model import PaintTransformer, RefineConfig, resolve_loss_config
from brushstroke.stroke_model.synthetic import SyntheticConfig, SyntheticStrokeGenerator
from brushstroke.stroke_model.train import TrainConfig
from brushstroke.utils.config import load_config
from scripts import exp_pt_ablations, exp_pt_table1, pt_repro_common
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)


# ------------------------------------------------------------------ L_pcpt


def test_pcpt_loss_basic_properties() -> None:
    torch.manual_seed(0)
    a, b = torch.rand(3, 64, 64), torch.rand(3, 64, 64)
    assert LAYER == "relu3_3"
    assert vgg_features(a[None]).shape == (1, 256, 16, 16)
    assert pcpt_loss(a, a) == 0.0
    assert pcpt_loss(a, b) == pytest.approx(pcpt_loss(b, a)) and pcpt_loss(a, b) > 0
    # a larger perturbation costs more; batches and single images agree
    near = (a + 0.02 * torch.randn_like(a)).clamp(0, 1)
    assert pcpt_loss(a, near) < pcpt_loss(a, b)
    assert pcpt_loss(torch.stack([a, a]), torch.stack([b, b])) == pytest.approx(pcpt_loss(a, b))
    with pytest.raises(ValueError):
        pcpt_loss(a, b[:, :32])


# ------------------------------------------------------------------ stroke metrics


@pytest.fixture(scope="module")
def synthetic() -> tuple[torch.Tensor, list]:
    brushes = load_pt_brushes("train", po.BRUSHES, "cpu")
    gen = SyntheticStrokeGenerator(brushes, SyntheticConfig())
    rng = torch.Generator().manual_seed(3)
    return brushes, [gen.sample(16, rng) for _ in range(2)]


@needs_brushes
def test_unweighted_terms_of_a_perfect_prediction_are_zero(
    synthetic: tuple[torch.Tensor, list],
) -> None:
    """Ground truth as prediction: matching is the identity, D_L1 = D_W = 0, L_pixel ~ 0."""
    brushes, data = synthetic
    loss = unweighted_loss(brushes)
    scores = StrokeScores()
    for batch in data:
        logits = torch.where(batch.gt_decision.bool(), 10.0, -10.0)
        scores.add(loss, batch.gt_param, batch.gt_color, logits, batch)
    result = scores.result()
    assert result["d_l1"] < 1e-6 and result["d_w"] < 1e-3
    assert result["l_pixel"] < 5e-3 and result["h_negative"] == 0.0
    assert 5 < result["painted_per_patch"] < 8.01


@needs_brushes
def test_untrained_network_is_far_worse_and_abs_h_only_changes_negative_heights(
    synthetic: tuple[torch.Tensor, list],
) -> None:
    brushes, data = synthetic
    loss = unweighted_loss(brushes)
    net = PaintTransformer().eval()
    torch.manual_seed(0)
    raw = network_scores(net, loss, data, abs_h=False)
    folded = network_scores(net, loss, data, abs_h=True)
    assert raw["d_l1"] > 0.1 and raw["d_w"] > 0.05
    assert raw["h_negative"] > 0 and folded["h_negative"] == 0  # the fold removes every negative h
    assert folded["d_l1"] <= raw["d_l1"] + 1e-9  # |h| is never farther from a positive ground truth
    both = network_scores(net, loss, data, abs_h=True, abs_w=True)
    assert both["d_l1"] <= folded["d_l1"] + 1e-9
    assert 0 <= raw["w_negative"] <= 1


@needs_brushes
def test_optim_comparison_improves_a_random_initialisation(
    synthetic: tuple[torch.Tensor, list],
) -> None:
    brushes, data = synthetic
    batch = data[0]
    loss = unweighted_loss(brushes)
    painted = batch.gt_decision.bool()  # as many strokes as the ground truth
    results = {}
    for iters in (0, 40):
        cfg = RefineConfig(enabled=True, iters=iters, lr=0.05, keep_if_better=False)
        shape, colour, logits = optim_strokes(batch, painted, brushes, cfg, seed=1)
        assert torch.equal(logits > 0, painted)
        scores = StrokeScores()
        scores.add(loss, shape, colour, logits, batch)
        results[iters] = scores.result()
    assert results[40]["l_pixel"] < results[0]["l_pixel"]
    # deterministic given the seed
    again = optim_strokes(batch, painted, brushes, RefineConfig(enabled=True, iters=3), seed=1)
    other = optim_strokes(batch, painted, brushes, RefineConfig(enabled=True, iters=3), seed=2)
    first = optim_strokes(batch, painted, brushes, RefineConfig(enabled=True, iters=3), seed=1)
    assert torch.equal(again[0], first[0]) and not torch.equal(again[0], other[0])


# ------------------------------------------------------------------ ablation setup


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("no_pixel", "lambda_pixel"),
        ("no_dl1", "lambda_gt"),
        ("no_dw", "lambda_w"),
        ("no_dbce", "lambda_decision"),
    ],
)
def test_each_ablation_removes_exactly_one_loss_weight(name: str, field: str) -> None:
    cmd = exp_pt_ablations.command(name, 1, Path("outputs/x"))
    overrides = [c for c in cmd if "=" in c and not c.startswith("-")]
    cfg = load_config(TrainConfig, Path("configs/stroke_model/pt.yaml"), overrides)
    loss, official = resolve_loss_config(cfg.loss), resolve_loss_config(TrainConfig().loss)
    assert getattr(loss, field) == 0.0
    for other in ("lambda_pixel", "lambda_gt", "lambda_w", "lambda_decision", "pos_weight"):
        if other != field:
            assert getattr(loss, other) == getattr(official, other), other
    assert loss.matching_cost == official.matching_cost
    assert (
        cfg.seed == 1
        and Path(cfg.output_dir).parts[-2:] == (name, "seed1")
        and cfg.resume == "auto"
    )
    assert cfg.schedule.total_steps == 56_320  # the official recipe


# ------------------------------------------------------------------ report helpers


def test_report_helpers(tmp_path: Path) -> None:
    assert pt_repro_common.mean_std([1.0, 3.0]) == (2.0, pytest.approx(2**0.5))
    assert pt_repro_common.mean_std([5.0]) == (5.0, 0.0)
    assert pt_repro_common.fmt_ms([1.0]) == "1.000" and "±" in pt_repro_common.fmt_ms([1.0, 2.0])
    doc = tmp_path / "doc.md"
    doc.write_text("intro\n", encoding="utf-8")
    pt_repro_common.write_block("X", "first", doc)
    pt_repro_common.write_block("Y", "other", doc)
    pt_repro_common.write_block("X", "second", doc)
    text = doc.read_text(encoding="utf-8")
    assert "second" in text and "first" not in text and "other" in text and text.startswith("intro")
    assert text.count("<!-- BEGIN X -->") == 1


def _fake_stroke(l_pixel: float, d_l1: float, d_w: float) -> dict:
    scores = {
        "l_pixel": l_pixel,
        "d_l1": d_l1,
        "d_w": d_w,
        "h_negative": 0.0,
        "w_negative": 0.0,
        "painted_per_patch": 7.5,
    }
    return {"raw": scores, "abs_h": scores, "abs_wh": scores}


def _fake_real(l_pixel: float, l_pcpt: float) -> dict:
    row = {
        "l_pixel": l_pixel,
        "l_pcpt": l_pcpt,
        "lpips_alex": 0.5,
        "lpips_vgg": 0.6,
        "strokes": 4000.0,
    }
    return {"mona_lisa": row, "great_wave": row}


def test_ablation_table_reports_effects_against_the_full_model() -> None:
    stroke = {
        "ours": {"seed0": _fake_stroke(0.04, 0.1, 0.02), "seed1": _fake_stroke(0.05, 0.1, 0.02)},
        "abl_no_pixel": {"seed0": _fake_stroke(0.08, 0.2, 0.03)},
    }
    real = {
        "ours": {"seed0": _fake_real(0.05, 0.8), "seed1": _fake_real(0.06, 0.9)},
        "abl_no_pixel": {"seed0": _fake_real(0.1, 1.0)},
    }
    table = exp_pt_table1.build_ablation_table(real, stroke)
    assert "paper w/o L_pixel" in table and "**ours** w/o L_pixel (n = 1)" in table
    assert "paper 1.45, ours 1.82" in table  # real L_pixel: 0.081/0.056 and 0.1/0.055
    assert "w/o D_L1" not in table.split("Effect of removing")[1]  # no run for the other terms
    assert "n/a" in exp_pt_table1.build_stroke_table({"ours": stroke["ours"]}).splitlines()[2]


def test_real_stage_skips_runs_already_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    """``real`` evaluates only the runs missing from ``table1_real.json``."""
    stored = {"ours": {"seed0": {"mona_lisa": {"l_pixel": 0.1}}}}

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a stored run was evaluated again")

    monkeypatch.setattr(exp_pt_table1, "GROUPS", {"ours": [("seed0", Path("x.pt"))]})
    monkeypatch.setattr(exp_pt_table1, "_stored", lambda name: stored)
    monkeypatch.setattr(exp_pt_table1, "_finished", lambda path, group: True)
    monkeypatch.setattr(exp_pt_table1, "get_device", lambda name: torch.device("cpu"))
    monkeypatch.setattr(exp_pt_table1, "load_pt_brushes", lambda *a, **k: torch.zeros(1))
    monkeypatch.setattr(exp_pt_table1, "_images", lambda device: {"mona_lisa": torch.zeros(1)})
    monkeypatch.setattr(exp_pt_table1, "load_pt_model", forbidden)
    monkeypatch.setattr(exp_pt_table1, "paint", forbidden)
    monkeypatch.setattr(exp_pt_table1, "dump", forbidden)
    exp_pt_table1.real()


# ------------------------------------------------------------------ T-017 verdicts


def test_ablation_verdict_rules() -> None:
    from scripts import exp_pt_ablation_verdict as v

    assert v.welch_t([2.0, 2.1, 1.9], [1.0, 1.1, 0.9]) > v.T_CRITICAL
    assert v.welch_t([1.0, 1.1, 0.9], [1.0, 1.1, 0.9]) == 0.0
    assert (v.verdict(3), v.verdict(2), v.verdict(0)) == ("matters", "partly", "not reproduced")
    assert v.kendall_tau([1, 2, 3], [1, 2, 3]) == 1.0
    assert v.kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
    assert v.ordering({"a": 1.5, "b": 1.48, "c": 1.1}) == "a ~ b > c"


def test_ablation_verdict_report_marks_worse_and_better_cells() -> None:
    from scripts import exp_pt_ablation_verdict as v

    def stroke(l_pixel: float) -> dict:
        return {"abs_h": _fake_stroke(l_pixel, 0.1, 0.02)["abs_h"]} | {
            "abs_wh": _fake_stroke(l_pixel, 0.1, 0.02)["abs_h"]
        }

    def real(l_pixel: float) -> dict:
        return {"mona_lisa": _fake_real(l_pixel, 0.8)["mona_lisa"]}

    seeds = ("seed0", "seed1", "seed2")
    full = (0.050, 0.051, 0.049)
    worse = (0.080, 0.081, 0.079)
    better = (0.030, 0.031, 0.029)
    real_data = {
        "ours": {s: real(x) for s, x in zip(seeds, full, strict=True)},
        "abl_no_pixel": {s: real(x) for s, x in zip(seeds, worse, strict=True)},
        "abl_no_dbce": {s: real(x) for s, x in zip(seeds, better, strict=True)},
    }
    stroke_data = {
        "ours": {s: stroke(x) for s, x in zip(seeds, full, strict=True)},
        "abl_no_pixel": {s: stroke(x) for s, x in zip(seeds, worse, strict=True)},
        "abl_no_dbce": {s: stroke(x) for s, x in zip(seeds, better, strict=True)},
    }
    result = v.analyse(real_data, stroke_data, fold=False)
    assert result["w/o L_pixel"]["worse"][0] and result["w/o D_bce"]["better"][0]
    assert result["w/o L_pixel"]["verdict"] == "partly"  # only the two L_pixel columns move
    assert "w/o D_bce" in v.build_report(real_data, stroke_data)
