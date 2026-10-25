"""Tests of the instrumentation of the late-training instability study (T-023)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from brushstroke.stroke_model import model as model_module
from brushstroke.stroke_model.model import PaintTransformer, PTModelConfig
from brushstroke.stroke_model.train import _Context
from scripts import exp_pt_instability as exp
from tests.test_pt_train import assert_same_state, needs_brushes, tiny_config


def test_every_parameter_belongs_to_a_known_group() -> None:
    net = PaintTransformer(PTModelConfig(hidden_dim=32, heads=4, enc_layers=1, dec_layers=1))
    groups = {exp.group_of(name) for name, _ in net.named_parameters()}
    assert groups == set(exp.GROUPS)
    assert exp.group_of("transformer.encoder.layers.0.linear1.weight") == "tr_enc"
    assert exp.group_of("transformer.decoder.norm.weight") == "tr_dec"
    assert exp.group_of("linear_decider.bias") == "head"
    assert exp.group_of("query_pos") == "embed"


def run_detector(totals: list[float]) -> int | None:
    detector = exp.SpikeDetector()
    for step, total in enumerate(totals, start=1):
        if detector.update(step, total):
            return step
    return None


def test_the_detector_finds_a_jump_and_ignores_a_slow_decay_and_noise() -> None:
    gen = torch.Generator().manual_seed(0)
    noise = (0.05 * torch.randn(20_000, generator=gen)).tolist()
    flat = [4.0 + n for n in noise]
    assert run_detector(flat) is None
    decay = [5.0 - 1.0 * i / 20_000 + n for i, n in enumerate(noise)]
    assert run_detector(decay) is None
    jump = [4.0 + n if i < 12_000 else 6.0 + n for i, n in enumerate(noise)]
    hit = run_detector(jump)
    assert hit is not None and 12_000 < hit <= 12_000 + exp.SPIKE_WINDOW


def test_the_detector_needs_the_warmup_and_a_reference() -> None:
    assert (
        run_detector([4.0] * 100 + [9.0] * 3000) is None
    )  # before the warm-up and the reference level


@needs_brushes
def test_the_traced_step_trains_exactly_like_the_training_step(tmp_path: Path) -> None:
    a = _Context(tiny_config(tmp_path / "a"))
    for _ in range(3):  # sequential: the dropout masks come from the global torch RNG
        a.train_step(a.generator.sample(a.cfg.batch_size, a.data_rng))
    b = _Context(tiny_config(tmp_path / "b"))
    tap = exp.Tap(b.net)
    for _ in range(3):
        row = exp.traced_step(b, b.generator.sample(b.cfg.batch_size, b.data_rng), tap)
    assert row["gnorm"] > 0 and row["enc_std"] > 0
    assert row["r_head"] == row["r_head"]  # Adam has a second moment by the third step
    sa = {
        "step": a.step,
        "model": a.net.state_dict(),
        "optimizer": a.optimizer.state_dict(),
        "scheduler": a.scheduler.state_dict(),
    }
    sb = {
        "step": b.step,
        "model": b.net.state_dict(),
        "optimizer": b.optimizer.state_dict(),
        "scheduler": b.scheduler.state_dict(),
    }
    assert_same_state(sa, sb)


@needs_brushes
def test_run_loop_writes_the_trace_and_checkpoints(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    ctx = _Context(cfg)
    out = tmp_path / "trace"
    summary = exp.run_loop(ctx, out, 3)
    assert summary["steps"] == 3 and summary["spike"] is None
    rows = list(csv.DictReader((out / "trace.csv").open(encoding="utf-8")))
    assert [int(r["step"]) for r in rows] == [1, 2, 3] and set(exp.FIELDS) == set(rows[0])


@needs_brushes
def test_scale_lr_scales_the_current_and_the_future_learning_rate(tmp_path: Path) -> None:
    ctx = _Context(tiny_config(tmp_path))
    before = ctx.optimizer.param_groups[0]["lr"]
    exp.scale_lr(ctx, 0.3)
    assert ctx.optimizer.param_groups[0]["lr"] == pytest.approx(0.3 * before)
    ctx.train_step(ctx.generator.sample(ctx.cfg.batch_size, ctx.data_rng))
    assert ctx.optimizer.param_groups[0]["lr"] == pytest.approx(0.3 * before)


@needs_brushes
def test_bn_stays_in_eval_mode_and_keeps_its_statistics(tmp_path: Path) -> None:
    ctx = _Context(tiny_config(tmp_path))
    exp.keep_batchnorm_in_eval(ctx.net)
    bn = next(m for m in ctx.net.modules() if isinstance(m, torch.nn.BatchNorm2d))
    mean = bn.running_mean.clone()  # type: ignore[union-attr]
    ctx.train_step(ctx.generator.sample(ctx.cfg.batch_size, ctx.data_rng))
    assert not bn.training and torch.equal(bn.running_mean, mean)  # type: ignore[arg-type]
    assert ctx.net.transformer.training  # the rest of the network is in training mode


def test_the_branch_parameterizations_and_their_restoration() -> None:
    raw = torch.tensor([[[0.5, 0.5, -0.7, 0.2, 2.3, 0.0]]])
    original = model_module.positive_convention
    with exp.patched_convention("exp", "none"):
        out = model_module.positive_convention(raw)
        assert out[0, 0, 2].item() == pytest.approx(torch.exp(torch.tensor(-0.7)).item())
        assert out[0, 0, 4].item() == pytest.approx(2.3)  # theta not wrapped
    assert model_module.positive_convention is original
    default = exp._positive("softplus", "wrap")(raw)
    assert torch.equal(default, original(raw))  # the default branch is option A itself
    with pytest.raises(ValueError):
        exp._positive("nope", "wrap")(raw)
