"""Tests of the instrumentation of the late-training instability study (T-023)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from brushstroke.stroke_model import model as model_module
from brushstroke.stroke_model.model import PaintTransformer, PTModelConfig
from brushstroke.stroke_model.train import _Context, save_checkpoint
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


def run_alert(totals: list[float]) -> int | None:
    detector = exp.AlertDetector()
    for step, total in enumerate(totals, start=1):
        if detector.update(step, total):
            return step
    return None


def test_the_alert_fires_at_the_onset_of_a_slow_excursion_long_before_the_spike() -> None:
    gen = torch.Generator().manual_seed(0)
    noise = (0.05 * torch.randn(20_000, generator=gen)).tolist()
    assert run_alert([4.0 + n for n in noise]) is None  # noise only
    # a ramp of 0.15 per 1000 steps from step 12,000: the 1.25x spike rule needs a rise of 25 %
    ramp = [4.0 + n + max(0.0, (i - 12_000) * 0.00015) for i, n in enumerate(noise)]
    alert, spike = run_alert(ramp), run_spike(ramp)
    assert alert is not None and 12_000 < alert < 13_500
    assert spike is None or spike > alert + 3000


def run_spike(totals: list[float]) -> int | None:
    return run_detector(totals)


@needs_brushes
def test_an_alert_freezes_the_checkpoint_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class AlertAtThree(exp.AlertDetector):
        def update(self, step: int, total: float) -> bool:
            return step == 3

    monkeypatch.setattr(exp, "AlertDetector", AlertAtThree)
    monkeypatch.setattr(exp, "CKPT_EVERY", 1)
    monkeypatch.setattr(exp, "CKPT_KEEP", 2)
    ctx = _Context(tiny_config(tmp_path))
    out = tmp_path / "trace"
    summary = exp.run_loop(ctx, out, 6)
    assert summary["alerts"] == [3] and summary["spike"] is None
    assert sorted(p.name for p in out.glob("ckpt_*.pt")) == ["ckpt_1.pt", "ckpt_2.pt"]


@needs_brushes
def test_branch_runs_end_to_end_and_refuses_a_raw_parent_with_a_positive_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(
        exp, "make_cfg", lambda seed, convention, output, extra=None: tiny_config(output)
    )
    parent = tmp_path / "parent"
    ctx = _Context(tiny_config(parent))
    exp.run_loop(ctx, parent, 2, keep_checkpoints=False)
    ckpt = parent / "ckpt_2.pt"
    save_checkpoint(ctx, ckpt)
    args = exp.build_parser().parse_args(
        ["branch", "--name", "b", "--from-trace", "parent", "--ckpt", str(ckpt), "--steps", "2"]
    )
    exp.cmd_branch(args)
    summary = json.loads((tmp_path / "branches" / "b" / "summary.json").read_text(encoding="utf-8"))
    assert summary["start"] == 2 and summary["steps"] == 4 and summary["factors"]["theta"] == "wrap"
    args.theta = "none"  # tiny_config is a raw-convention model
    with pytest.raises(SystemExit):
        exp.cmd_branch(args)


@needs_brushes
def test_the_branch_point_is_before_the_first_alert_of_the_episode_that_led_to_the_spike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """alert, quiet stretch, second alert, spike: the hold must not expire in between."""

    class Alerts(exp.AlertDetector):
        def update(self, step: int, total: float) -> bool:
            return step in (4, 9)

    class SpikeAtTen(exp.SpikeDetector):
        def update(self, step: int, total: float) -> bool:
            if step == 10:
                self.spike = step
                return True
            return False

    monkeypatch.setattr(exp, "AlertDetector", Alerts)
    monkeypatch.setattr(exp, "SpikeDetector", SpikeAtTen)
    monkeypatch.setattr(exp, "CKPT_EVERY", 1)
    monkeypatch.setattr(exp, "CKPT_KEEP", 20)
    monkeypatch.setattr(exp, "AFTER_SPIKE", 0)
    ctx = _Context(tiny_config(tmp_path))
    out = tmp_path / "trace"
    summary = exp.run_loop(ctx, out, 12)
    assert (
        summary["alerts"] == [4, 9]
        and summary["alert_before_spike"] == 4
        and summary["spike"] == 10
    )
    assert exp.last_good_checkpoint(out, 2) == out / "ckpt_2.pt"
    monkeypatch.setattr(exp, "ALERT_HOLD", 2)  # an episode that expires before the spike
    ctx = _Context(tiny_config(tmp_path / "b"))
    summary = exp.run_loop(ctx, tmp_path / "b" / "trace", 12)
    assert summary["alert_before_spike"] == 9  # the second alert starts a new episode


def test_last_good_checkpoint_keeps_the_oldest_retained_step_in_reach(tmp_path: Path) -> None:
    for step in (1000, 1500, 2000):
        (tmp_path / f"ckpt_{step}.pt").write_bytes(b"")
    (tmp_path / "summary.json").write_text(
        json.dumps({"spike": 3000, "alert_before_spike": 2400}), encoding="utf-8"
    )
    assert exp.last_good_checkpoint(tmp_path, 500) == tmp_path / "ckpt_1500.pt"
    with pytest.raises(SystemExit):
        exp.last_good_checkpoint(tmp_path, 1500)  # 900: nothing retained that far back
