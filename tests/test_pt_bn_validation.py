"""Tests of the BatchNorm modes of the validation of the PT training (T-020)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from brushstroke.stroke_model.train import (
    BN_MODES,
    _Context,
    evaluate,
    make_calibration_set,
    make_validation_set,
    train,
    validation_network,
)
from tests.test_pt_train import (
    assert_same_state,
    final_state,
    needs_brushes,
    read_csv,
    tiny_config,
)

pytestmark = needs_brushes


def with_bn(cfg, mode: str):  # type: ignore[no-untyped-def]
    return replace(cfg, val=replace(cfg.val, bn=mode, bn_batches=2))


def trained_context(tmp_path: Path, steps: int = 4) -> _Context:
    """A context whose network took a few steps, so that running stats and weights are non-trivial."""
    ctx = _Context(with_bn(tiny_config(tmp_path), "running"))
    for _ in range(steps):
        ctx.train_step(ctx.generator.sample(ctx.cfg.batch_size, ctx.data_rng))
    return ctx


def test_the_mode_is_validated(tmp_path: Path) -> None:
    assert BN_MODES == ("running", "recalibrate", "batch")
    with pytest.raises(ValueError, match="val.bn"):
        _Context(with_bn(tiny_config(tmp_path), "nope"))


def test_the_default_mode_is_the_former_behaviour(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    assert cfg.val.bn == "running"
    ctx = _Context(cfg)
    assert validation_network(ctx, "running") is ctx.net


@pytest.mark.parametrize("mode", ["recalibrate", "batch"])
def test_validation_leaves_the_training_state_untouched(tmp_path: Path, mode: str) -> None:
    ctx = trained_context(tmp_path)
    val = make_validation_set(ctx)
    before = {k: v.clone() for k, v in ctx.net.state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    evaluate(ctx, val, mode, make_calibration_set(ctx))
    assert ctx.net.training  # train mode restored
    assert all(torch.equal(v, ctx.net.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng_before, torch.get_rng_state())


def test_the_copies_use_batch_statistics_but_no_dropout(tmp_path: Path) -> None:
    ctx = trained_context(tmp_path)
    calibration = make_calibration_set(ctx)
    for mode in ("recalibrate", "batch"):
        net = validation_network(ctx, mode, calibration)
        assert net is not ctx.net
        assert not any(m.training for m in net.modules() if isinstance(m, nn.Dropout))
        assert not net.training or mode == "batch"  # the eval-mode network of `recalibrate`
    # same input, same output twice (no dropout, no stochastic statistics)
    val = make_validation_set(ctx)
    a = evaluate(ctx, val, "batch")
    b = evaluate(ctx, val, "batch")
    assert a == b


def test_recalibration_estimates_the_statistics_of_the_calibration_batches(tmp_path: Path) -> None:
    ctx = trained_context(tmp_path)
    one = make_calibration_set(ctx)[:1]
    first_bn = next(m for m in ctx.net.modules() if isinstance(m, nn.BatchNorm2d))
    captured: list[torch.Tensor] = []
    handle = first_bn.register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))
    try:
        ctx.net.eval()
        ctx.forward(one[0])  # only to capture the input of that layer (on the original network)
    finally:
        handle.remove()
    net = validation_network(ctx, "recalibrate", one)
    copy_bn = next(m for m in net.modules() if isinstance(m, nn.BatchNorm2d))
    x = captured[0]
    # the first layer sees the images, so its input does not depend on the BatchNorm statistics
    assert torch.allclose(copy_bn.running_mean, x.mean(dim=(0, 2, 3)), atol=1e-5)
    n = x.numel() / x.shape[1]
    unbiased = x.var(dim=(0, 2, 3), unbiased=True)
    assert torch.allclose(copy_bn.running_var, unbiased, rtol=1e-4)
    assert n > 1


def test_modes_change_the_score_of_a_net_whose_running_statistics_are_stale(
    tmp_path: Path,
) -> None:
    ctx = trained_context(tmp_path, steps=2)
    for m in ctx.net.modules():  # stale statistics, as in the spikes of T-014
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.add_(0.5)
            m.running_var.mul_(3.0)
    val = make_validation_set(ctx)
    calibration = make_calibration_set(ctx)
    scores = {mode: evaluate(ctx, val, mode, calibration)["total"] for mode in BN_MODES}
    assert len(set(scores.values())) == 3
    # recalibrated and batch statistics agree with each other much better than with the stale ones
    assert abs(scores["recalibrate"] - scores["batch"]) < abs(scores["running"] - scores["batch"])


@pytest.mark.parametrize("mode", ["recalibrate", "batch"])
def test_training_weights_are_bitwise_unaffected_by_the_validation_mode(
    tmp_path: Path, mode: str
) -> None:
    train(with_bn(tiny_config(tmp_path / "running"), "running"))
    train(with_bn(tiny_config(tmp_path / mode), mode))
    assert_same_state(final_state(tmp_path / "running"), final_state(tmp_path / mode))
    a, b = (read_csv(tmp_path / d / "val.csv") for d in ("running", mode))
    assert [r["step"] for r in a] == [r["step"] for r in b]
    assert a != b  # the validation values themselves do differ


def test_resume_stays_bitwise_identical_with_recalibration(tmp_path: Path) -> None:
    """The T-013 resume test with ``val.bn = recalibrate``."""
    straight = with_bn(tiny_config(tmp_path / "straight"), "recalibrate")
    train(straight)
    split = with_bn(tiny_config(tmp_path / "split"), "recalibrate")
    train(replace(split, schedule=replace(split.schedule, max_steps=3)))
    train(replace(split, resume="auto"))
    assert_same_state(final_state(tmp_path / "straight"), final_state(tmp_path / "split"))
    val_a, val_b = (read_csv(tmp_path / d / "val.csv") for d in ("straight", "split"))
    assert val_a == val_b
