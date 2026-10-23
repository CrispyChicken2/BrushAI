"""Tests for the optimization baseline: Sinkhorn, losses, initialization, progressive painter."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from brushstroke.baselines.optim.config import LossConfig, OptimConfig, PaintOptimConfig
from brushstroke.baselines.optim.init import cosine_lr, dead_strokes, sample_strokes
from brushstroke.baselines.optim.losses import CombinedLoss, l1_loss, lpips_loss
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.baselines.optim.sinkhorn import (
    grid_cost,
    image_ot_loss,
    sinkhorn_divergence,
    sinkhorn_ot,
)
from brushstroke.renderer import BrushBank, BrushConfig, render
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.config import load_config
from brushstroke.utils.seed import seed_everything

BRUSH = BrushConfig(kind="oil", bank_size=4, mask_height=32, mask_width=64, seed=0)


# ------------------------------------------------------------------ Sinkhorn


def blob(size: int, cx: float, cy: float, sigma: float = 0.12) -> torch.Tensor:
    c = (torch.arange(size) + 0.5) / size
    ys, xs = torch.meshgrid(c, c, indexing="ij")
    d = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).reshape(-1) + 1e-4
    return (d / d.sum())[None]


def test_sinkhorn_identical_is_zero_and_symmetric() -> None:
    cost = grid_cost(12)
    a, b = blob(12, 0.3, 0.3), blob(12, 0.7, 0.6)
    assert float(sinkhorn_divergence(a, a, cost)) == pytest.approx(0.0, abs=1e-5)
    ab = float(sinkhorn_divergence(a, b, cost))
    ba = float(sinkhorn_divergence(b, a, cost))
    assert ab > 1e-3
    assert ab == pytest.approx(ba, rel=1e-4, abs=1e-6)


def test_sinkhorn_decreases_as_distributions_approach() -> None:
    cost = grid_cost(12)
    ref = blob(12, 0.2, 0.5)
    values = [
        float(sinkhorn_divergence(ref, blob(12, 0.2 + t, 0.5), cost))
        for t in (0.6, 0.45, 0.3, 0.15, 0.0)
    ]
    assert all(x > y for x, y in zip(values, values[1:], strict=False)), values
    assert values[-1] == pytest.approx(0.0, abs=1e-5)


def test_sinkhorn_ot_matches_closed_form_for_dirac_like_distributions() -> None:
    # With (almost) all mass in one cell the OT cost is the squared distance between the cells.
    size = 6
    cost = grid_cost(size)
    a = torch.full((1, size * size), 1e-9)
    b = a.clone()
    a[0, 0] = 1.0
    b[0, size * size - 1] = 1.0
    a, b = a / a.sum(), b / b.sum()
    expected = float(cost[0, -1])
    assert float(sinkhorn_ot(a, b, cost, eps=0.01, iters=60)) == pytest.approx(expected, rel=0.05)


def test_sinkhorn_batched_and_differentiable() -> None:
    cost = grid_cost(8)
    a = torch.cat([blob(8, 0.3, 0.3), blob(8, 0.6, 0.4)])
    b = torch.cat([blob(8, 0.5, 0.5), blob(8, 0.2, 0.8)]).requires_grad_()
    d = sinkhorn_divergence(a, b, cost)
    assert d.shape == (2,) and (d > 0).all()
    d.sum().backward()
    assert b.grad is not None and torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


def test_image_ot_loss() -> None:
    g = torch.Generator().manual_seed(0)
    img = torch.rand(2, 3, 48, 48, generator=g)
    assert float(image_ot_loss(img, img, size=8)) == pytest.approx(0.0, abs=1e-5)
    shifted = torch.roll(img, 12, dims=-1)
    other = torch.rand(2, 3, 48, 48, generator=g)
    assert float(image_ot_loss(img, shifted, size=8)) >= 0.0
    # a bright blob moving away from the target costs more
    base = torch.zeros(1, 3, 48, 48)
    near, far = base.clone(), base.clone()
    base[..., 8:16, 8:16] = 1
    near[..., 12:20, 12:20] = 1
    far[..., 32:40, 32:40] = 1
    assert float(image_ot_loss(base, near, size=12)) < float(image_ot_loss(base, far, size=12))
    assert float(image_ot_loss(img, other, size=8)) >= 0.0


# ------------------------------------------------------------------ losses


def test_each_loss_flag_has_an_effect() -> None:
    g = torch.Generator().manual_seed(1)
    pred = torch.rand(1, 3, 32, 32, generator=g, requires_grad=True)
    target = torch.rand(1, 3, 32, 32, generator=g)
    l1 = CombinedLoss(LossConfig(l1=1.0, lpips=0.0, ot=0.0))
    assert set(l1.terms(pred, target)) == {"l1"}
    assert float(l1(pred, target)) == pytest.approx(float(l1_loss(pred, target).detach()))
    ot = CombinedLoss(LossConfig(l1=0.0, lpips=0.0, ot=2.0, ot_size=8))
    assert set(ot.terms(pred, target)) == {"ot"}
    both = CombinedLoss(LossConfig(l1=1.0, lpips=0.0, ot=2.0, ot_size=8))
    assert float(both(pred, target)) == pytest.approx(
        float(l1(pred, target)) + float(ot(pred, target)), rel=1e-5
    )
    both(pred, target).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0
    assert float(l1(target, target)) == 0.0 and float(ot(target, target)) == pytest.approx(
        0.0, abs=1e-5
    )
    with pytest.raises(ValueError):
        CombinedLoss(LossConfig(l1=0.0, lpips=0.0, ot=0.0))
    with pytest.raises(ValueError):
        CombinedLoss(LossConfig(l1=-1.0))


@pytest.mark.slow
def test_lpips_loss_flag() -> None:
    g = torch.Generator().manual_seed(2)
    a = torch.rand(1, 3, 64, 64, generator=g, requires_grad=True)
    b = torch.rand(1, 3, 64, 64, generator=g)
    assert float(lpips_loss(a, a.detach())) == pytest.approx(0.0, abs=1e-6)
    assert float(lpips_loss(a, b)) > 0.05
    combo = CombinedLoss(LossConfig(l1=0.0, lpips=1.0))
    assert set(combo.terms(a, b)) == {"lpips"}
    combo(a, b).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0


# ------------------------------------------------------------------ initialization helpers


def test_sample_strokes_inside_patch_follow_error_and_target_colour() -> None:
    h = w = 64
    target = torch.zeros(3, h, w)
    target[:, :, :] = torch.tensor([0.2, 0.4, 0.6]).view(3, 1, 1)
    canvas = target.clone()
    canvas[:, 40:56, 8:24] += 0.5  # the only error is in patch (row 1, col 0) of a 2x2 grid
    cfg = OptimConfig(height=h, width=w, init_error_power=4.0)
    gen = torch.Generator().manual_seed(0)
    ids = torch.full((200,), 2)  # patch 2 = row 1, col 0
    p = sample_strokes(target, canvas, 2, ids, cfg, gen)
    assert p.shape == (200, 9) and p.dtype == torch.float32
    x, y = p[:, 0] * w, p[:, 1] * h
    assert ((x >= 0) & (x < 32) & (y >= 32) & (y < 64)).all()
    inside = ((x >= 4) & (x < 28) & (y >= 36) & (y < 60)).float().mean()
    assert inside > 0.9  # sampling concentrates on the (blurred) error region
    assert torch.allclose(p[:, 5:8], torch.tensor([0.2, 0.4, 0.6]), atol=1e-3)
    assert (
        (p[:, 2] <= 1).all()
        and (p[:, 3] > 0).all()
        and (p[:, 4] >= 0).all()
        and (p[:, 4] < 1).all()
    )
    assert torch.allclose(p[:, 8], torch.tensor(cfg.init_opacity))
    # zero error everywhere: uniform fallback, still inside the patch
    q = sample_strokes(target, target, 2, ids, cfg, torch.Generator().manual_seed(1))
    assert ((q[:, 0] * w < 32) & (q[:, 1] * h >= 32)).all()
    assert q[:, 0].std() > 0.05
    # reproducible
    again = sample_strokes(target, canvas, 2, ids, cfg, torch.Generator().manual_seed(0))
    assert torch.equal(p, again)


def test_dead_strokes_and_cosine_lr() -> None:
    cfg = OptimConfig(dead_opacity=0.05, dead_area=1e-3)
    p = torch.tensor(
        [
            [0.5, 0.5, 0.3, 0.2, 0.1, 1, 1, 1, 0.9],  # alive
            [0.5, 0.5, 0.3, 0.2, 0.1, 1, 1, 1, 0.01],  # transparent
            [0.5, 0.5, 0.01, 0.02, 0.1, 1, 1, 1, 0.9],  # tiny
        ]
    )
    assert dead_strokes(p, cfg).tolist() == [False, True, True]
    assert cosine_lr(1.0, 0.1, 0, 10) == pytest.approx(1.0)
    assert cosine_lr(1.0, 0.1, 9, 10) == pytest.approx(0.1)
    assert cosine_lr(1.0, 0.1, 4, 10) < 1.0


# ------------------------------------------------------------------ progressive painter


def synthetic_target(size: int = 48, n: int = 6, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(1, n, 9, generator=g)
    p[..., 2:4] = 0.2 + 0.4 * p[..., 2:4]
    p[..., 8] = 1.0
    bank = BrushBank.build(BRUSH.kind, BRUSH.bank_size, (BRUSH.mask_height, BRUSH.mask_width), 0)
    return render(p, bank, size, size)[0]


def small_cfg(**kw: object) -> OptimConfig:
    base = {
        "height": 48,
        "width": 48,
        "levels": [1, 2, 3],
        "strokes_per_patch": [3, 2, 2],
        "iters": 60,
        "iters_per_level": [],
        "finetune_all": False,
        "finetune_lr_factor": 0.3,
    }
    return dataclasses.replace(OptimConfig(), **{**base, **kw})  # type: ignore[arg-type]


def l1_only() -> LossConfig:
    return LossConfig(l1=1.0, lpips=0.0, ot=0.0)


def test_synthetic_recovery_reduces_error_far_below_initial() -> None:
    target = synthetic_target()
    initial = float((torch.ones_like(target) - target).abs().mean())
    seq = paint_optim(target, small_cfg(), l1_only(), BRUSH, seed=0)
    final = float((seq.render() - target).abs().mean())
    # measured on this seed: initial ~0.147 -> final ~0.043; thresholds keep a wide margin
    assert initial > 0.1
    assert final < 0.6 * initial and final < 0.08, (initial, final)


def test_output_sequence_structure_and_meta() -> None:
    target = synthetic_target()
    cfg = small_cfg()
    seq = paint_optim(target, cfg, l1_only(), BRUSH, seed=3)
    counts = [g * g * s for g, s in zip(cfg.levels, cfg.strokes_per_patch, strict=True)]
    assert len(seq) == sum(counts) and seq.canvas_size == (48, 48)
    assert seq.scale_level is not None and seq.patch_id is not None
    expected_levels = torch.cat(
        [torch.full((c,), i, dtype=torch.int32) for i, c in enumerate(counts)]
    )
    assert torch.equal(seq.scale_level, expected_levels)
    assert int(seq.patch_id[: counts[0]].max()) == 0
    assert int(seq.patch_id[counts[0] : counts[0] + counts[1]].max()) == 3  # 2x2 grid
    assert set(seq.source or []) == {"optim_baseline"}
    for key in (
        "config_hash",
        "seed",
        "loss_curve",
        "level_seconds",
        "runtime_seconds",
        "peak_vram_mb",
    ):
        assert key in seq.meta
    assert seq.meta["seed"] == 3 and len(seq.meta["loss_curve"]) == 3
    assert (seq.params[:, :2] >= 0).all() and (seq.params[:, :2] <= 1).all()
    assert (seq.params[:, 2:4] > 0).all() and (seq.params[:, 8] <= 1).all()
    assert torch.isfinite(seq.params).all()
    # the sequence re-renders what was optimized: final loss in meta equals the render error
    final = float((seq.render() - target).abs().mean())
    assert final == pytest.approx(seq.meta["level_final_loss"][-1]["l1_metric"], abs=1e-4)


def test_loss_decreases_in_trend_within_each_level() -> None:
    target = synthetic_target(seed=2)
    seq = paint_optim(target, small_cfg(iters=80), l1_only(), BRUSH, seed=0)
    for curve in seq.meta["loss_curve"]:
        head = sum(curve[:8]) / 8
        tail = sum(curve[-8:]) / 8
        assert tail < head, (head, tail)
    # and the loss at the end of each level improves on the previous level
    ends = [c[-1] for c in seq.meta["loss_curve"]]
    assert ends[-1] < ends[0]


def test_same_seed_and_deterministic_flag_gives_identical_sequence() -> None:
    target = synthetic_target()
    try:
        seed_everything(5, deterministic=True)
        a = paint_optim(target, small_cfg(iters=20), l1_only(), BRUSH, seed=5)
        seed_everything(5, deterministic=True)
        b = paint_optim(target, small_cfg(iters=20), l1_only(), BRUSH, seed=5)
    finally:
        seed_everything(0, deterministic=False)
    assert torch.equal(a.params, b.params)
    assert a.brush_index is not None and b.brush_index is not None
    assert torch.equal(a.brush_index, b.brush_index)
    c = paint_optim(target, small_cfg(iters=20), l1_only(), BRUSH, seed=6)
    assert not torch.equal(a.params, c.params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimizer": "rmsprop", "lr": 0.05},
        {"finetune_all": True},
        {"resample_every": 10},
        {"checkpoint_bytes": 0.0},  # forces gradient checkpointing
    ],
)
def test_optimizer_and_feature_flags_run_and_improve(kwargs: dict[str, object]) -> None:
    target = synthetic_target()
    initial = float((torch.ones_like(target) - target).abs().mean())
    seq = paint_optim(target, small_cfg(iters=40, **kwargs), l1_only(), BRUSH, seed=0)
    assert float((seq.render() - target).abs().mean()) < 0.8 * initial


def test_ot_loss_flag_and_flat_brush_run() -> None:
    target = synthetic_target()
    loss = LossConfig(l1=1.0, lpips=0.0, ot=0.5, ot_size=8, ot_iters=10)
    seq = paint_optim(
        target, small_cfg(iters=20), loss, dataclasses.replace(BRUSH, kind="flat"), seed=0
    )
    assert seq.brush.kind == "flat" and "ot" in seq.meta["level_final_loss"][-1]
    assert seq.meta["level_final_loss"][-1]["ot"] >= 0.0


@pytest.mark.slow
def test_lpips_loss_flag_in_painter() -> None:
    target = synthetic_target(size=64)
    cfg = small_cfg(height=64, width=64, iters=10, levels=[1, 2], strokes_per_patch=[2, 2])
    seq = paint_optim(target, cfg, LossConfig(l1=0.5, lpips=0.5), BRUSH, seed=0)
    assert "lpips" in seq.meta["level_final_loss"][-1]


def test_callback_and_validation() -> None:
    target = synthetic_target()
    seen: list[tuple[int, int]] = []

    def cb(level: int, canvas: torch.Tensor, seq: StrokeSequence) -> None:
        assert canvas.shape == (3, 48, 48)
        seen.append((level, len(seq)))

    paint_optim(target, small_cfg(iters=5), l1_only(), BRUSH, callback=cb)
    assert [s[0] for s in seen] == [0, 1, 2] and seen[-1][1] == 3 + 8 + 18
    with pytest.raises(ValueError):
        paint_optim(target[:, :40], small_cfg(), l1_only(), BRUSH)
    with pytest.raises(ValueError):
        paint_optim(target, small_cfg(strokes_per_patch=[1]), l1_only(), BRUSH)
    with pytest.raises(ValueError):
        paint_optim(target, small_cfg(iters=0), l1_only(), BRUSH)
    with pytest.raises(ValueError):
        paint_optim(target, small_cfg(optimizer="sgd"), l1_only(), BRUSH)


def test_baseline_yaml_matches_dataclass_defaults() -> None:
    path = Path(__file__).parents[1] / "configs" / "baseline_optim.yaml"
    assert load_config(PaintOptimConfig, path) == PaintOptimConfig()


@pytest.mark.gpu
def test_paint_optim_on_gpu_matches_cpu_quality() -> None:
    target = synthetic_target()
    seq = paint_optim(target, small_cfg(iters=40), l1_only(), BRUSH, seed=0, device="cuda")
    assert seq.meta["peak_vram_mb"] > 0
    initial = float((torch.ones_like(target) - target).abs().mean())
    assert float((seq.render() - target).abs().mean()) < 0.7 * initial
