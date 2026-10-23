"""Tests of the synthetic stroke data generator (T-011), against the official ``set_input`` when
available."""

from __future__ import annotations

import math

import pytest
import torch

from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model import SyntheticConfig, SyntheticStrokeGenerator
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
needs_official = pytest.mark.skipif(
    not (po.have_brushes() and po.have_official()),
    reason="official code/assets missing: run fetch_pt_brushes + pt_reference_study fetch",
)
# Official statistics, measured with the official set_input over 200 batches of 64
# (scripts/pt_reference_study.py datastats)
OFFICIAL_MEAN, OFFICIAL_STD, OFFICIAL_PATCHES = 6.923281192779541, 0.9107642769813538, 12800
OFFICIAL_HISTOGRAM = [0, 0, 1, 4, 107, 761, 2857, 5331, 3739]


def make_generator(device: str = "cpu") -> SyntheticStrokeGenerator:
    return SyntheticStrokeGenerator(load_pt_brushes("train", po.BRUSHES, device))


def draw(gen: SyntheticStrokeGenerator, batches: int, size: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=gen.device).manual_seed(seed)
    return torch.cat([gen.sample(size, g).gt_decision.sum(1).cpu() for _ in range(batches)])


@needs_brushes
def test_batch_structure_and_ranges() -> None:
    gen = make_generator()
    b = gen.sample(16, torch.Generator().manual_seed(0))
    assert b.canvas.shape == (16, 3, 32, 32) and b.target.shape == (16, 3, 32, 32)
    assert (
        b.gt_param.shape == (16, 8, 5)
        and b.gt_decision.shape == (16, 8)
        and b.gt_color.shape == (16, 8, 3)
    )
    assert set(b.gt_decision.unique().tolist()) <= {0.0, 1.0} and b.gt_decision[:, 0].eq(1).all()
    xywh = b.gt_param[..., :4]
    assert (
        float(xywh.min()) >= 0.2
        and float(xywh.max()) < 0.7
        and float(b.gt_param[..., 4].max()) < 1.0
    )
    assert float(b.target.min()) >= 0.0 and float(b.target.max()) <= 1.0 + 1e-6
    assert float(b.canvas.mean()) > 0.05  # the background has strokes
    assert not torch.equal(b.canvas, b.target)
    for bad in (0, 3, 10):
        with pytest.raises(ValueError):
            gen.sample(bad)


@needs_brushes
def test_generator_is_deterministic_for_a_seed() -> None:
    gen = make_generator()
    a = gen.sample(8, torch.Generator().manual_seed(5))
    b = gen.sample(8, torch.Generator().manual_seed(5))
    c = gen.sample(8, torch.Generator().manual_seed(6))
    for name in ("canvas", "target", "gt_param", "gt_decision", "gt_color"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(a.target, c.target) and not torch.equal(a.gt_param, c.gt_param)


@needs_official
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batches_equal_the_official_set_input(seed: int) -> None:
    """Same random stream (CPU) -> same canvases, targets, ground-truth parameters and labels as
    the official code."""
    model = po.official_train_model()
    torch.manual_seed(seed)
    model.set_input({"A_paths": ["x"] * 64})
    ours = make_generator().sample(64, torch.Generator().manual_seed(seed))
    canvas_diff = float((ours.canvas - model.old).abs().max())
    target_diff = float((ours.target - model.render).abs().max())
    print(f"seed {seed}: canvas diff {canvas_diff:.2e}, target diff {target_diff:.2e}")
    assert canvas_diff <= 1.0 / 255 and target_diff <= 1.0 / 255
    assert torch.equal(ours.gt_param, model.gt_param)
    assert torch.equal(ours.gt_decision, model.gt_decision)


@needs_brushes
def test_validity_rule_with_constructed_overlaps() -> None:
    gen = make_generator()
    alphas = torch.zeros(1, 4, 3, 8, 8)
    alphas[0, 0, :, :4, :] = 1  # stroke 0 covers the top half
    alphas[0, 1, :, :, :] = 1  # stroke 1 covers everything: it covers 100 % of stroke 0 -> invalid
    alphas[0, 2, :, 6:, :] = 1  # stroke 2 is elsewhere -> valid
    alphas[0, 3, :, :4, :] = (
        1  # stroke 3 covers all of stroke 0 and stroke 1 is invalid -> invalid because of 0
    )
    assert gen._validity(alphas).tolist() == [[1.0, 0.0, 1.0, 0.0]]
    # an invalid earlier stroke does not invalidate later ones
    alphas2 = torch.zeros(1, 3, 3, 8, 8)
    alphas2[0, 0] = 1
    alphas2[0, 1] = 1  # covers stroke 0 fully -> invalid
    alphas2[0, 2, :, :2, :2] = (
        1  # small, overlaps stroke 1's area only through stroke 0: covers 4/64 of 0 -> valid
    )
    assert gen._validity(alphas2).tolist() == [[1.0, 0.0, 1.0]]
    # threshold 0.75: 3/4 covered -> invalid, just under -> valid
    a3 = torch.zeros(1, 2, 3, 8, 8)
    a3[0, 0, :, :4, :] = 1
    a3[0, 1, :, :3, :] = 1  # covers 6/8 rows of stroke 0 = 0.75 exactly -> invalid (iou >= 0.75)
    assert gen._validity(a3).tolist() == [[1.0, 0.0]]
    a3[0, 1, :, 2:, :] = 0
    a3[0, 1, :, :2, :] = 1  # covers half -> valid
    assert gen._validity(a3).tolist() == [[1.0, 1.0]]


@needs_brushes
def test_statistics_match_the_official_generator() -> None:
    """6.92 +- 0.91 valid strokes of 8 and the official histogram, over the same number of patches.

    Tolerances: the standard error of a mean of 12,800 patches is 0.91 / sqrt(12800) = 0.008; the
    difference of two
    independent means has standard deviation 0.011, so 0.06 is more than 5 sigma. Histogram bins: 5
    sigma of the
    difference of two multinomial proportions plus 0.001.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = make_generator(device)
    valid = draw(gen, 50, 256, seed=123)  # 12,800 patches
    assert valid.numel() == OFFICIAL_PATCHES
    assert abs(float(valid.mean()) - OFFICIAL_MEAN) < 0.06
    assert abs(float(valid.std()) - OFFICIAL_STD) < 0.04
    n = OFFICIAL_PATCHES
    for k, ref_count in enumerate(OFFICIAL_HISTOGRAM):
        p_ref = ref_count / n
        p = float((valid == k).sum()) / n
        sigma = math.sqrt(2 * max(p_ref, 1 / n) * (1 - p_ref) / n)
        assert abs(p - p_ref) < 5 * sigma + 0.001, (k, p, p_ref)


@needs_brushes
@pytest.mark.gpu
def test_gpu_generator_is_deterministic_and_fast() -> None:
    import time

    gen = make_generator("cuda")
    a = gen.sample(64, torch.Generator(device="cuda").manual_seed(0))
    b = gen.sample(64, torch.Generator(device="cuda").manual_seed(0))
    assert torch.equal(a.target, b.target) and torch.equal(a.gt_decision, b.gt_decision)
    assert a.target.is_cuda
    g = torch.Generator(device="cuda").manual_seed(1)
    gen.sample(64, g)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        gen.sample(64, g)
    torch.cuda.synchronize()
    per_batch = (time.perf_counter() - t0) / 20
    print(
        f"GPU generator: {per_batch * 1000:.1f} ms per batch of 64 = {64 / per_batch:.0f} patches/s"
    )
    # the official training step (data + model + Hungarian) takes 0.237 s per 64 patches
    assert per_batch < 0.05


def test_config_defaults_follow_the_official_recipe() -> None:
    cfg = SyntheticConfig()
    assert (cfg.patch_size, cfg.strokes_per_patch, cfg.low, cfg.span, cfg.iou_threshold) == (
        32,
        8,
        0.2,
        0.5,
        0.75,
    )
