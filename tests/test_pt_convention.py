"""Tests of the stroke-parameter convention (T-019): equivalence classes verified by rendering, the
invariant D_L1, and the three convention options (flags; defaults unchanged)."""

from __future__ import annotations

import pytest
import torch

from brushstroke.eval.pt_strokes import INVARIANT_LEVELS, invariant_l1
from brushstroke.renderer.pt import load_pt_brushes, pt_param2stroke
from brushstroke.stroke_model import (
    PaintTransformer,
    PaintTransformerLoss,
    PTLossConfig,
    PTModelConfig,
    SyntheticStrokeGenerator,
    init_official_weights,
    resolve_loss_config,
)
from brushstroke.stroke_model.convention import (
    LEVELS,
    class_indicator,
    stroke_l1_sum,
    variant_specs,
    variants,
    wrap_theta,
)
from brushstroke.stroke_model.model import positive_convention
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
SMALL = PTModelConfig(hidden_dim=64, heads=4, enc_layers=1, dec_layers=1, dropout=0.0)
SIZE = 48


def random_strokes(n: int, seed: int = 0) -> torch.Tensor:
    """Ground-truth-like strokes ``(n, 5)``: ``x, y, w, h`` in [0.2, 0.7], ``theta`` in [0, 1)."""
    u = torch.rand(n, 5, generator=torch.Generator().manual_seed(seed))
    u[:, :4] = 0.2 + 0.5 * u[:, :4]
    return u


def alpha_of(shape: torch.Tensor, brushes: torch.Tensor, mode: str) -> torch.Tensor:
    param8 = torch.cat([shape, torch.ones(len(shape), 3)], dim=1)
    return pt_param2stroke(param8, SIZE, SIZE, brushes, mode)[1][:, 0]


# ------------------------------------------------------------------ the classes (no rendering)


def test_variant_counts_and_identity_first() -> None:
    sizes = {level: len(variant_specs(level)) for level in LEVELS}
    assert sizes == {"none": 1, "exact": 2, "mirror": 4, "rot180": 8, "swap": 16}
    s = random_strokes(7)
    for level in LEVELS:
        assert torch.equal(variants(s, level)[0], s)
    with pytest.raises(ValueError):
        variant_specs("nope")


def test_wrap_theta_period_two() -> None:
    d = torch.tensor([0.3, -0.3, 1.2, -1.2, 2.0, 3.4, -2.6])
    w = wrap_theta(d)
    assert float(w.min()) >= -1 and float(w.max()) < 1
    assert torch.allclose(wrap_theta(d + 2), w, atol=1e-5)
    assert float(wrap_theta(torch.tensor(1.0))) == -1.0


def test_none_level_is_the_official_distance() -> None:
    a, b = random_strokes(40, 1), random_strokes(40, 2)
    assert torch.equal(stroke_l1_sum(a, b, "none"), (a - b).abs().sum(-1))


def test_invariant_distance_is_constant_on_a_class() -> None:
    """Every variant of the prediction has the same class-minimal distance to any target."""
    pred, gt = random_strokes(60, 3), random_strokes(60, 4)
    for level in LEVELS[1:]:
        reference = stroke_l1_sum(pred, gt, level)
        for v in variants(pred, level):
            assert torch.allclose(stroke_l1_sum(v, gt, level), reference, atol=1e-5), level
            # shifting theta by the exact period changes nothing either
            shifted = v.clone()
            shifted[..., 4] += 2
            assert torch.allclose(stroke_l1_sum(shifted, gt, level), reference, atol=1e-5)


def test_invariant_distance_never_exceeds_the_raw_one_and_nests() -> None:
    pred, gt = random_strokes(80, 5), random_strokes(80, 6)
    raw = stroke_l1_sum(pred, gt, "none")
    previous = raw
    for level in LEVELS[1:]:
        d = stroke_l1_sum(pred, gt, level)
        assert bool((d <= previous + 1e-6).all()), level
        previous = d
    assert bool((stroke_l1_sum(gt, gt, "mirror") < 1e-6).all())


def test_flipped_prediction_is_free_under_the_invariant_but_not_the_raw_distance() -> None:
    gt = random_strokes(50, 7)
    flipped = gt.clone()
    flipped[:, 2:4] *= -1
    flipped[:, 4] += 1  # (-w, -h, theta + 1): the exact class of gt
    mirrored = gt.clone()
    mirrored[:, 3] *= -1  # (w, -h, theta): the mirror class of gt
    assert float(stroke_l1_sum(flipped, gt, "none").mean()) > 1.0
    assert float(stroke_l1_sum(flipped, gt, "exact").max()) < 1e-5
    assert float(stroke_l1_sum(mirrored, gt, "exact").mean()) > 0.5
    assert float(stroke_l1_sum(mirrored, gt, "mirror").max()) < 1e-5
    assert bool((class_indicator(mirrored, gt, "mirror") == 2).all())


def test_invariant_l1_is_a_per_parameter_mean_and_ignores_colour_columns() -> None:
    a, b = random_strokes(10, 8), random_strokes(10, 9)
    assert torch.allclose(invariant_l1(a, b, "none"), (a - b).abs().mean(-1), atol=1e-6)
    a8 = torch.cat([a, torch.rand(10, 3)], dim=1)
    assert torch.allclose(invariant_l1(a8, b, "mirror"), invariant_l1(a, b, "mirror"))
    assert set(INVARIANT_LEVELS) <= set(LEVELS)


def test_invariant_distance_has_finite_gradients() -> None:
    pred = random_strokes(20, 10).requires_grad_(True)
    stroke_l1_sum(pred, random_strokes(20, 11), "swap").sum().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# ------------------------------------------------------------------ the classes, by rendering


@needs_brushes
@pytest.mark.parametrize("variant", ["train", "inference"])
def test_exact_class_renders_identically(variant: str) -> None:
    """``(w, h, theta + 1)`` and ``(-w, -h, theta)`` paint the same alpha map (the exact class)."""
    brushes = load_pt_brushes(variant, po.BRUSHES)
    mode = "pt_train" if variant == "train" else "pt_infer"
    s = random_strokes(300, 12)
    base = alpha_of(s, brushes, mode)
    flipped = variants(s, "exact")[1]
    other = alpha_of(flipped, brushes, mode)
    assert float((other - base).abs().mean()) < 1e-4  # train: bit-exact; infer: a few edge pixels
    # theta has the exact period 2
    shifted = s.clone()
    shifted[:, 4] += 2
    # float rounding of sin / cos only
    assert float((alpha_of(shifted, brushes, mode) - base).abs().mean()) < 1e-5


@needs_brushes
def test_mirror_class_is_close_but_not_exact_and_the_rest_is_far() -> None:
    """The mirror variant differs on a fraction of the stroke (the brush mask is nearly, not
    exactly, symmetric); the other sign / rotation / exchange variants differ twice as much."""
    brushes = load_pt_brushes("train", po.BRUSHES)
    s = random_strokes(400, 13)
    base = alpha_of(s, brushes, "pt_train")
    area = float(base.mean())
    errors = {}
    for spec, v in zip(variant_specs("swap"), variants(s, "swap"), strict=True):
        errors[spec] = float((alpha_of(v, brushes, "pt_train") - base).abs().mean()) / area
    exact = [errors[spec] for spec in variant_specs("exact")]
    mirror = [errors[spec] for spec in variant_specs("mirror")[2:]]
    others = [e for spec, e in errors.items() if spec not in variant_specs("mirror")]
    assert max(exact) < 1e-3
    assert all(0.05 < e < 0.25 for e in mirror), mirror  # about 14 % of the stroke area
    assert min(others) > 1.5 * max(mirror), (min(others), max(mirror))


# ------------------------------------------------------------------ option A (model)


def test_positive_convention_ranges_and_other_columns() -> None:
    p = torch.randn(3, 8, 8) * 3
    out = positive_convention(p)
    assert bool((out[..., 2:4] > 0).all())
    assert bool(((out[..., 4] >= 0) & (out[..., 4] < 1)).all())
    assert torch.equal(out[..., :2], p[..., :2]) and torch.equal(out[..., 5:], p[..., 5:])
    # gradient flows to w, h and theta
    q = p.clone().requires_grad_(True)
    positive_convention(q)[..., 2:5].sum().backward()
    assert q.grad is not None and bool((q.grad[..., 2:5] != 0).all())


def test_model_convention_flag() -> None:
    img = torch.rand(2, 3, 32, 32)
    torch.manual_seed(0)
    raw = PaintTransformer(SMALL).eval()
    torch.manual_seed(0)
    pos = PaintTransformer(PTModelConfig(**{**SMALL.__dict__, "convention": "positive"})).eval()
    pos.load_state_dict(raw.state_dict())
    with torch.no_grad():
        shape_raw, _, _ = raw.strokes(img, img)
        shape_pos, _, _ = pos.strokes(img, img)
    assert torch.equal(shape_pos, positive_convention(shape_raw))
    assert bool((shape_pos[..., 2:4] > 0).all())
    with pytest.raises(ValueError):
        PaintTransformer(PTModelConfig(convention="nope"))


# ------------------------------------------------------------------ options B and C (loss)


def make_batch(size: int = 16):  # type: ignore[no-untyped-def]
    brushes = load_pt_brushes("train", po.BRUSHES, "cpu")
    return brushes, SyntheticStrokeGenerator(brushes).sample(size, torch.Generator().manual_seed(0))


@needs_brushes
def test_default_flags_leave_the_loss_unchanged() -> None:
    brushes, batch = make_batch()
    torch.manual_seed(1)
    net = PaintTransformer(SMALL).eval()
    init_official_weights(net)
    shape, colour, logits = net.strokes(batch.target, batch.canvas)
    default = PaintTransformerLoss(brushes, PTLossConfig(matcher="scipy"))(
        shape, colour, logits, batch
    )
    explicit = PaintTransformerLoss(
        brushes, PTLossConfig(matcher="scipy", l1_invariance="none", lambda_sign=0.0)
    )(shape, colour, logits, batch)
    assert default.sign is None and explicit.sign is None
    assert torch.equal(default.total, default.pixel + default.gt + default.w + default.decision)
    assert torch.equal(default.total, explicit.total)
    assert resolve_loss_config(PTLossConfig()).l1_invariance == "none"
    with pytest.raises(ValueError):
        resolve_loss_config(PTLossConfig(l1_invariance="nope"))


@needs_brushes
def test_invariant_l1_term_ignores_the_class_of_the_prediction() -> None:
    """Option B: a prediction replaced by an equivalent one has the same L1 term and the same
    matching; the official term sees a huge error."""
    brushes, batch = make_batch()
    valid = batch.gt_decision.bool()
    perfect = batch.gt_param.clone()
    flipped = perfect.clone()
    flipped[..., 2:4] *= -1
    flipped[..., 4] += 1  # exact class
    logits = torch.where(valid, 5.0, -5.0)
    colour = batch.gt_color
    kwargs = {"matcher": "scipy"}
    official = PaintTransformerLoss(brushes, PTLossConfig(**kwargs))(flipped, colour, logits, batch)
    exact = PaintTransformerLoss(brushes, PTLossConfig(l1_invariance="exact", **kwargs))
    out = exact(flipped, colour, logits, batch)
    reference = exact(perfect, colour, logits, batch)
    assert float(official.gt) > 0.5
    assert float(out.gt) < 1e-5 and float(reference.gt) < 1e-5
    assert torch.equal(out.assign, reference.assign)
    assert float(out.w) < 1e-3  # the Wasserstein distance is invariant to the signs already


@needs_brushes
def test_negative_size_penalty_option() -> None:
    """Option C: ``lambda_sign * mean(relu(-w) + relu(-h))``, added to the total (0 if w, h > 0)."""
    brushes, batch = make_batch(8)
    shape = batch.gt_param.clone()
    logits = torch.zeros(8, 8)
    cfg = PTLossConfig(matcher="scipy", lambda_sign=10.0)
    loss = PaintTransformerLoss(brushes, cfg)
    ok = loss(shape, batch.gt_color, logits, batch)
    assert ok.sign is not None and float(ok.sign) == 0.0
    bad = shape.clone()
    bad[..., 2] = -0.3
    bad[..., 3] = -0.1
    out = loss(bad, batch.gt_color, logits, batch)
    assert out.sign is not None and float(out.sign) == pytest.approx(10 * 0.4, rel=1e-5)
    assert float(out.total) == pytest.approx(
        float(out.pixel + out.gt + out.w + out.decision + out.sign), rel=1e-6
    )
    raw = bad.clone().requires_grad_(True)
    loss(raw, batch.gt_color, logits, batch).sign.backward()  # type: ignore[union-attr]
    assert raw.grad is not None and bool((raw.grad[..., 2:4] < 0).all())  # pushes w, h up
