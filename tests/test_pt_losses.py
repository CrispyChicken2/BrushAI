"""Tests of the Paint Transformer losses (T-012)."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import pytest
import torch

from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model import (
    PaintTransformer,
    PaintTransformerLoss,
    PTLossConfig,
    PTModelConfig,
    SyntheticStrokeGenerator,
    gaussian_w_distance,
    init_official_weights,
    resolve_loss_config,
)
from brushstroke.stroke_model.losses import match_gpu, match_scipy, total_cost
from brushstroke.utils.config import load_config
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
needs_official = pytest.mark.skipif(
    not (po.have_brushes() and po.have_official()),
    reason="official code/assets missing: run fetch_pt_brushes + pt_reference_study fetch",
)


# ------------------------------------------------------------------ Wasserstein distance


def test_wasserstein_zero_for_identical_and_symmetric() -> None:
    torch.manual_seed(0)
    a = torch.rand(50, 5) * 0.5 + 0.2
    b = torch.rand(50, 5) * 0.5 + 0.2
    assert float(gaussian_w_distance(a, a).abs().max()) < 1e-5
    assert torch.allclose(gaussian_w_distance(a, b), gaussian_w_distance(b, a), atol=1e-5)
    assert float(gaussian_w_distance(a, b).min()) >= -1e-6


def test_wasserstein_closed_form_axis_aligned() -> None:
    """Commuting covariances: W2^2 = |dmu|^2 + (w1/2 - w2/2)^2 + (h1/2 - h2/2)^2."""
    p1 = torch.tensor([0.5, 0.5, 0.4, 0.2, 0.0])
    p2 = torch.tensor([0.6, 0.7, 0.2, 0.3, 0.0])
    expected = 0.1**2 + 0.2**2 + (0.2 - 0.1) ** 2 + (0.1 - 0.15) ** 2
    assert float(gaussian_w_distance(p1, p2)) == pytest.approx(expected, abs=1e-6)
    # same centre and the same rotation: the closed form does not depend on theta
    for theta in (0.1, 0.25, 0.7):
        q1, q2 = p1.clone(), p2.clone()
        q1[:2] = q2[:2] = 0.5
        q1[4] = q2[4] = theta
        assert float(gaussian_w_distance(q1, q2)) == pytest.approx(
            (0.2 - 0.1) ** 2 + (0.1 - 0.15) ** 2, abs=1e-5
        )


def test_wasserstein_rotation_by_a_quarter_turn_swaps_axes() -> None:
    a = torch.tensor([0.5, 0.5, 0.4, 0.2, 0.0])
    b = torch.tensor([0.5, 0.5, 0.2, 0.4, 0.5])  # same rectangle, rotated by 90 degrees
    assert float(gaussian_w_distance(a, b)) < 1e-5


# ------------------------------------------------------------------ matching


def brute_force(cost: torch.Tensor, valid: torch.Tensor) -> float:
    n = cost.shape[0]
    cols = valid.nonzero().flatten().tolist()
    best = math.inf
    for rows in itertools.permutations(range(n), len(cols)):
        best = min(best, sum(float(cost[r, c]) for r, c in zip(rows, cols, strict=True)))
    return best


@pytest.mark.parametrize("n", [3, 4, 5])
def test_matching_equals_brute_force(n: int) -> None:
    g = torch.Generator().manual_seed(n)
    cost = torch.rand(12, n, n, generator=g) * 3 - 1  # include negative costs
    valid = torch.rand(12, n, generator=g) > 0.3
    valid[:, 0] = True
    expected = torch.tensor([brute_force(cost[i], valid[i]) for i in range(12)])
    for assign in (match_scipy(cost, valid), match_gpu(cost, valid)):
        assert torch.allclose(total_cost(cost, valid, assign), expected, atol=1e-5)
        assert (assign[~valid] == -1).all()
        for i in range(12):  # a one-to-one assignment
            picked = assign[i][valid[i]]
            assert picked.unique().numel() == picked.numel()


def test_gpu_matcher_rejects_large_sets() -> None:
    with pytest.raises(ValueError):
        match_gpu(torch.zeros(1, 9, 9), torch.ones(1, 9, dtype=torch.bool))


# ------------------------------------------------------------------ presets


def test_presets_resolve_from_yaml_style_configs() -> None:
    official = resolve_loss_config(PTLossConfig())
    assert (official.pos_weight, official.lambda_decision, official.lambda_pixel) == (10, 10, 10)
    assert (official.lambda_gt, official.lambda_w, official.matching_cost) == (1, 10, "logit")
    paper = resolve_loss_config(PTLossConfig(preset="paper_weights"))
    assert (paper.pos_weight, paper.lambda_decision, paper.lambda_pixel) == (8, 1, 1)
    assert (paper.lambda_gt, paper.lambda_w, paper.matching_cost) == (1, 10, "bce")
    custom = resolve_loss_config(PTLossConfig(preset="custom", lambda_pixel=3.0))
    assert custom.lambda_pixel == 3.0
    with pytest.raises(ValueError):
        resolve_loss_config(PTLossConfig(preset="nope"))


def test_presets_are_selectable_from_the_yaml_files() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "pt"
    official = resolve_loss_config(load_config(PTLossConfig, root / "loss_official.yaml"))
    paper = resolve_loss_config(load_config(PTLossConfig, root / "loss_paper.yaml"))
    assert official.pos_weight == 10 and paper.pos_weight == 8 and paper.matching_cost == "bce"
    assert load_config(PTModelConfig, root / "model.yaml") == PTModelConfig()


# ------------------------------------------------------------------ the full loss


def make_batch(device: str, size: int, seed: int = 0):  # type: ignore[no-untyped-def]
    brushes = load_pt_brushes("train", po.BRUSHES, device)
    gen = SyntheticStrokeGenerator(brushes)
    return brushes, gen.sample(size, torch.Generator(device=device).manual_seed(seed))


SMALL = PTModelConfig(hidden_dim=64, heads=4, enc_layers=1, dec_layers=1, dropout=0.0)


@needs_brushes
@pytest.mark.parametrize("preset", ["official", "paper_weights"])
def test_every_term_has_finite_nonzero_gradients(preset: str) -> None:
    """At init the strokes are sub-pixel so the pixel term has exactly zero gradient (as in
    the official code); after a few optimization steps all four terms carry a gradient."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brushes, batch = make_batch(device, 16)
    torch.manual_seed(0)
    net = PaintTransformer(SMALL).to(device)
    init_official_weights(net)
    loss = PaintTransformerLoss(brushes, PTLossConfig(preset=preset, matcher="scipy"))
    params = [p for p in net.parameters() if p.requires_grad]

    def gradient_norms() -> dict[str, float]:
        norms = {}
        for name in ("pixel", "gt", "w", "decision"):
            term = getattr(loss(*net.strokes(batch.target, batch.canvas), batch), name)
            assert torch.isfinite(term) and float(term) > 0, name
            grads = torch.autograd.grad(term, params, allow_unused=True)
            assert all(g is None or torch.isfinite(g).all() for g in grads), name
            norms[name] = sum(float(g.norm()) for g in grads if g is not None)
        return norms

    initial = gradient_norms()
    assert all(initial[k] > 0 for k in ("gt", "w", "decision")), initial
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    for _ in range(120):
        out = loss(*net.strokes(batch.target, batch.canvas), batch)
        opt.zero_grad()
        out.total.backward()
        opt.step()
    trained = gradient_norms()
    print(preset, "gradient norms", {k: round(v, 4) for k, v in trained.items()})
    assert all(math.isfinite(v) and v > 0 for v in trained.values()), trained


@needs_brushes
def test_loss_is_independent_of_the_matcher() -> None:
    brushes, batch = make_batch("cpu", 16)
    torch.manual_seed(1)
    net = PaintTransformer().eval()
    init_official_weights(net)
    with torch.no_grad():
        outs = [
            PaintTransformerLoss(brushes, PTLossConfig(matcher=m))(
                *net.strokes(batch.target, batch.canvas), batch
            )
            for m in ("scipy", "gpu")
        ]
    for name in ("pixel", "gt", "w", "decision"):
        assert float(getattr(outs[0], name)) == pytest.approx(
            float(getattr(outs[1], name)), rel=1e-5
        )


@needs_official
@pytest.mark.parametrize("seed", [0, 1])
def test_loss_terms_equal_the_official_ones(seed: int) -> None:
    """Same weights and batch (eval mode) -> same four loss terms as ``optimize_parameters``."""
    model = po.official_train_model()
    model.net_g.eval()
    torch.manual_seed(seed)
    model.set_input({"A_paths": ["x"] * 32})
    ours_net = PaintTransformer().eval()
    ours_net.load_state_dict(model.net_g.state_dict())
    batch = _official_batch(model)
    brushes = load_pt_brushes("train", po.BRUSHES, "cpu")
    with torch.no_grad():
        out = PaintTransformerLoss(brushes)(*ours_net.strokes(batch.target, batch.canvas), batch)
    model.optimize_parameters()  # computes the losses, then steps the (discarded) optimizer
    for name in ("pixel", "gt", "w", "decision"):
        ref = float(getattr(model, f"loss_{name}"))
        value = float(getattr(out, name))
        print(f"{name}: ours {value:.6f} official {ref:.6f}")
        assert value == pytest.approx(ref, rel=2e-4, abs=1e-5), name


def _official_batch(model):  # type: ignore[no-untyped-def]
    from brushstroke.stroke_model import PatchBatch

    return PatchBatch(
        canvas=model.old,
        target=model.render,
        gt_param=model.gt_param,
        gt_decision=model.gt_decision,
        gt_color=torch.zeros(*model.gt_param.shape[:2], 3),
    )


@needs_brushes
def test_overfitting_a_tiny_batch_drives_the_loss_down() -> None:
    """A small network must fit 4 patches: the total loss falls by more than 95 %."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brushes, batch = make_batch(device, 4)
    torch.manual_seed(0)
    net = PaintTransformer(SMALL).to(device)
    init_official_weights(net)
    loss = PaintTransformerLoss(brushes, PTLossConfig(matcher="scipy"))
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    first = None
    for _ in range(700):
        out = loss(*net.strokes(batch.target, batch.canvas), batch)
        first = first if first is not None else float(out.total)
        opt.zero_grad()
        out.total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    print(
        f"total {first:.3f} -> {float(out.total):.3f}",
        {k: round(float(getattr(out, k)), 4) for k in ("pixel", "gt", "w", "decision")},
    )
    assert float(out.total) < 0.05 * first
    assert float(out.decision) < 0.2 and float(out.w) < 0.5
