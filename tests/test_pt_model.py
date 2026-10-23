"""Tests of the Paint Transformer model (T-012): size, shapes, official-weights equivalence."""

from __future__ import annotations

import pytest
import torch

from brushstroke.stroke_model import (
    PaintTransformer,
    PTModelConfig,
    count_parameters,
    init_official_weights,
)
from tests import pt_official as po

needs_weights = pytest.mark.skipif(
    not (po.have_official() and po.have_weights()),
    reason="official code/weights missing: run `python -m scripts.pt_reference_study fetch`",
)


def test_parameter_count_is_exactly_the_official_one() -> None:
    assert count_parameters(PaintTransformer()) == 9_072_902


def test_forward_shapes_and_colour_modes() -> None:
    img, canvas = torch.rand(3, 3, 32, 32), torch.rand(3, 3, 32, 32)
    net = PaintTransformer().eval()
    param, decision = net(img, canvas)
    assert param.shape == (3, 8, 5) and decision.shape == (3, 8, 1)
    shape, colour, logits = net.strokes(img, canvas)
    assert shape.shape == (3, 8, 5) and colour.shape == (3, 8, 3) and logits.shape == (3, 8)
    # the sampled colour is the image at the centre: a constant image gives that constant
    const = torch.full((3, 3, 32, 32), 0.3)
    assert torch.allclose(
        PaintTransformer.sample_colour(const, torch.full((3, 8, 5), 0.5)), torch.tensor(0.3)
    )
    predicted = PaintTransformer(PTModelConfig(colour_mode="predicted")).eval()
    shape, colour, logits = predicted.strokes(img, canvas)
    assert shape.shape == (3, 8, 5) and colour.shape == (3, 8, 3)
    with pytest.raises(ValueError):
        PaintTransformer(PTModelConfig(colour_mode="nope"))


def test_official_initialization_statistics() -> None:
    net = PaintTransformer()
    init_official_weights(net)
    std = float(net.enc_img[1].weight.std())
    assert abs(std - 0.02) < 0.002
    assert float(net.enc_img[1].bias.abs().max()) == 0.0
    assert abs(float(net.enc_img[2].weight.mean()) - 1.0) < 0.02


@needs_weights
def test_official_weights_load_and_reproduce_the_official_outputs() -> None:
    official = po.official_network().Painter(5, 8, 256, 8, 3, 3).eval()
    state = torch.load(po.OFFICIAL / "model.pth", map_location="cpu", weights_only=True)
    official.load_state_dict(state)
    ours = PaintTransformer().eval()
    ours.load_state_dict(state)  # the official parameter names are kept
    torch.manual_seed(0)
    img, canvas = torch.rand(16, 3, 32, 32), torch.rand(16, 3, 32, 32)
    with torch.no_grad():
        ref_param, ref_decision = official(img, canvas)
        param, decision = ours(img, canvas)
    diff = max(float((param - ref_param).abs().max()), float((decision - ref_decision).abs().max()))
    print(f"official weights: max abs output difference {diff:.2e}")
    assert diff <= 1e-5
    assert any(float(ref_decision.max()) > 0 for _ in [0])  # the trained net is not degenerate
