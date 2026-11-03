"""Learning to Paint pieces (T-028): analytic Bezier rasterizer, neural renderer, rollout and producer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from brushstroke.baselines.l2p.actor import Actor, load_actor
from brushstroke.baselines.l2p.bezier import draw_reference, stroke_coverage
from brushstroke.baselines.l2p.producer import L2PProducer
from brushstroke.baselines.l2p.renderer import (
    FCN,
    RendererTrainConfig,
    decode,
    load_renderer,
    lr_at,
    random_strokes,
    renderer_error,
    train_renderer,
)
from brushstroke.baselines.l2p.rollout import RolloutConfig, large2small, rollout, small2large
from brushstroke.eval.producers import validate_result

OFFICIAL = Path("data/raw/l2p_official")
needs_official = pytest.mark.skipif(
    not (OFFICIAL / "renderer.pkl").exists(), reason="official weights not fetched (outside git)"
)


def test_bezier_rasterizer_matches_the_official_cv2_draw() -> None:
    pytest.importorskip("cv2")
    f = np.random.default_rng(0).uniform(0, 1, (12, 10))
    ref = np.stack([1 - draw_reference(x) for x in f])
    out = stroke_coverage(torch.tensor(f).float()).numpy()
    assert np.abs(ref - out).max() < 1e-5  # same integer geometry; float rounding only


def test_a_stroke_covers_something_and_the_value_follows_w() -> None:
    f = torch.tensor([[0.2, 0.2, 0.5, 0.5, 0.8, 0.8, 0.3, 0.3, 0.7, 0.7]])
    cov = stroke_coverage(f)[0]
    assert cov.shape == (128, 128) and 0.69 < float(cov.max()) < 0.71 and float(cov.min()) == 0.0


def test_renderer_has_the_official_architecture_and_output_range() -> None:
    net = FCN()
    out = net(torch.rand(3, 10))
    assert out.shape == (3, 128, 128) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert list(net.state_dict())[:2] == ["fc1.weight", "fc1.bias"]


def test_decode_is_the_over_operator_in_bundle_order() -> None:
    net = FCN().eval()
    s = torch.rand(2, 65)
    canvas = torch.zeros(2, 3, 128, 128)
    out, steps = decode(s, canvas, net)
    ref = canvas
    x = s.reshape(2, 5, 13)
    for i in range(5):
        cov = (1 - net(x[:, i, :10]))[:, None]
        ref = ref * (1 - cov) + cov * x[:, i, 10:13].reshape(2, 3, 1, 1)
        torch.testing.assert_close(steps[i], ref, atol=1e-5, rtol=0)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)


def test_training_loop_runs_resumes_and_lowers_the_loss(tmp_path: Path) -> None:
    cfg = RendererTrainConfig(
        steps=40,
        batch_size=8,
        log_every=20,
        eval_strokes=16,
        output_dir=str(tmp_path),
        device="cpu",
    )
    train_renderer(cfg, log=False)
    assert (tmp_path / "renderer.pt").exists() and len(
        (tmp_path / "history.jsonl").read_text().splitlines()
    ) == 2
    net = load_renderer(tmp_path / "renderer.pt")
    s = random_strokes(8, torch.Generator().manual_seed(1), "cpu")
    assert renderer_error(net, s)["mse"] < renderer_error(FCN().eval(), s)["mse"]
    assert lr_at(0, cfg) == 1e-4 and lr_at(200_000, cfg) == 1e-5 and lr_at(400_000, cfg) == 1e-6


def test_patch_layout_roundtrip() -> None:
    x = np.random.default_rng(1).uniform(size=(512, 512, 3)).astype("float32")
    assert np.array_equal(small2large(large2small(x, 4), 4), x)
    assert np.array_equal(large2small(x, 4)[1], x[0:128, 128:256])  # row-major patches


def _tiny_models(tmp_path: Path) -> tuple[Actor, FCN, Path, Path]:
    torch.manual_seed(0)
    actor, renderer = Actor().eval(), FCN().eval()
    ap, rp = tmp_path / "actor.pt", tmp_path / "renderer.pt"
    torch.save(actor.state_dict(), ap)
    torch.save(renderer.state_dict(), rp)
    return actor, renderer, ap, rp


def test_rollout_counts_order_and_prefix_canvases(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    actor, renderer, _, _ = _tiny_models(tmp_path)
    cfg = RolloutConfig(max_step=4, divide=2)
    target = torch.rand(3, 256, 256)
    painting = rollout(actor, renderer, target, cfg)
    assert painting.n_total == 2 * 5 + 2 * 4 * 5 and painting.strokes.shape == (
        painting.n_total,
        13,
    )
    assert float(painting.canvas(0).abs().max()) == 0.0
    last = painting.canvas(painting.n_total)
    assert last.shape == (3, 256, 256) and float(last.min()) >= 0.0 and float(last.max()) <= 1.0
    # a stroke of patch p only changes patch p (patches are independent), and only after the global stage
    k = painting.n_stage1 + 7  # first step, patch 1, 2nd stroke
    prev = painting.canvas(k - 1)
    now = painting.canvas(k)
    changed = (now - prev).abs().amax(0) > 1e-6
    assert (
        changed.any() and not changed[136:, :].any() and not changed[:, :126].any()
    )  # patch 1 = rows 0..127, cols 128..255 (+ the smoothed seam)
    assert torch.equal(painting.canvas(k), painting.canvas(k))  # deterministic


def test_producer_returns_a_valid_harness_result(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    _, _, ap, rp = _tiny_models(tmp_path)
    prod = L2PProducer({0: str(ap)}, rp, RolloutConfig(max_step=4, divide=2), device="cpu")
    target = torch.rand(3, 256, 256)
    grid = [0, 5, 10, 25, 50, 1000]
    prod.load(0)
    res = prod.paint(target, grid, 0)
    validate_result(res, grid, 256)
    assert (
        res.checkpoints == [0, 5, 10, 25, 50]
        and res.n_total == 50
        and res.background == (0.0, 0.0, 0.0)
    )
    assert res.meta["upsampled"] is True and res.meta["replicate_kind"] == "training seed"
    with pytest.raises(KeyError):
        prod.load(3)


@needs_official
def test_official_weights_load_strictly_and_approximate_the_analytic_strokes() -> None:
    net = load_renderer(OFFICIAL / "renderer.pkl")
    s = random_strokes(200, torch.Generator().manual_seed(5), "cpu")
    err = renderer_error(net, s)
    assert err["mse"] < 2e-3  # measured 5.8e-4 on 1000 strokes (docs/design/learning_to_paint.md)
    if (OFFICIAL / "actor.pkl").exists():
        load_actor(OFFICIAL / "actor.pkl")
