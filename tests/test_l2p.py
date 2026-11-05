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
    assert err["mse"] < 2e-3  # measured 6.1e-4 on 1000 strokes (docs/design/learning_to_paint.md)
    if (OFFICIAL / "actor.pkl").exists():
        load_actor(OFFICIAL / "actor.pkl")


@needs_official
def test_rollout_matches_a_transcription_of_the_official_test_script() -> None:
    """CPU, float inputs: every global-stroke canvas and every divided-step canvas equals a line-by-line transcription of
    ``baseline/test.py`` (divide 2, max_step 8). On CUDA the closed loop drifts at kernel-level float noise, hence CPU."""
    cv2 = pytest.importorskip("cv2")
    img = OFFICIAL / "image" / "lisa.png"
    if not (OFFICIAL / "actor.pkl").exists() or not img.exists():
        pytest.skip("official actor or sample image not fetched")
    width, divide, max_step = 128, 2, 8
    actor, dec = load_actor(OFFICIAL / "actor.pkl"), load_renderer(OFFICIAL / "renderer.pkl")
    imgf = cv2.resize(cv2.imread(str(img), cv2.IMREAD_COLOR), (256, 256)).astype(np.float32) / 255.0
    origin = (256, 256)
    r = torch.arange(width, dtype=torch.float32) / (width - 1.0)
    coord = torch.stack([r[:, None].expand(width, width), r[None, :].expand(width, width)])[None]

    def odecode(x: torch.Tensor, canvas: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = x.view(-1, 13)
        stroke = (1 - dec(x[:, :10])).view(-1, width, width, 1)
        cs = (stroke * x[:, -3:].view(-1, 1, 1, 3)).permute(0, 3, 1, 2)
        stroke = stroke.permute(0, 3, 1, 2).reshape(-1, 5, 1, width, width)
        cs = cs.reshape(-1, 5, 3, width, width)
        out = []
        for i in range(5):
            canvas = canvas * (1 - stroke[:, i]) + cs[:, i]
            out.append(canvas)
        return canvas, out

    def smooth(a: np.ndarray) -> np.ndarray:
        n = divide * width

        def sp(tx: int, ty: int) -> None:
            if tx in (0, n - 1) or ty in (0, n - 1):
                return
            a[tx, ty] = (
                a[tx, ty]
                + a[tx + 1, ty]
                + a[tx, ty + 1]
                + a[tx - 1, ty]
                + a[tx, ty - 1]
                + a[tx + 1, ty - 1]
                + a[tx - 1, ty + 1]
                + a[tx - 1, ty - 1]
                + a[tx + 1, ty + 1]
            ) / 9

        for p in range(divide):
            for q in range(divide):
                x, y = p * width, q * width
                for k in range(width):
                    sp(x + k, y + width - 1)
                    if q != divide - 1:
                        sp(x + k, y + width)
                for k in range(width):
                    sp(x + width - 1, y + k)
                    if p != divide - 1:
                        sp(x + width, y + k)
        return a

    def l2s(x: np.ndarray) -> np.ndarray:
        x = x.reshape(divide, width, divide, width, 3)
        return np.transpose(x, (0, 2, 1, 3, 4)).reshape(divide * divide, width, width, 3)

    def nchw(x: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.transpose(x, (0, 3, 1, 2))).float()

    steps = max_step // 2
    ones = torch.ones(1, 1, width, width)
    with torch.no_grad():
        patch = nchw(l2s(cv2.resize(imgf, (width * divide, width * divide))))
        im = nchw(cv2.resize(imgf, (width, width))[None])
        canvas = torch.zeros(1, 3, width, width)
        ref: dict[int, np.ndarray] = {}
        for i in range(steps):
            a = actor(torch.cat([canvas, im, ones * i / steps, coord], 1))
            canvas, res = odecode(a, canvas)
            for j, c in enumerate(res):
                ref[5 * i + j + 1] = cv2.resize(np.transpose(c[0].numpy(), (1, 2, 0)), origin)
        c = nchw(l2s(cv2.resize(np.transpose(canvas[0].numpy(), (1, 2, 0)), (256, 256))))
        n1 = 5 * steps
        for i in range(steps):
            a = actor(
                torch.cat(
                    [
                        c,
                        patch,
                        ones.expand(divide**2, 1, width, width) * i / steps,
                        coord.expand(divide**2, 2, width, width),
                    ],
                    1,
                )
            )
            c, res = odecode(a, c)
            big = (
                np.transpose(res[4].numpy(), (0, 2, 3, 1))
                .reshape(divide, divide, width, width, 3)
                .transpose(0, 2, 1, 3, 4)
                .reshape(divide * width, divide * width, 3)
            )
            ref[n1 + (i + 1) * divide**2 * 5] = cv2.resize(smooth(big), origin)
        target = torch.from_numpy(imgf).permute(2, 0, 1)
        painting = rollout(actor, dec, target, RolloutConfig(max_step=max_step, divide=divide))
    for k, want in ref.items():
        got = painting.canvas(k).permute(1, 2, 0).numpy()
        assert float(np.abs(np.clip(want, 0, 1) - got).max()) < 1e-6, k
