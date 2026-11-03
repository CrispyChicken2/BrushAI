"""Tests of the harness (T-026): SW-LPIPS, prefix consistency, the runner, failures, determinism."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from brushstroke.eval import metrics as em
from brushstroke.eval import producers as pr
from brushstroke.eval import runner as rn
from brushstroke.eval.protocol import load_protocol
from brushstroke.renderer.pipeline import RenderConfig
from brushstroke.strokes import StrokeSequence

SIZE = 64
PROTO = load_protocol()


def make_target(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(3, 8, 8, generator=g)
    return torch.nn.functional.interpolate(base[None], size=(SIZE, SIZE), mode="bilinear")[0].clamp(
        0, 1
    )


class FakeProducer:
    """Deterministic producer: the canvas moves linearly from a constant colour to the target over ``total`` strokes."""

    name = "fake"
    stochastic = False
    replicate_kind = "none"
    dagger_metrics: tuple[str, ...] = ()

    def __init__(self, total: int = 600, nan_at: int | None = None, fail: bool = False) -> None:
        self.total, self.nan_at, self.fail = total, nan_at, fail
        self.loaded: list[int] = []

    def load(self, seed: int) -> None:
        self.loaded.append(seed)

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> pr.PaintResult:
        if self.fail:
            raise RuntimeError("boom")
        cps = pr.usable_checkpoints(checkpoints, self.total)
        bg = torch.full_like(target, 0.5)
        canvases = torch.stack([bg + (target - bg) * min(1.0, n / 500) for n in cps])
        if self.nan_at is not None:
            canvases[self.nan_at] = float("nan")
        return pr.PaintResult(cps, canvases, self.total, (0.5, 0.5, 0.5), meta={"seed": seed})

    def control(self) -> dict[str, Any]:
        return {"note": "fake"}


def context(image_id: str = "img", seed: int = 0) -> rn.ImageContext:
    t = make_target(seed)
    sal = {
        "inspyrenet": torch.ones(SIZE, SIZE) * 0.5,
        "spectral_residual": torch.linspace(0, 1, SIZE * SIZE).reshape(SIZE, SIZE),
    }
    return rn.ImageContext(image_id, "dev", "photo", "photo_scene", t, sal)


# ------------------------------------------------------------------------ metrics


def test_sw_lpips_with_uniform_saliency_equals_lpips() -> None:
    t = make_target(1)
    canvases = torch.stack([make_target(2), make_target(3), torch.zeros(3, SIZE, SIZE)])
    m = em.canvas_metrics(canvases, t, {"u": torch.full((SIZE, SIZE), 0.3)}, (0.0, 0.1, 0.3))
    for lam in ("0", "0.1", "0.3"):
        assert m[f"sw_u_{lam}"] == pytest.approx(m["lpips_alex"], abs=1e-4)
    # the metric changes with a non-uniform map and with lambda; lambda = 1 is the plain LPIPS whatever the map
    ramp = torch.linspace(0, 1, SIZE * SIZE).reshape(SIZE, SIZE)
    m2 = em.canvas_metrics(canvases, t, {"r": ramp}, (0.0, 0.1, 0.3))
    assert m2["sw_r_0"] != pytest.approx(m2["lpips_alex"], abs=1e-4)
    assert em.sw_lpips(em.lpips_alex_map(canvases, t), ramp, 1.0).tolist() == pytest.approx(
        m["lpips_alex"], abs=1e-4
    )


def test_saliency_weights_sum_to_one_and_handle_an_empty_map() -> None:
    s = torch.rand(SIZE, SIZE)
    for lam in (0.0, 0.1, 0.3, 1.0):
        assert float(em.saliency_weights(s, lam).sum()) == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(
        em.saliency_weights(torch.zeros(SIZE, SIZE), 0.1), torch.full((SIZE, SIZE), 1 / SIZE**2)
    )
    with pytest.raises(ValueError):
        em.saliency_weights(s, 1.5)


def test_lpips_vgg_is_computed_in_chunks_and_the_metrics_have_the_expected_keys() -> None:
    t = make_target(1)
    c = torch.stack([make_target(i) for i in range(2, 7)])
    assert em.lpips_vgg_chunked(c, t, 2).tolist() == pytest.approx(
        em.lpips_vgg_chunked(c, t, 5).tolist(), abs=1e-6
    )
    m = em.canvas_metrics(c, t, {"a": torch.ones(SIZE, SIZE)}, (0.1,), low_res=32)
    assert {
        "l1",
        "psnr",
        "ssim",
        "lpips_alex",
        "lpips_vgg",
        "lpips_alex_32",
        "l1_32",
        "sw_a_0.1",
    } == set(m)
    assert all(len(v) == 5 for v in m.values())
    identical = em.canvas_metrics(t[None], t)
    assert identical["l1"][0] == 0.0 and identical["ssim"][0] == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------------ prefix consistency (section 7.5)


def random_sequence(n: int, seed: int = 0) -> StrokeSequence:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, 9, generator=g)
    p[:, 2:4] = 0.1 + 0.2 * p[:, 2:4]  # sizes
    return StrokeSequence(p, canvas_size=(SIZE, SIZE), background=(1.0, 1.0, 1.0))


def test_the_canvas_at_k_is_the_render_of_the_first_k_strokes() -> None:
    seq = random_sequence(40)
    cfg = RenderConfig()
    cps, canvases = pr.prefix_canvases(seq, [0, 5, 10, 25, 100], cfg)
    assert cps == [0, 5, 10, 25]  # 100 > 40 strokes: not reachable, not imputed
    for k, canvas in zip(cps, canvases, strict=True):
        assert torch.allclose(canvas, seq[:k].render(cfg=cfg), atol=1e-6)
    assert torch.allclose(
        canvases[0], torch.ones(3, SIZE, SIZE)
    )  # n = 0 is the declared blank canvas
    full = pr.prefix_canvases(seq, [0, 40], cfg)[1][-1]
    assert torch.allclose(
        full, seq.render(cfg=cfg), atol=1e-6
    )  # adapter canvas at n = len(sequence) == sequence.render()


# ------------------------------------------------------------------------ contract and measurement


def test_the_contract_rejects_nan_wrong_grid_and_out_of_range_canvases() -> None:
    t = make_target()
    ok = FakeProducer(600).paint(t, PROTO.checkpoints, 0)
    pr.validate_result(ok, PROTO.checkpoints, SIZE)
    nan = FakeProducer(600, nan_at=3).paint(t, PROTO.checkpoints, 0)
    with pytest.raises(ValueError, match="NaN"):
        pr.validate_result(nan, PROTO.checkpoints, SIZE)
    bad_grid = FakeProducer(600).paint(t, PROTO.checkpoints, 0)
    bad_grid.checkpoints = bad_grid.checkpoints[:-1]
    with pytest.raises(ValueError, match="grid"):
        pr.validate_result(bad_grid, PROTO.checkpoints, SIZE)
    wide = FakeProducer(600).paint(t, PROTO.checkpoints, 0)
    wide.canvases = wide.canvases * 3
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        pr.validate_result(wide, PROTO.checkpoints, SIZE)


def test_the_measured_paint_loads_first_and_fills_the_measurements() -> None:
    prod = FakeProducer(600)
    res = pr.measured_paint(prod, make_target(), PROTO.checkpoints, 2)
    assert prod.loaded == [2] and res.runtime_s >= 0.0 and res.checkpoints[-1] == 500


# ------------------------------------------------------------------------ the runner


def test_controls_and_references_of_an_image() -> None:
    ctx = context()
    c = rn.image_controls(ctx, PROTO)
    assert set(c["controls"]) == {"blank_black", "blank_white", "blur"}
    assert c["references"]["q_ref"] == pytest.approx(
        max(c["controls"]["blank_black"]["lpips_alex"], c["controls"]["blank_white"]["lpips_alex"])
    )
    assert (
        c["controls"]["blur"]["lpips_alex"] < c["references"]["q_ref"]
    )  # a blurred copy is far better than blank
    assert "ref_sw_inspyrenet_0.1" in c["references"]
    blurred = rn.blur(ctx.target, 1.0)
    assert blurred.shape == ctx.target.shape and (blurred - ctx.target).abs().max() > 0


def test_a_record_holds_curves_aucs_and_cuts_unreachable_budgets() -> None:
    ctx = context()
    controls = rn.image_controls(ctx, PROTO)
    rec = rn.run_one(FakeProducer(600), ctx, 0, PROTO, controls, {})
    assert rec["status"] == "ok" and rec["checkpoints"][-1] == 500 and rec["n_total"] == 600
    assert set(rec["auc"]) == {"250", "500"}  # 1000 and 2000 are not reachable with 600 strokes
    assert all(a["lpips_alex"]["in_range"] for a in rec["auc"].values())
    assert len(rec["metrics"]["lpips_alex"]) == len(rec["checkpoints"])
    # the fake painting converges to the target: the LPIPS curve decreases to ~0 and AUC@500 < AUC@250
    assert rec["metrics"]["lpips_alex"][-1] < 1e-3
    assert rec["auc"]["500"]["lpips_alex"]["norm"] < rec["auc"]["250"]["lpips_alex"]["norm"]
    assert rec["control"] == {"note": "fake"} and rec["background"] == [0.5, 0.5, 0.5]


def test_failures_are_recorded_and_not_skipped() -> None:
    ctx = context()
    controls = rn.image_controls(ctx, PROTO)
    boom = rn.run_one(FakeProducer(fail=True), ctx, 0, PROTO, controls, {})
    assert boom["status"] == "failed" and "boom" in boom["error"]
    nan = rn.run_one(FakeProducer(600, nan_at=2), ctx, 0, PROTO, controls, {})
    assert nan["status"] == "failed" and "NaN" in nan["error"]


def test_the_runner_is_deterministic_resumable_and_sorted(tmp_path: Path) -> None:
    images = [context("b", 1), context("a", 0)]
    log: list[str] = []
    counts = rn.run_harness(
        [FakeProducer(600)],
        images,
        {"fake": [0, 1]},
        PROTO,
        tmp_path / "r1",
        log=log.append,
        timed=False,
    )
    assert counts == {"ran": 4, "skipped": 0, "failed": 0}
    assert [line.split()[1] for line in log] == ["a", "a", "b", "b"]  # sorted image ids
    again = rn.run_harness(
        [FakeProducer(600)],
        images,
        {"fake": [0, 1]},
        PROTO,
        tmp_path / "r2",
        log=log.append,
        timed=False,
    )
    assert again["ran"] == 4
    for image in ("a", "b"):
        for seed in (0, 1):
            h1 = json.loads(rn.record_path(tmp_path / "r1", "fake", image, seed).read_text())[
                "content_hash"
            ]
            h2 = json.loads(rn.record_path(tmp_path / "r2", "fake", image, seed).read_text())[
                "content_hash"
            ]
            assert h1 == h2  # same producer, same image, same seed: same record (timings excluded)
    resumed = rn.run_harness(
        [FakeProducer(600)],
        images,
        {"fake": [0, 1]},
        PROTO,
        tmp_path / "r1",
        log=log.append,
        timed=False,
    )
    assert resumed == {"ran": 0, "skipped": 4, "failed": 0}
    assert (tmp_path / "r1" / "environment.json").exists()


def test_a_contended_gpu_refuses_a_timed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rn, "other_gpu_memory_gb", lambda: 3.5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="idle GPU"):
        rn.run_harness([FakeProducer(600)], [context()], {"fake": [0]}, PROTO, tmp_path, timed=True)


def test_the_first_image_of_each_producer_and_seed_is_flagged_as_the_warm_up(
    tmp_path: Path,
) -> None:
    images = [context("c", 2), context("a", 0), context("b", 1)]
    rn.run_harness(
        [FakeProducer(600)],
        images,
        {"fake": [0, 1]},
        PROTO,
        tmp_path,
        log=lambda _: None,
        timed=False,
    )
    flags = {
        (img, seed): json.loads(rn.record_path(tmp_path, "fake", img, seed).read_text())["warmup"]
        for img in ("a", "b", "c")
        for seed in (0, 1)
    }
    assert flags == {
        ("a", 0): True,
        ("a", 1): True,
        ("b", 0): False,
        ("b", 1): False,
        ("c", 0): False,
        ("c", 1): False,
    }
    # the warm-up flag is not part of the content hash: it depends on the order of a run, not on the painting
    first = json.loads(rn.record_path(tmp_path, "fake", "a", 0).read_text())
    assert first["content_hash"] == rn.content_hash({**first, "warmup": False})
