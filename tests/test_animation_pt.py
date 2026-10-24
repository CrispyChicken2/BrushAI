"""Tests of the Paint Transformer frames of the animation (T-018)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from brushstroke.animation import (
    AnimationConfig,
    FrameRenderer,
    PTFrameRenderer,
    is_pt_sequence,
    make_frame_renderer,
)
from brushstroke.renderer.pipeline import RenderConfig
from brushstroke.strokes import StrokeSequence
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
SIZE = 96


def pt_sequence(n: int = 12, source: str = "paint_transformer") -> StrokeSequence:
    g = torch.Generator().manual_seed(0)
    xy = 0.2 + 0.6 * torch.rand(n, 2, generator=g)
    wh = 0.1 + 0.25 * torch.rand(n, 2, generator=g)
    theta = torch.rand(n, 1, generator=g)
    colour = torch.rand(n, 3, generator=g)
    params = torch.cat([xy, wh, theta, colour, torch.ones(n, 1)], dim=1)
    return StrokeSequence(
        params=params,
        canvas_size=(SIZE, SIZE),
        background=(0.0, 0.0, 0.0),
        scale_level=torch.zeros(n, dtype=torch.int32),
        patch_id=torch.arange(n, dtype=torch.int32),
        source=[source] * n,
    )


def pt_cfg(**kw: object) -> AnimationConfig:
    base = {
        "renderer": "pt",
        "fps": 10,
        "duration": 2.0,
        "hold": 0.5,
        "canvas_enabled": False,
        "pt_brush_dir": str(po.BRUSHES),
        "min_reveal_frames": 0.0,
    }
    return AnimationConfig(**(base | kw))  # type: ignore[arg-type]


def test_renderer_choice() -> None:
    seq = pt_sequence()
    assert is_pt_sequence(seq) and not is_pt_sequence(
        StrokeSequence(params=torch.zeros(1, 9), source=["optim"])
    )
    with pytest.raises(ValueError):
        make_frame_renderer(seq, AnimationConfig(renderer="bogus"))


@needs_brushes
def test_auto_dispatch_and_last_frame_is_the_generic_pt_render() -> None:
    seq = pt_sequence()
    cfg = pt_cfg(renderer="auto")
    renderer = make_frame_renderer(seq, cfg)
    assert isinstance(renderer, PTFrameRenderer)
    frames = list(renderer.frames())
    assert len(frames) == renderer.timeline.num_frames
    expected = seq.render(cfg=RenderConfig(mode="pt", pt_brush_dir=str(po.BRUSHES)))
    got = torch.from_numpy(frames[-1]).permute(2, 0, 1).float() / 255
    assert (got - expected).abs().max() < 2 / 255
    assert not isinstance(make_frame_renderer(seq, pt_cfg(renderer="procedural")), PTFrameRenderer)


@needs_brushes
def test_reveal_is_monotone_and_the_canvas_starts_black() -> None:
    renderer = PTFrameRenderer(pt_sequence(), pt_cfg())
    renderer.reset()
    frames = [renderer.render_frame(f) for f in range(renderer.timeline.paint_frames)]
    assert float(frames[0].max()) < 0.2  # (almost) nothing painted yet
    coverage = [float(f.sum()) for f in frames]
    assert all(b >= a - 1e-3 for a, b in itertools_pairs(coverage))  # paint is only added
    assert coverage[-1] > coverage[len(coverage) // 4] > coverage[0]


def itertools_pairs(values: list[float]) -> list[tuple[float, float]]:
    return list(zip(values[:-1], values[1:], strict=True))


@needs_brushes
def test_partial_reveal_is_a_part_of_the_full_stroke() -> None:
    """Halfway through a lone stroke the revealed alpha is inside the full stroke's alpha."""
    seq = pt_sequence(1)
    renderer = PTFrameRenderer(seq, pt_cfg(duration=2.0, overlap=1.0))
    renderer.reset()
    mid = renderer.render_frame(renderer.timeline.paint_frames // 2).sum(0)
    renderer.reset()
    full = renderer.render_frame(renderer.timeline.paint_frames - 1).sum(0)
    assert float((mid > 1e-6).sum()) < float((full > 1e-6).sum())
    assert float(((mid > 1e-6) & ~(full > 1e-6)).sum()) == 0
    assert np.isfinite(float(full.sum()))
    assert isinstance(renderer, FrameRenderer)


def test_m2_results_table_is_stable_across_reruns() -> None:
    """Runtimes are coarse in the committed table (a rerun must not dirty it)."""
    from scripts import make_m2_demos as m2

    assert m2.runtime_label(0.2) == m2.runtime_label(0.8) == "< 1 s"
    assert m2.runtime_label(565.6) == m2.runtime_label(566.4) == "9.4 min"
    metrics = {
        "num_strokes": 10.0,
        "l1": 0.1,
        "psnr": 20.0,
        "ssim": 0.5,
        "lpips_vgg": 0.3,
        "lpips_alex": 0.3,
        "runtime_seconds": 0.4,
        "peak_vram_mb": 1024.0,
    }
    row = {
        "name": "x",
        "image_id": "y",
        "pt": metrics,
        "m1": metrics | {"runtime_seconds": 566.0},
        "sequence_sha256": "ab" * 32,
        "gif_mb": 0.5,
        "sbs_gif_mb": 1.0,
    }
    text = m2.results_markdown([row])
    assert "Paint Transformer" in text and "9.4 min" in text and "`abababababab`" in text
    assert len(m2.sequence_sha256(pt_sequence())) == 64
