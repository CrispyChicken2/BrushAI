"""Tests for the animation: timeline, partial reveal, incremental frames and ffmpeg export."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from brushstroke.animation import (
    AnimationConfig,
    FrameRenderer,
    build_timeline,
    export_frames,
    find_ffmpeg,
    num_frames,
)
from brushstroke.animation import export as export_module
from brushstroke.renderer import BrushBank, apply_canvas, composite_sequential, make_canvas_texture
from brushstroke.renderer.params import canonicalize
from brushstroke.renderer.rasterize import rasterize_flat, reveal_front
from brushstroke.strokes import BrushSpec, StrokeSequence
from brushstroke.utils.config import load_config

SPEC = BrushSpec(kind="oil", bank_size=4, mask_height=32, mask_width=64, seed=1)
HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed")


def make_seq(n: int = 30, size: tuple[int, int] = (64, 64), seed: int = 0) -> StrokeSequence:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, 9, generator=g)
    p[:, 2:4] = 0.1 + 0.35 * p[:, 2:4]
    p[:, 8] = 0.6 + 0.4 * p[:, 8]
    return StrokeSequence(p, canvas_size=size, brush=SPEC, background=(0.9, 0.85, 0.8))


def cfg_for(**kw: object) -> AnimationConfig:
    base: dict[str, object] = {"fps": 10, "duration": 3.0, "hold": 0.5, "device": "cpu"}
    return dataclasses.replace(AnimationConfig(), **{**base, **kw})  # type: ignore[arg-type]


# ------------------------------------------------------------------ timeline


@pytest.mark.parametrize(
    ("fps", "duration", "hold"), [(30, 10.0, 1.5), (24, 2.5, 0.7), (10, 0.35, 0.0)]
)
def test_frame_count(fps: int, duration: float, hold: float) -> None:
    cfg = dataclasses.replace(
        AnimationConfig(), fps=fps, duration=duration, hold=hold, device="cpu"
    )
    expected = round(fps * duration) + round(fps * hold)
    assert num_frames(cfg) == expected
    assert build_timeline(20, cfg).num_frames == expected
    seq = make_seq(8)
    assert sum(1 for _ in FrameRenderer(seq, cfg).frames()) == expected


def test_timeline_structure_and_determinism() -> None:
    cfg = cfg_for(fps=30, duration=4.0)
    tl = build_timeline(100, cfg)
    again = build_timeline(100, cfg)
    for a, b in zip(
        (tl.start, tl.duration, tl.reverse),
        (again.start, again.duration, again.reverse),
        strict=True,
    ):
        assert np.array_equal(a, b)
    assert (np.diff(tl.start) >= 0).all() and tl.start[0] == 0.0
    assert (tl.duration > 0).all() and (tl.start + tl.duration <= tl.paint_seconds + 1e-9).all()
    assert abs(float((tl.start + tl.duration).max()) - tl.paint_seconds) < 1e-9
    assert tl.reverse.any() and not tl.reverse.all()  # direction "random"
    assert not build_timeline(10, cfg_for(direction="forward")).reverse.any()
    assert build_timeline(10, cfg_for(direction="reverse")).reverse.all()
    assert not np.array_equal(
        tl.reverse, build_timeline(100, cfg_for(fps=30, duration=4.0, seed=1)).reverse
    )
    # prefix-stable directions: stroke k's direction does not depend on N
    assert np.array_equal(build_timeline(40, cfg).reverse, tl.reverse[:40])


def test_timeline_edge_cases_and_validation() -> None:
    cfg = cfg_for()
    assert build_timeline(0, cfg).start.size == 0
    one = build_timeline(1, cfg)
    assert one.start[0] == 0.0 and one.duration[0] <= one.paint_seconds
    for bad in (
        {"easing": "bounce"},
        {"direction": "up"},
        {"stroke_ease": "cubic"},
        {"fps": 0},
        {"duration": 0.0},
        {"overlap": 0.0},
        {"duration": 0.01, "fps": 10},
    ):
        with pytest.raises(ValueError):
            build_timeline(5, cfg_for(**bad))


def test_overlap_and_easing() -> None:
    seq_n = 60
    sequential = build_timeline(seq_n, cfg_for(overlap=1.0, min_reveal_frames=0.0, duration=6.0))
    assert (
        sequential.start[1:] >= (sequential.start + sequential.duration)[:-1] - 1e-9
    ).all()  # no overlap
    heavy = build_timeline(seq_n, cfg_for(overlap=5.0, min_reveal_frames=0.0, duration=6.0))
    in_flight = (heavy.start[None, :] <= heavy.start[:, None]) & (
        heavy.start[:, None] < (heavy.start + heavy.duration)[None, :]
    )
    assert in_flight.sum(axis=1).mean() > 3.0
    decel = build_timeline(
        seq_n,
        cfg_for(
            easing="decelerate", decel_ratio=3.0, min_reveal_frames=0.0, overlap=1.0, duration=6.0
        ),
    )
    assert decel.duration[-1] > 2.5 * decel.duration[0]  # later strokes are slower
    assert (np.diff(decel.start) >= 0).all()


def test_progress_is_monotone_and_complete() -> None:
    cfg = cfg_for(fps=20, duration=2.0, hold=0.5)
    tl = build_timeline(25, cfg)
    prog = np.stack([tl.progress(f) for f in range(tl.num_frames)])
    assert (np.diff(prog, axis=0) >= -1e-12).all()
    assert (prog >= 0).all() and (prog <= 1).all()
    assert (prog[tl.paint_frames - 1] == 1.0).all()  # painting phase ends with everything done
    assert (prog[tl.paint_frames :] == 1.0).all()
    lin = build_timeline(25, cfg_for(fps=20, duration=2.0, stroke_ease="linear"))
    assert (lin.progress(5) >= 0).all()


def test_no_stroke_pops_in_when_reveal_lasts_enough_frames() -> None:
    cfg = cfg_for(fps=30, duration=2.0, hold=0.0, min_reveal_frames=3.0, overlap=1.0)
    tl = build_timeline(80, cfg)  # 80 strokes in 60 frames: would pop without the minimum
    prog = np.stack([tl.progress(f) for f in range(tl.num_frames)])
    partial = ((prog > 0) & (prog < 1)).sum(axis=0)
    assert (partial >= 1).all(), "every stroke must be visible in at least one partial frame"
    # the documented exception: a reveal shorter than one frame may pop in
    fast = cfg_for(fps=30, duration=2.0, hold=0.0, min_reveal_frames=0.0, overlap=0.2)
    tl2 = build_timeline(300, fast)
    assert (tl2.duration * fast.fps < 1.0).all()
    prog2 = np.stack([tl2.progress(f) for f in range(tl2.num_frames)])
    assert (((prog2 > 0) & (prog2 < 1)).sum(axis=0) == 0).any()


# ------------------------------------------------------------------ partial reveal


def test_reveal_front_is_monotone_zero_at_start_and_exact_at_end() -> None:
    gx = torch.linspace(-1.3, 1.3, 53)[None, None, :].expand(1, 4, 53).contiguous()
    prev = torch.zeros_like(gx)
    for p in torch.linspace(0, 1, 21):
        cur = reveal_front(gx, p.view(1), 0.15)
        assert (cur >= prev - 1e-7).all()
        prev = cur
    assert (reveal_front(gx, torch.zeros(1), 0.15) == 0).all()
    assert (reveal_front(gx, torch.ones(1), 0.15) == 1).all()  # also beyond the box edges
    rev = reveal_front(gx, torch.tensor([0.5]), 0.15, torch.tensor([True]))
    fwd = reveal_front(gx, torch.tensor([0.5]), 0.15, torch.tensor([False]))
    assert torch.allclose(rev, torch.flip(fwd, dims=[-1]), atol=1e-6)  # mirrored


def test_stroke_coverage_monotone_in_progress_with_renderer_geometry() -> None:
    seq = make_seq(12, seed=3)
    assert seq.brush_index is not None
    bank = BrushBank.build(
        SPEC.kind, SPEC.bank_size, (SPEC.mask_height, SPEC.mask_width), SPEC.seed
    )
    params = canonicalize(seq.params[None], 64, 64)[0]
    masks = bank.masks[seq.brush_index]
    reverse = torch.arange(12) % 2 == 0
    prev = torch.zeros(12, 1, 64, 64)
    for p in torch.linspace(0, 1, 11):
        a = rasterize_flat(params, masks, 64, 64, reveal=p.expand(12), reveal_reverse=reverse)
        assert (a >= prev - 1e-6).all()
        prev = a
    full = rasterize_flat(params, masks, 64, 64)
    assert torch.equal(prev, full)  # at p = 1 the stroke equals the normally rendered one
    first = rasterize_flat(params, masks, 64, 64, reveal=torch.zeros(12))
    assert (first == 0).all()
    ones = rasterize_flat(params, masks, 64, 64, reveal=torch.ones(12))
    assert torch.equal(ones, full)


def test_reveal_front_advances_along_the_main_axis() -> None:
    ones = torch.ones(1, 1, 256, 256)
    params = torch.tensor([[0.5, 0.5, 0.6, 0.2, 0.0, 1, 1, 1, 1.0]])
    xs: list[float] = []
    for p in (0.25, 0.5, 0.75):
        a = rasterize_flat(params, ones, 128, 128, reveal=torch.tensor([p]))[0, 0]
        cols = torch.nonzero(a.max(dim=0).values > 0.5).flatten()
        xs.append(float(cols.max()))
    assert xs[0] < xs[1] < xs[2]
    # forward: the left end is revealed first; half-way the front is at the stroke middle (x = 64)
    assert abs(xs[1] - 64) < 3


# ------------------------------------------------------------------ frames


def reference_frame(
    seq: StrokeSequence,
    cfg: AnimationConfig,
    renderer: FrameRenderer,
    frame: int,
    tex: torch.Tensor | None,
) -> torch.Tensor:
    """Brute force: every started stroke at its progress, composited from scratch in index order."""
    tl = renderer.timeline
    prog = tl.progress(frame)
    assert seq.brush_index is not None
    bank = BrushBank.build(
        SPEC.kind, SPEC.bank_size, (SPEC.mask_height, SPEC.mask_width), SPEC.seed
    )
    h, w = seq.canvas_size
    params = canonicalize(seq.params[None], h, w)[0]
    alphas = rasterize_flat(
        params,
        bank.masks[seq.brush_index],
        h,
        w,
        reveal=torch.from_numpy(prog).float(),
        reveal_edge=cfg.reveal_edge,
        reveal_reverse=torch.from_numpy(tl.reverse),
    )
    bg = torch.tensor(seq.background).view(3, 1, 1).expand(3, h, w)
    img = composite_sequential(bg[None], alphas[None], seq.params[None, :, 5:8])[0]
    return apply_canvas(img, tex, cfg.canvas_strength) if tex is not None else img


def test_incremental_frames_equal_brute_force_composite() -> None:
    seq = make_seq(40, seed=4)
    cfg = cfg_for(fps=10, duration=3.0, hold=0.4, overlap=6.0, chunk_size=7)
    renderer = FrameRenderer(seq, cfg)
    tex = make_canvas_texture(
        64,
        64,
        cfg.canvas_seed,
        weave_period=cfg.weave_period,
        low_freq_amount=cfg.low_freq_amount,
        low_freq_scale=cfg.low_freq_scale,
    )
    for f in range(renderer.timeline.num_frames):
        got = renderer.render_frame(f)
        want = reference_frame(seq, cfg, renderer, f, tex)
        assert torch.allclose(got, want, atol=1e-5), f
    # requesting the frames again from a fresh state gives the same uint8 frames
    a = list(renderer.frames())
    b = list(renderer.frames())
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_first_and_last_frames() -> None:
    seq = make_seq(25, seed=2)
    cfg = cfg_for(chunk_size=4)
    frames = list(FrameRenderer(seq, cfg).frames())
    tex = make_canvas_texture(
        64,
        64,
        cfg.canvas_seed,
        weave_period=cfg.weave_period,
        low_freq_amount=cfg.low_freq_amount,
        low_freq_scale=cfg.low_freq_scale,
    )
    final = apply_canvas(seq.render(), tex, cfg.canvas_strength)
    last = torch.from_numpy(frames[-1]).permute(2, 0, 1).float() / 255.0
    assert (last - final).abs().max() <= 1.0 / 255.0 + 1e-6  # last frame == full render (+ canvas)
    first = torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 255.0
    blank = apply_canvas(
        torch.tensor(seq.background).view(3, 1, 1).expand(3, 64, 64), tex, cfg.canvas_strength
    )
    assert (first - blank).abs().max() <= 1.0 / 255.0 + 1e-6
    assert all(f.shape == (64, 64, 3) and f.dtype == np.uint8 for f in frames)
    # no hold: the painting phase itself ends complete
    short = list(FrameRenderer(seq, cfg_for(hold=0.0)).frames())
    assert np.array_equal(short[-1], frames[-1])


def test_canvas_disabled_and_output_size_override() -> None:
    seq = make_seq(15, seed=6)
    cfg = cfg_for(canvas_enabled=False, height=96, width=96)
    last = list(FrameRenderer(seq, cfg).frames())[-1]
    assert last.shape == (96, 96, 3)
    want = seq.render(96, 96)
    got = torch.from_numpy(last).permute(2, 0, 1).float() / 255.0
    assert (got - want).abs().max() <= 1.0 / 255.0 + 1e-6


def test_empty_and_single_stroke_sequences() -> None:
    cfg = cfg_for(fps=5, duration=1.0, hold=0.2)
    frames = list(FrameRenderer(StrokeSequence.empty((32, 32), brush=SPEC), cfg).frames())
    assert len(frames) == 6 and all(np.array_equal(frames[0], f) for f in frames)
    one = list(FrameRenderer(make_seq(1, (32, 32)), cfg).frames())
    assert not np.array_equal(one[0], one[-1])


@pytest.mark.gpu
def test_frames_on_gpu_match_cpu() -> None:
    seq = make_seq(30)
    cpu = list(FrameRenderer(seq, cfg_for()).frames())
    gpu = list(FrameRenderer(seq, cfg_for(device="cuda")).frames())
    diff = max(
        int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(cpu, gpu, strict=True)
    )
    assert diff <= 2


# ------------------------------------------------------------------ export


def probe(path: Path) -> dict[str, object]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_packets",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,r_frame_rate,width,height,pix_fmt,nb_read_packets:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)  # type: ignore[no-any-return]


@needs_ffmpeg
def test_export_mp4_webm_gif_with_ffprobe(tmp_path: Path) -> None:
    seq = make_seq(20, (64, 64))
    cfg = cfg_for(
        fps=10, duration=2.0, hold=0.5, formats=["mp4", "webm", "gif"], gif_width=48, gif_fps=5
    )
    files = export_frames(FrameRenderer(seq, cfg).frames(), tmp_path / "v", (64, 64), cfg)
    assert set(files) == {"mp4", "webm", "gif"} and all(
        p.stat().st_size > 0 for p in files.values()
    )
    frames = num_frames(cfg)  # 25 frames -> 2.5 s
    mp4 = probe(files["mp4"])
    s = mp4["streams"][0]  # type: ignore[index]
    assert s["codec_name"] == "h264" and s["pix_fmt"] == "yuv420p" and s["r_frame_rate"] == "10/1"
    assert (
        int(s["nb_read_packets"]) == frames
        and abs(float(mp4["format"]["duration"]) - frames / 10) < 0.15
    )  # type: ignore[index]
    webm = probe(files["webm"])
    s = webm["streams"][0]  # type: ignore[index]
    assert s["codec_name"] == "vp9" and int(s["nb_read_packets"]) == frames
    assert abs(float(webm["format"]["duration"]) - frames / 10) < 0.15  # type: ignore[index]
    gif = probe(files["gif"])
    s = gif["streams"][0]  # type: ignore[index]
    assert s["codec_name"] == "gif" and s["width"] == 48 and s["r_frame_rate"] == "5/1"
    assert abs(float(gif["format"]["duration"]) - frames / 10) < 0.35  # type: ignore[index]


@needs_ffmpeg
def test_export_odd_size_and_crf_options(tmp_path: Path) -> None:
    seq = make_seq(10, (63, 65))
    cfg = cfg_for(fps=10, duration=1.0, hold=0.0, formats=["mp4"], mp4_crf=30, gif_dither="none")
    files = export_frames(FrameRenderer(seq, cfg).frames(), tmp_path / "odd", (65, 63), cfg)
    s = probe(files["mp4"])["streams"][0]  # type: ignore[index]
    assert (s["width"], s["height"]) == (64, 62)  # made even for yuv420p
    gif_cfg = dataclasses.replace(cfg, formats=["gif"], gif_dither="floyd_steinberg", gif_width=0)
    files = export_frames(
        FrameRenderer(seq, gif_cfg).frames(), tmp_path / "odd2", (65, 63), gif_cfg
    )
    assert probe(files["gif"])["streams"][0]["width"] == 65  # type: ignore[index]


@needs_ffmpeg
def test_export_validation_errors(tmp_path: Path) -> None:
    cfg = cfg_for(formats=["mp4"])
    bad = (np.zeros((8, 8, 3), np.uint8) for _ in range(2))
    with pytest.raises(ValueError, match="uint8"):
        export_frames(bad, tmp_path / "x", (16, 16), cfg)
    with pytest.raises(ValueError):
        export_frames([], tmp_path / "y", (16, 16), cfg_for(formats=["avi"]))
    with pytest.raises(ValueError):
        export_frames([], tmp_path / "z", (16, 16), cfg_for(formats=["gif"], gif_dither="rainbow"))


def test_missing_ffmpeg_gives_clear_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_module.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        find_ffmpeg()
    with pytest.raises(RuntimeError, match="not found"):
        export_frames([], tmp_path / "x", (16, 16), cfg_for())


def test_animation_yaml_matches_defaults() -> None:
    path = Path(__file__).parents[1] / "configs" / "animation.yaml"
    assert load_config(AnimationConfig, path) == AnimationConfig()


def test_by_level_easing_gives_coarse_levels_more_time_per_stroke() -> None:
    levels = np.concatenate([np.full(1, 0), np.full(4, 1), np.full(100, 2)])
    cfg = cfg_for(
        fps=30,
        duration=6.0,
        easing="by_level",
        level_time_power=0.5,
        min_reveal_frames=0.0,
        overlap=1.0,
    )
    tl = build_timeline(len(levels), cfg, levels)
    assert (np.diff(tl.start) >= 0).all() and (
        tl.start + tl.duration <= tl.paint_seconds + 1e-9
    ).all()
    assert tl.duration[0] > 5 * tl.duration[-1]  # the single coarse stroke is revealed more slowly
    # time per level ~ count ** power: the 100-stroke level gets ~10 / (1 + 2 + 10) of the time
    per_level = [float(tl.duration[levels == lv].sum()) for lv in (0, 1, 2)]
    assert per_level[2] / sum(per_level) == pytest.approx(10 / 13, rel=0.05)
    uniform = build_timeline(len(levels), dataclasses.replace(cfg, easing="uniform"))
    assert tl.duration[0] > uniform.duration[0]
    with pytest.raises(ValueError):
        build_timeline(len(levels), cfg)  # levels are required
    with pytest.raises(ValueError):
        build_timeline(len(levels), cfg, levels[:-1])
    # the frame renderer reads the tags of the sequence
    seq = make_seq(12)
    seq = dataclasses.replace(
        seq, scale_level=torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], dtype=torch.int32)
    )
    renderer = FrameRenderer(seq, cfg_for(easing="by_level"))
    assert renderer.timeline.duration[0] > renderer.timeline.duration[-1]
