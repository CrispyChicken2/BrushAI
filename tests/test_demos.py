"""Tests for the M1 demo pipeline (configs, fetch wrapper, painting + animation on a tiny case)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

from brushstroke.animation import AnimationConfig
from brushstroke.baselines.optim.config import OptimConfig, PaintOptimConfig
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.config import load_config
from scripts.fetch_data import fetch
from scripts.fetch_demo_inputs import demo_image_ids
from scripts.make_m1_demos import (
    DemoConfig,
    animate_demo,
    compare_image,
    paint_demo,
    results_markdown,
    sequence_sha256,
)

ROOT = Path(__file__).parents[1]
DEMOS = sorted((ROOT / "configs" / "demos").glob("m1_*.yaml"))
HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def test_three_demo_configs_inherit_the_baseline() -> None:
    assert [p.stem for p in DEMOS] == ["m1_landscape", "m1_photo", "m1_portrait"]
    manifest = yaml.safe_load(
        (ROOT / "configs/experiments/images.yaml").read_text(encoding="utf-8")
    )
    known = {i["id"] for i in manifest["images"]}
    baseline = load_config(PaintOptimConfig, ROOT / "configs" / "baseline_optim.yaml")
    ids = set()
    for path in DEMOS:
        cfg = load_config(DemoConfig, path)
        assert cfg.name == path.stem and cfg.image_id in known
        ids.add(cfg.image_id)
        # the painting settings are the baseline's, plus the reproducibility switch
        assert dataclasses.replace(cfg.paint, cudnn_deterministic=False) == baseline
        assert cfg.paint.cudnn_deterministic and cfg.paint.seed == 0
        a = cfg.animation
        assert (
            8.0 <= a.duration <= 12.0
            and a.hold == 2.0
            and "gif" in a.formats
            and "mp4" in a.formats
        )
        assert a.gif_width <= 512 and a.easing == "by_level"
    assert len(ids) == 3
    assert demo_image_ids() == [load_config(DemoConfig, p).image_id for p in DEMOS]
    for img in ids:  # every demo input is documented with its license in docs/data_sources.md
        assert img in (ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")


def test_fetch_ids_filter(tmp_path: Path) -> None:
    files = {}
    for name in ("a", "b"):
        f = tmp_path / f"{name}.jpg"
        f.write_bytes(name.encode() * 20)
        files[name] = f
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"images": [{"id": n, "url": f.as_uri(), "sha256": ""} for n, f in files.items()]}
        )
    )
    out = tmp_path / "raw"
    got = fetch(manifest, out, update=True, ids=["b"])
    assert list(got) == ["b"] and not (out / "a.jpg").exists()
    with pytest.raises(KeyError):
        fetch(manifest, out, ids=["zzz"])


def test_sequence_hash_and_results_markdown(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    seq = StrokeSequence(torch.rand(5, 9, generator=g), canvas_size=(32, 32))
    other = StrokeSequence(seq.params + 1e-3, canvas_size=(32, 32))
    assert sequence_sha256(seq) == sequence_sha256(
        StrokeSequence(seq.params.clone(), canvas_size=(32, 32))
    )
    assert sequence_sha256(seq) != sequence_sha256(other)
    base = {
        "l1": 0.03,
        "psnr": 25.0,
        "ssim": 0.7,
        "lpips_vgg": 0.30,
        "lpips_alex": 0.35,
        "num_strokes": 2037.0,
        "runtime_seconds": 560.0,
        "peak_vram_mb": 3400.0,
    }
    rows = [
        {"name": "m1_a", "image_id": "x", "seed": 0, "metrics": base, "gif_mb": 1.5},
        {
            "name": "m1_b",
            "image_id": "y",
            "seed": 0,
            "metrics": base | {"lpips_alex": 0.45},
            "gif_mb": 2.0,
        },
    ]
    text = results_markdown(rows)
    lines = [ln for ln in text.splitlines() if ln.startswith("| m1_")]
    assert lines[0].endswith("| yes |") and lines[1].endswith("| NO |")
    assert "0.300" in lines[0] and "0.450" in lines[1]
    assert "[m1_a](m1_a_compare.jpg)" in text


def test_compare_image_is_small(tmp_path: Path) -> None:
    for n in ("t", "f"):
        Image.new("RGB", (512, 512), (200, 120, 40) if n == "t" else (30, 60, 90)).save(
            tmp_path / f"{n}.png"
        )
    compare_image(tmp_path / "t.png", tmp_path / "f.png", tmp_path / "x" / "c.jpg")
    img = Image.open(tmp_path / "x" / "c.jpg")
    assert img.size == (2 * 384 + 8, 384) and (tmp_path / "x" / "c.jpg").stat().st_size < 500_000


def tiny_demo() -> DemoConfig:
    optim = dataclasses.replace(
        OptimConfig(),
        height=64,
        width=64,
        levels=[1, 2],
        strokes_per_patch=[2, 2],
        iters_per_level=[8, 8],
        finetune_all=False,
    )
    paint = dataclasses.replace(
        PaintOptimConfig(), optim=optim, device="cpu", cudnn_deterministic=True
    )
    paint.loss.lpips = 0.0
    anim = dataclasses.replace(
        AnimationConfig(),
        fps=6,
        duration=1.5,
        hold=0.5,
        formats=["mp4", "gif"],
        device="cpu",
        easing="by_level",
        gif_fps=3,
        gif_width=32,
    )
    return DemoConfig(name="m1_tiny", image_id="tiny", paint=paint, animation=anim)


def tiny_input(tmp_path: Path) -> Path:
    src = tmp_path / "in.png"
    g = torch.Generator().manual_seed(3)
    Image.fromarray((torch.rand(80, 100, 3, generator=g) * 255).byte().numpy()).save(src)
    return src


def test_tiny_painting_is_reproducible_and_exports(tmp_path: Path) -> None:
    from brushstroke.baselines.optim.optimize import paint_optim
    from brushstroke.utils.io import center_crop_resize, load_image
    from brushstroke.utils.seed import seed_everything

    cfg = tiny_demo()
    target = center_crop_resize(load_image(tiny_input(tmp_path)), (64, 64))
    hashes = []
    for _ in range(2):
        seed_everything(cfg.paint.seed, False, cfg.paint.cudnn_deterministic)
        p = cfg.paint
        seq = paint_optim(target, p.optim, p.loss, p.brush, p.render, p.seed, "cpu")
        hashes.append(sequence_sha256(seq))
        assert len(seq) == 1 * 2 + 4 * 2  # level 1: 1 patch x 2 strokes; level 2: 4 patches x 2
    assert hashes[0] == hashes[1]
    if HAVE_FFMPEG:
        files = animate_demo(cfg, seq, tmp_path / "anim")
        assert files["mp4"].stat().st_size > 0 and files["gif"].stat().st_size > 0
    loaded = StrokeSequence.load(seq.save(tmp_path / "s.npz"))
    assert sequence_sha256(loaded) == hashes[0]  # the hash survives save/load


@pytest.mark.slow
def test_paint_demo_writes_outputs_and_metrics(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    seq, target, metrics = paint_demo(tiny_demo(), tiny_input(tmp_path), out)
    assert target.shape == (3, 64, 64) and len(seq) == 10
    assert {
        "l1",
        "psnr",
        "ssim",
        "lpips_vgg",
        "lpips_alex",
        "num_strokes",
        "runtime_seconds",
    } <= set(metrics)
    for name in ("target.png", "final.png", "sequence.npz"):
        assert (out / name).exists()
