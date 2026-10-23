"""Tests for the T-009 tooling: fetch script, sweep jobs/aggregation, new baseline flags."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from brushstroke.baselines.optim.config import LossConfig, OptimConfig
from brushstroke.baselines.optim.init import sample_strokes
from brushstroke.baselines.optim.losses import get_lpips, lpips_distance
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.renderer import BrushBank, BrushConfig, render
from brushstroke.utils.config import load_config
from scripts.exp_optim_sweep import (
    SweepConfig,
    aggregate,
    build_jobs,
    mean_std,
    paired_delta,
    render_tables,
)
from scripts.fetch_data import fetch, sha256_of

BRUSH = BrushConfig(kind="oil", bank_size=4, mask_height=32, mask_width=64, seed=0)
ROOT = Path(__file__).parents[1]


def target_image(size: int = 48) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    p = torch.rand(1, 6, 9, generator=g)
    p[..., 2:4] = 0.2 + 0.4 * p[..., 2:4]
    p[..., 8] = 1.0
    bank = BrushBank.build("oil", 4, (32, 64), 0)
    return render(p, bank, size, size)[0]


def cfg_small(**kw: object) -> OptimConfig:
    base = {
        "height": 48,
        "width": 48,
        "levels": [1, 2],
        "strokes_per_patch": [3, 2],
        "iters": 25,
        "iters_per_level": [],
        "finetune_all": False,
    }
    return dataclasses.replace(OptimConfig(), **{**base, **kw})  # type: ignore[arg-type]


L1 = LossConfig(l1=1.0, lpips=0.0, ot=0.0)


# ------------------------------------------------------------------ baseline flags added for T-009


def test_init_random_and_density_modes() -> None:
    target = torch.rand(3, 64, 64, generator=torch.Generator().manual_seed(1))
    canvas = torch.zeros_like(target)
    ids = torch.arange(4).repeat_interleave(4)  # 2x2 grid, 4 strokes per patch
    base = OptimConfig(height=64, width=64)
    patch = sample_strokes(
        target, canvas, 2, ids, base, torch.Generator().manual_seed(0), per_patch=4
    )
    dens = sample_strokes(
        target,
        canvas,
        2,
        ids,
        dataclasses.replace(base, init_size_mode="density"),
        torch.Generator().manual_seed(0),
        per_patch=4,
    )
    assert torch.allclose(dens[:, 2] * 2.0, patch[:, 2])  # density: length / sqrt(4)
    rnd = sample_strokes(
        target,
        canvas,
        2,
        ids,
        dataclasses.replace(base, init_mode="random"),
        torch.Generator().manual_seed(0),
        per_patch=4,
    )
    x, y = rnd[:, 0] * 64, rnd[:, 1] * 64
    patch_of = (y // 32).long() * 2 + (x // 32).long()
    assert torch.equal(patch_of, ids)  # positions still inside their patch
    colour_from_target = torch.stack([target[:, int(b), int(a)] for a, b in zip(x, y, strict=True)])
    assert not torch.allclose(rnd[:, 5:8], colour_from_target, atol=1e-2)  # colours are random
    with pytest.raises(ValueError):
        sample_strokes(
            target,
            canvas,
            2,
            ids,
            dataclasses.replace(base, init_mode="x"),
            torch.Generator().manual_seed(0),
        )


@pytest.mark.parametrize("kw", [{"init_mode": "random"}, {"init_size_mode": "density"}])
def test_new_init_flags_run_in_the_painter(kw: dict[str, object]) -> None:
    target = target_image()
    seq = paint_optim(target, cfg_small(**kw), L1, BRUSH, seed=0)
    for curve in seq.meta["loss_curve"]:  # the optimization works whatever the initialization
        assert sum(curve[-5:]) < sum(curve[:5])
    if kw.get("init_mode") != "random":  # random colours need far more than 25 steps to beat blank
        initial = float((torch.ones_like(target) - target).abs().mean())
        assert float((seq.render() - target).abs().mean()) < 0.9 * initial


def test_final_pass_jointly_refines_all_strokes() -> None:
    target = target_image()
    base = paint_optim(target, cfg_small(), L1, BRUSH, seed=0)
    fin = paint_optim(target, cfg_small(final_pass_iters=30), L1, BRUSH, seed=0)
    assert len(fin) == len(base) and fin.meta["final_pass_seconds"] > 0
    assert len(fin.meta["final_pass_curve"]) == 30 and base.meta["final_pass_curve"] == []
    assert fin.meta["final_pass_curve"][-1] < fin.meta["final_pass_curve"][0]
    assert not torch.equal(fin.params, base.params)
    # the stored final loss matches what the (re-rendered) sequence achieves
    err = float((fin.render() - target).abs().mean())
    assert err == pytest.approx(fin.meta["final_loss"]["l1_metric"], abs=1e-4)
    assert fin.meta["final_loss"]["l1_metric"] <= base.meta["final_loss"]["l1_metric"] + 1e-3


@pytest.mark.slow
def test_lpips_alex_is_a_distinct_held_out_metric() -> None:
    g = torch.Generator().manual_seed(3)
    a, b = torch.rand(1, 3, 64, 64, generator=g), torch.rand(1, 3, 64, 64, generator=g)
    assert float(lpips_distance(a, a, "alex")) == pytest.approx(0.0, abs=1e-6)
    vgg, alex = float(lpips_distance(a, b, "vgg")), float(lpips_distance(a, b, "alex"))
    assert vgg > 0.05 and alex > 0.05 and vgg != alex
    assert get_lpips("cpu", "alex") is get_lpips("cpu", "alex") and get_lpips(
        "cpu", "vgg"
    ) is not get_lpips("cpu", "alex")
    with pytest.raises(ValueError):
        get_lpips("cpu", "squeeze")


# ------------------------------------------------------------------ fetch script


def test_fetch_downloads_verifies_and_records_checksums(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    src.write_bytes(b"not really a jpeg" * 10)
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    manifest = tmp_path / "images.yaml"
    manifest.write_text(
        yaml.safe_dump({"images": [{"id": "a", "url": src.as_uri(), "sha256": ""}]})
    )
    out = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="--update"):
        fetch(manifest, out)
    paths = fetch(manifest, out, update=True)
    assert paths["a"].read_bytes() == src.read_bytes() and sha256_of(paths["a"]) == digest
    assert yaml.safe_load(manifest.read_text())["images"][0]["sha256"] == digest
    fetch(manifest, out)  # verified, no download needed
    paths["a"].write_bytes(b"corrupted")  # a corrupted local file is re-downloaded
    fetch(manifest, out)
    assert sha256_of(paths["a"]) == digest
    src.write_bytes(b"changed upstream")  # but a changed upstream file fails the checksum
    paths["a"].unlink()
    with pytest.raises(RuntimeError, match="mismatch"):
        fetch(manifest, out)


def test_repo_image_manifest_is_complete() -> None:
    spec = yaml.safe_load(
        (ROOT / "configs" / "experiments" / "images.yaml").read_text(encoding="utf-8")
    )
    assert len(spec["images"]) == 5
    for item in spec["images"]:
        assert len(item["sha256"]) == 64 and item["license"].lower().startswith("public domain")
        assert item["url"].startswith("https://upload.wikimedia.org/")
        assert item["id"] in (ROOT / "docs" / "data_sources.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ sweep tooling


def sweep_cfg() -> SweepConfig:
    return load_config(SweepConfig, ROOT / "configs" / "experiments" / "optim_sweep.yaml")


def test_sweep_config_jobs_are_well_formed() -> None:
    cfg = sweep_cfg()
    jobs = build_jobs(cfg)
    assert len({(j.stage, j.variant, j.image, j.seed) for j in jobs}) == len(jobs)  # unique ids
    images = {
        i["id"]
        for i in yaml.safe_load(Path(cfg.images_manifest).read_text(encoding="utf-8"))["images"]
    }
    assert {j.image for j in jobs} <= images
    for stage in cfg.stages:
        assert stage.schedule in cfg.schedules
        for v in stage.variants:
            assert (v.schedule or stage.schedule) in cfg.schedules
    only = build_jobs(cfg, ["h5_brush"])
    assert {j.stage for j in only} == {"h5_brush"} and 0 < len(only) < len(jobs)
    # every job's overrides are accepted by the typed baseline config
    from brushstroke.baselines.optim.config import PaintOptimConfig

    for job in jobs:
        run = load_config(PaintOptimConfig, ROOT / cfg.base_config, list(job.overrides))
        assert len(run.optim.levels) == len(run.optim.strokes_per_patch)
    # budgets of the named schedules
    totals = {
        n: sum(g * g * s for g, s in zip(sc.levels, sc.strokes_per_patch, strict=True))
        for n, sc in cfg.schedules.items()
    }
    assert totals["s1000"] == 994 and totals["s2000"] == 1938
    assert all(
        850 <= totals[n] <= 1100
        for n in ("fine_heavy", "coarse_heavy", "uniform_per_level", "two_levels")
    )


def fake_record(stage: str, variant: str, image: str, seed: int, lpips: float) -> dict[str, object]:
    metrics = {
        "l1": 0.03,
        "psnr": 26.0,
        "lpips_vgg": lpips,
        "lpips_alex": lpips + 0.1,
        "runtime_seconds": 100.0,
        "peak_vram_mb": 2048.0,
        "num_strokes": 1000.0,
    }
    return {"stage": stage, "variant": variant, "image": image, "seed": seed, "metrics": metrics}


def test_mean_std_and_aggregate_and_tables() -> None:
    assert mean_std([2.0]) == (2.0, 0.0)
    m, s = mean_std([1.0, 3.0])
    assert m == 2.0 and s == pytest.approx(2**0.5)
    records = [
        fake_record("base", "default", "a", 0, 0.40),
        fake_record("base", "default", "b", 0, 0.50),
        fake_record("h1_iters", "iters200", "a", 0, 0.30),
    ]
    agg = aggregate(records)  # type: ignore[arg-type]
    row = agg[("base", "default")]
    assert row["n"] == 2 and row["lpips_vgg"][0] == pytest.approx(0.45)
    assert row["lpips_alex"][0] == pytest.approx(0.55) and row["images"] == ["a", "b"]
    cfg = sweep_cfg()
    text = render_tables(
        cfg,
        records,
        [
            {
                "image": "a",
                "reference": "blur",
                "l1": 0.01,
                "psnr": 30.0,  # type: ignore[arg-type]
                "lpips_vgg": 0.2,
                "lpips_alex": 0.1,
            }
        ],
    )
    assert "## base" in text and "0.450 ± 0.071" in text and "LPIPS-Alex" in text
    assert "## references" in text and "| blur | 1 |" in text
    assert "h2_alloc" not in text  # stages without records are omitted
    # paired deltas: iters200/a (0.30) vs default/a (0.40) -> -0.100 in LPIPS-VGG and Alex
    assert "-0.100" in text
    d = paired_delta(records, "h1_iters", "iters200", "base/default", "lpips_vgg")  # type: ignore[arg-type]
    assert d is not None and d[0] == pytest.approx(-0.10) and d[2] == 1
    assert paired_delta(records, "base", "default", "base/default", "lpips_vgg") is None  # type: ignore[arg-type]
    assert paired_delta(records, "h1_iters", "missing", "base/default", "l1") is None  # type: ignore[arg-type]


def test_iters_per_level() -> None:
    target = target_image()
    seq = paint_optim(target, cfg_small(iters_per_level=[7, 13]), L1, BRUSH, seed=0)
    assert [len(c) for c in seq.meta["loss_curve"]] == [7, 13]
    for bad in ([5], [5, 0]):
        with pytest.raises(ValueError):
            paint_optim(target, cfg_small(iters_per_level=bad), L1, BRUSH, seed=0)
