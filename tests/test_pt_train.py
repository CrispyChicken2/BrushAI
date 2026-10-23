"""Tests of the Paint Transformer training pipeline (T-013)."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from brushstroke.stroke_model import PTModelConfig
from brushstroke.stroke_model.train import (
    ScheduleConfig,
    TrainConfig,
    ValConfig,
    _Context,
    make_validation_set,
    profile_step,
    train,
)
from brushstroke.utils.config import load_config
from tests import pt_official as po

ROOT = Path(__file__).resolve().parents[1]
needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
TINY = PTModelConfig(hidden_dim=32, heads=4, enc_layers=1, dec_layers=1)  # dropout 0.1 kept


def tiny_config(out: Path, **kw: object) -> TrainConfig:
    return TrainConfig(
        device="cpu",
        output_dir=str(out),
        brush_dir=str(po.BRUSHES),
        batch_size=8,
        log_every=2,
        checkpoint_every=2,
        model=TINY,
        schedule=ScheduleConfig(max_steps=6),
        val=ValConfig(batches=2, every=3),
        **kw,  # type: ignore[arg-type]
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------------ schedule and config


def test_default_schedule_is_the_official_one() -> None:
    s = ScheduleConfig()
    assert s.total_steps == 56_320 and s.total_steps * 64 == 3_604_480  # 3.60 M samples
    lr = lambda epoch: 1e-4 * s.lr_factor((epoch - 1) * s.steps_per_epoch)  # noqa: E731
    first_step_of = lambda epoch: (epoch - 1) * s.steps_per_epoch  # noqa: E731
    for epoch in (1, 100, 204):
        assert lr(epoch) == pytest.approx(1e-4)
    for epoch in (205, 209):
        assert lr(epoch) == pytest.approx(3e-5)
    for epoch in (210, 214):
        assert lr(epoch) == pytest.approx(9e-6)
    for epoch in (215, 219):
        assert lr(epoch) == pytest.approx(2.7e-6)
    assert lr(220) == pytest.approx(8.1e-7)
    # the factor changes exactly at the epoch boundary
    assert s.lr_factor(first_step_of(205) - 1) == 1.0 and s.lr_factor(first_step_of(205)) == 0.3


def test_yaml_config_matches_the_defaults_and_the_official_recipe() -> None:
    cfg = load_config(TrainConfig, ROOT / "configs" / "stroke_model" / "pt.yaml")
    assert cfg == TrainConfig()
    assert (cfg.optim.lr, cfg.optim.beta1, cfg.optim.beta2, cfg.batch_size) == (
        1e-4,
        0.5,
        0.999,
        64,
    )
    assert cfg.loss.preset == "official" and cfg.precision == "fp32"


# ------------------------------------------------------------------ validation set


@needs_brushes
def test_validation_set_depends_only_on_its_seed(tmp_path: Path) -> None:
    a = make_validation_set(_Context(tiny_config(tmp_path, seed=0)))
    b = make_validation_set(_Context(tiny_config(tmp_path, seed=7)))  # other training seed
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x.target, y.target) and torch.equal(x.gt_param, y.gt_param)
    base = tiny_config(tmp_path)
    c = make_validation_set(_Context(replace(base, val=replace(base.val, seed=1))))
    assert not torch.equal(a[0].target, c[0].target)
    assert len(a) == 2 and a[0].target.shape[0] == 8


# ------------------------------------------------------------------ determinism and resume


def final_state(out: Path) -> dict[str, object]:
    return torch.load(out / "latest.pt", map_location="cpu", weights_only=False)  # type: ignore[no-any-return]


def assert_same_state(a: dict[str, object], b: dict[str, object]) -> None:
    assert a["step"] == b["step"]
    for k, v in a["model"].items():  # type: ignore[attr-defined]
        assert torch.equal(v, b["model"][k]), k  # type: ignore[index]
    for key in ("exp_avg", "exp_avg_sq", "step"):
        for sa, sb in zip(
            a["optimizer"]["state"].values(),  # type: ignore[index, attr-defined]
            b["optimizer"]["state"].values(),  # type: ignore[index, attr-defined]
            strict=True,
        ):
            assert torch.equal(sa[key], sb[key]), key
    assert a["scheduler"]["last_epoch"] == b["scheduler"]["last_epoch"]  # type: ignore[index]


@needs_brushes
def test_resume_is_bitwise_identical_on_cpu(tmp_path: Path) -> None:
    """6 steps straight == 3 steps then resume to 6 (model, optimizer, scheduler, RNG, val)."""
    straight = tiny_config(tmp_path / "straight")
    train(straight)
    split = tiny_config(tmp_path / "split")
    train(replace(split, schedule=replace(split.schedule, max_steps=3)))
    assert final_state(tmp_path / "split")["step"] == 3
    train(replace(split, resume="auto"))
    assert_same_state(final_state(tmp_path / "straight"), final_state(tmp_path / "split"))
    val_a, val_b = (read_csv(tmp_path / d / "val.csv") for d in ("straight", "split"))
    assert [r["step"] for r in val_a] == [r["step"] for r in val_b] == ["0", "3", "6"]
    assert val_a == val_b


@needs_brushes
def test_same_seed_same_run_and_different_seed_differs(tmp_path: Path) -> None:
    train(tiny_config(tmp_path / "a", seed=3))
    train(tiny_config(tmp_path / "b", seed=3))
    train(tiny_config(tmp_path / "c", seed=4))
    a, b, c = (final_state(tmp_path / d) for d in "abc")
    assert_same_state(a, b)
    key = "linear_decider.weight"
    assert not torch.equal(a["model"][key], c["model"][key])  # type: ignore[index]


@needs_brushes
def test_resume_without_checkpoint_starts_fresh_and_logs_are_written(tmp_path: Path) -> None:
    summary = train(replace(tiny_config(tmp_path), resume="auto"))
    assert summary["steps"] == 6 and summary["parameters"] > 0
    rows = read_csv(tmp_path / "train.csv")
    assert [int(r["step"]) for r in rows] == [2, 4, 6]
    assert {"total", "pixel", "gt", "w", "decision", "lr", "step_ms"} <= set(rows[0])
    assert all(float(r["total"]) > 0 for r in rows) and float(rows[0]["lr"]) == pytest.approx(1e-4)
    val = read_csv(tmp_path / "val.csv")
    assert [int(r["step"]) for r in val] == [0, 3, 6] and "count_error" in val[0]


@needs_brushes
def test_bf16_autocast_runs_and_keeps_finite_fp32_losses(tmp_path: Path) -> None:
    summary = train(tiny_config(tmp_path, precision="bf16"))
    last = summary["last_val"]
    assert all(torch.isfinite(torch.tensor(v)) for v in last.values())  # type: ignore[attr-defined]


def test_invalid_options_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _Context(tiny_config(tmp_path, precision="fp16"))
    with pytest.raises(ValueError):
        _Context(replace(tiny_config(tmp_path), batch_size=6))


@needs_brushes
def test_profile_reports_every_phase(tmp_path: Path) -> None:
    ms = profile_step(tiny_config(tmp_path), steps=2, warmup=1)
    assert set(ms) == {"data", "forward", "loss", "matching", "backward", "optimizer", "step"}
    assert all(v >= 0 for v in ms.values())
    assert ms["step"] == pytest.approx(
        sum(v for k, v in ms.items() if k not in ("step", "matching"))
    )


# ------------------------------------------------------------------ smoke run on the GPU


@needs_brushes
@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_smoke_run_decreases_the_validation_loss(tmp_path: Path) -> None:
    cfg = TrainConfig(
        output_dir=str(tmp_path),
        brush_dir=str(po.BRUSHES),
        log_every=50,
        schedule=ScheduleConfig(max_steps=300),
        val=ValConfig(batches=4, every=100),
    )
    summary = train(cfg)
    first, last = summary["first_val"], summary["last_val"]
    print(
        f"val total {first['total']:.2f} -> {last['total']:.2f}; "
        f"peak {summary['peak_vram_mb']:.0f} MB"
    )
    assert last["total"] < 0.2 * first["total"]
    assert last["decision"] < first["decision"] and summary["peak_vram_mb"] > 0
    assert [int(r["step"]) for r in read_csv(tmp_path / "val.csv")] == [0, 100, 200, 300]
