"""Tests for brushstroke.utils (config, seed, device, io)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf.errors import ConfigKeyError, ValidationError

from brushstroke.utils.config import RESOLVED_CONFIG_NAME, dump_config, load_config
from brushstroke.utils.device import get_device, peak_vram_mb, reset_peak_vram
from brushstroke.utils.io import load_image, resize_short_side, save_image
from brushstroke.utils.seed import seed_everything


@dataclass
class OptimCfg:
    lr: float = 0.01
    steps: int = 10


@dataclass
class RunCfg:
    seed: int = 0
    name: str = "run"
    optim: OptimCfg = field(default_factory=OptimCfg)


def test_config_defaults() -> None:
    assert load_config(RunCfg) == RunCfg()


def test_config_yaml_and_dot_overrides(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text("seed: 5\noptim:\n  lr: 0.5\n", encoding="utf-8")
    cfg = load_config(RunCfg, yml, ["optim.steps=7", "name=abc", "optim.lr=1e-3"])
    assert cfg == RunCfg(seed=5, name="abc", optim=OptimCfg(lr=1e-3, steps=7))
    assert isinstance(cfg.optim, OptimCfg)


def test_config_dump_round_trip(tmp_path: Path) -> None:
    cfg = load_config(RunCfg, overrides=["seed=3", "optim.lr=0.2"])
    path = dump_config(cfg, tmp_path / "out")
    assert path.name == RESOLVED_CONFIG_NAME
    assert load_config(RunCfg, path) == cfg


def test_config_rejects_unknown_key_and_bad_type() -> None:
    with pytest.raises(ConfigKeyError):
        load_config(RunCfg, overrides=["optim.nope=1"])
    with pytest.raises(ValidationError):
        load_config(RunCfg, overrides=["seed=notanint"])


def test_dump_config_requires_dataclass_instance(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        dump_config({"a": 1}, tmp_path)
    with pytest.raises(TypeError):
        dump_config(RunCfg, tmp_path)


def test_default_yaml_loads() -> None:
    from brushstroke.config import RunConfig

    cfg = load_config(RunConfig, Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.seed == 0
    assert cfg.device == "auto"
    assert cfg.brush.kind == "oil"
    assert cfg.canvas.weave_period == 6.0


def _draw() -> tuple[float, float, list[float]]:
    return (
        random.random(),
        float(np.random.rand()),  # noqa: NPY002
        torch.rand(4).tolist(),
    )


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = _draw()
    seed_everything(123)
    b = _draw()
    assert a == b
    seed_everything(124)
    assert _draw() != a


def test_seed_everything_deterministic_flag_and_reset() -> None:
    try:
        seed_everything(0, deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic
    finally:
        seed_everything(0, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()


def test_seed_everything_rejects_negative() -> None:
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_get_device() -> None:
    assert get_device("cpu").type == "cpu"
    assert get_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        get_device("tpu")


def test_peak_vram_cpu_safe() -> None:
    reset_peak_vram()
    assert peak_vram_mb() >= 0.0


@pytest.mark.gpu
def test_gpu_marker_runs_with_cuda() -> None:
    """Runs only with CUDA (auto-skipped otherwise): matmul + backward, VRAM helper."""
    dev = get_device("cuda")
    reset_peak_vram()
    x = torch.randn(256, 256, device=dev, requires_grad=True)
    (x @ x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert peak_vram_mb() > 0.0


def test_image_round_trip(tmp_path: Path) -> None:
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 24, 32, generator=g)
    path = tmp_path / "sub" / "a.png"
    save_image(img, path)
    out = load_image(path)
    assert out.shape == (3, 24, 32)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.allclose(out, img, atol=0.5 / 255 + 1e-6)


def test_load_image_resize_and_modes(tmp_path: Path) -> None:
    from PIL import Image

    Image.new("L", (40, 20), 128).save(tmp_path / "gray.png")
    Image.new("RGBA", (40, 20), (255, 0, 0, 10)).save(tmp_path / "rgba.png")
    assert load_image(tmp_path / "gray.png").shape == (3, 20, 40)
    rgba = load_image(tmp_path / "rgba.png", size=16)
    assert rgba.shape == (3, 16, 16)
    assert rgba[0].min() > 0.99
    assert load_image(tmp_path / "gray.png", size=(10, 30)).shape == (3, 10, 30)


def test_load_image_16bit_grayscale(tmp_path: Path) -> None:
    from PIL import Image

    arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(arr).save(tmp_path / "g16.png")
    out = load_image(tmp_path / "g16.png")
    assert out.shape == (3, 1, 3)
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 32768 / 65535, 1.0]), atol=1e-6)


def test_save_image_clamps_and_validates(tmp_path: Path) -> None:
    save_image(torch.full((3, 4, 4), 2.0), tmp_path / "w.png")
    assert torch.all(load_image(tmp_path / "w.png") == 1.0)
    with pytest.raises(ValueError):
        save_image(torch.zeros(1, 4, 4), tmp_path / "bad.png")


def test_resize_short_side() -> None:
    out = resize_short_side(torch.rand(3, 100, 200), 50)
    assert out.shape == (3, 50, 100)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_center_crop_resize() -> None:
    from brushstroke.utils.io import center_crop_resize

    img = torch.zeros(3, 100, 60)
    img[:, 20:80, :] = 1.0  # a band the crop to a square must keep
    out = center_crop_resize(img, (32, 32))
    assert out.shape == (3, 32, 32) and out.min() >= 0.0 and out.max() <= 1.0
    wide = center_crop_resize(torch.rand(3, 60, 100), (30, 60))
    assert wide.shape == (3, 30, 60)
