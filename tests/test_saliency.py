"""Tests of the evaluation saliency maps (T-025): spectral residual, pins, hashes, the wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from brushstroke.eval import saliency as sal
from brushstroke.eval.dataset import load_manifest

MODEL_FILES = Path("data/raw/inspyrenet")
needs_inspyrenet = pytest.mark.skipif(
    not (torch.cuda.is_available() and (MODEL_FILES / "swinb_duts_tr.pth").exists()),
    reason="needs CUDA and the fetched InSPyReNet files (python -m scripts.fetch_inspyrenet)",
)


def blob_image(top: int, left: int, size: int = 40) -> np.ndarray:
    """A smooth grey background (fixed seed) with one bright square: a lone salient object."""
    bg = gaussian_filter(np.random.default_rng(0).normal(size=(512, 512)), 6)
    bg = (bg - bg.min()) / (bg.max() - bg.min()) * 40 + 100
    img = np.repeat(bg[..., None], 3, axis=2).astype(np.uint8)
    img[top : top + size, left : left + size] = 250
    return img


@pytest.mark.parametrize(
    ("top", "left"), [(80, 360), (240, 240), (330, 150), (300, 300), (200, 380), (150, 100)]
)
def test_the_spectral_residual_peaks_on_a_lone_bright_blob(top: int, left: int) -> None:
    s = sal.spectral_residual(blob_image(top, left))
    peak = np.unravel_index(int(s.argmax()), s.shape)
    centre = (top + 20, left + 20)
    assert abs(peak[0] - centre[0]) <= 20 and abs(peak[1] - centre[1]) <= 20
    assert s.dtype == np.float32 and s.shape == (512, 512)
    assert s.min() == 0.0 and s.max() == 1.0
    assert (
        s[peak] == 1.0 and s[(centre[0] + 256) % 512, (centre[1] + 256) % 512] < 0.5
    )  # far from the blob


def test_the_spectral_residual_is_deterministic_and_zero_on_a_uniform_image() -> None:
    img = blob_image(100, 100)
    assert np.array_equal(sal.spectral_residual(img), sal.spectral_residual(img.copy()))
    assert not sal.spectral_residual(np.full((64, 64, 3), 77, dtype=np.uint8)).any()
    non_square = sal.spectral_residual(np.zeros((96, 160, 3), dtype=np.uint8))
    assert non_square.shape == (96, 160)


def test_inputs_are_checked_and_maps_are_hashed_as_float32() -> None:
    with pytest.raises(ValueError):
        sal.spectral_residual(np.zeros((8, 8, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        sal.spectral_residual(np.zeros((8, 8), dtype=np.uint8))
    m = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    assert sal.map_sha256(m) == sal.map_sha256(m.copy()) != sal.map_sha256(m + 1e-6)
    with pytest.raises(ValueError):
        sal.map_sha256(m.astype(np.float64))
    assert not sal.minmax(np.full((3, 3), 5.0)).any()
    assert sal.minmax(np.array([2.0, 4.0, 6.0]))[1] == pytest.approx(0.5)


def test_the_pins_are_complete_and_a_wrong_file_is_reported(tmp_path: Path) -> None:
    spec = sal.load_spec()
    assert len(spec["checkpoint"]["sha256"]) == 64 and len(spec["code"]["commit"]) == 40
    assert {"lib/InSPyReNet.py", "lib/backbones/SwinTransformer.py", "LICENSE"} <= set(
        spec["code"]["files"]
    )
    assert all(len(h) == 64 for h in spec["code"]["files"].values())
    assert any("missing" in p for p in sal.verify_files(tmp_path, spec))  # nothing fetched there
    (tmp_path / "code" / "lib").mkdir(parents=True)
    (tmp_path / "code" / "lib" / "InSPyReNet.py").write_text("not the pinned file")
    problems = sal.verify_files(tmp_path, spec)
    assert any("InSPyReNet.py: SHA-256 differs" in p for p in problems)


def test_the_model_refuses_files_that_are_not_the_pinned_ones(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fetch_inspyrenet"):
        sal.InSPyReNetSaliency("cpu", root=tmp_path)


def test_the_recorded_hashes_cover_the_whole_evaluation_set() -> None:
    record_path = Path("docs/experiments/data/saliency/saliency_maps.json")
    if not record_path.exists():
        pytest.skip("maps not computed yet")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    _, images = load_manifest(Path("configs/eval/eval_set.yaml"))
    assert set(record["images"]) == {im.id for im in images}
    spec = sal.load_spec()
    assert record["model"]["checkpoint_sha256"] == spec["checkpoint"]["sha256"]
    for entry in record["images"].values():
        for method in ("inspyrenet", "spectral_residual"):
            assert len(entry[method]["sha256"]) == 64 and 0.0 <= entry[method]["mean"] <= 1.0
    assert record["cost"]["inspyrenet_peak_vram_mb"] < 3000  # the planning estimate of the M3 plan


@needs_inspyrenet
def test_inspyrenet_finds_a_lone_object_and_is_deterministic() -> None:
    yy, xx = np.mgrid[:512, :512]
    ellipse = (((yy - 256) / 90) ** 2 + ((xx - 256) / 130) ** 2) < 1
    img = np.full((512, 512, 3), (90, 110, 90), dtype=np.uint8)
    img[ellipse] = (200, 40, 40)
    model = sal.InSPyReNetSaliency("cuda")
    a, b = model(img), model(img)
    assert a.shape == (512, 512) and a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0 + 1e-6
    assert np.array_equal(a, b)
    assert a[ellipse].mean() > 0.8 and a[~ellipse].mean() < 0.1
    with pytest.raises(ValueError, match="pyramid"):
        model(np.zeros((640, 640, 3), dtype=np.uint8))


@needs_inspyrenet
def test_the_cached_maps_match_the_record_for_one_image() -> None:
    record_path = Path("docs/experiments/data/saliency/saliency_maps.json")
    prepared = Path("data/raw/eval/prepared/met_748947.png")
    if not record_path.exists() or not prepared.exists():
        pytest.skip("maps or prepared images missing")
    from PIL import Image

    record = json.loads(record_path.read_text(encoding="utf-8"))
    pixels = np.asarray(Image.open(prepared).convert("RGB"))
    entry = record["images"]["met_748947"]
    assert sal.map_sha256(sal.InSPyReNetSaliency("cuda")(pixels)) == entry["inspyrenet"]["sha256"]
    assert sal.map_sha256(sal.spectral_residual(pixels)) == entry["spectral_residual"]["sha256"]
