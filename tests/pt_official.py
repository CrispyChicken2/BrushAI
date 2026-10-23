"""Helpers to import the *official* Paint Transformer code (downloaded, never committed) for parity
tests.

The files live in ``data/raw/pt_official`` (``python -m scripts.pt_reference_study fetch``) and the
brush images in
``data/raw/pt_brush`` (``python -m scripts.fetch_pt_brushes``); tests that need them skip with a
reason when absent.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).parents[1]
OFFICIAL = ROOT / "data" / "raw" / "pt_official"
BRUSHES = ROOT / "data" / "raw" / "pt_brush"


def have_official() -> bool:
    return (OFFICIAL / "inference" / "inference.py").exists() and (
        OFFICIAL / "train" / "models" / "painter_model.py"
    ).exists()


def have_brushes() -> bool:
    return all(
        (BRUSHES / f"{p}_brush_large_{o}.png").exists()
        for p in ("inference", "train")
        for o in ("vertical", "horizontal")
    )


@contextlib.contextmanager
def _paths(*dirs: Path) -> Iterator[None]:
    saved_path, saved_cwd = list(sys.path), Path.cwd()
    sys.path[:0] = [str(d) for d in dirs]
    try:
        yield
    finally:
        sys.path[:] = saved_path
        os.chdir(saved_cwd)


def official_inference() -> Any:
    """Module ``inference/inference.py`` of the official repository (``param2stroke``,
    ``param2img_parallel``)."""
    folder = OFFICIAL / "inference"
    with _paths(folder):
        spec = importlib.util.spec_from_file_location(
            "pt_official_inference", folder / "inference.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def official_train_model() -> Any:
    """The official ``PainterModel`` (CPU, options of ``train.sh``): ``set_input`` and
    ``param2stroke``."""
    train = OFFICIAL / "train"
    opt = types.SimpleNamespace(
        gpu_ids=[],
        isTrain=True,
        checkpoints_dir="/tmp/pt",
        name="pt",
        preprocess="resize_and_crop",
        used_strokes=8,
        num_blocks=3,
        lambda_w=10.0,
        lambda_pixel=10.0,
        lambda_gt=1.0,
        lambda_decision=10.0,
        lambda_recall=10.0,
        batch_size=64,
        ngf=256,
        init_type="normal",
        init_gain=0.02,
        lr=1e-4,
        beta1=0.5,
        dataset_mode="null",
        verbose=False,
    )
    with _paths(train):
        os.chdir(train)  # the model reads brush/*.png relative to train/
        from models.painter_model import PainterModel  # type: ignore[import-not-found]

        model = PainterModel(opt)
    for name in [
        m for m in sys.modules if m in ("models", "util") or m.startswith(("models.", "util."))
    ]:
        sys.modules.pop(name, None)
    return model


def official_train_morphology() -> Any:
    """``train/util/morphology.py`` (``Dilation2d``, ``Erosion2d``)."""
    path = OFFICIAL / "train" / "util" / "morphology.py"
    spec = importlib.util.spec_from_file_location("pt_official_train_morphology", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_official_brushes(variant: str) -> torch.Tensor:
    """Brush images exactly as the official ``read_img(..., 'L')`` + ``torch.cat``, from the
    official folder."""
    folder = OFFICIAL / variant / "brush"
    out = []
    for name in ("brush_large_vertical.png", "brush_large_horizontal.png"):
        arr = np.array(Image.open(folder / name).convert("L"))[None]
        out.append(torch.from_numpy(arr).unsqueeze(0).float() / 255.0)
    return torch.cat(out, dim=0)


def official_network() -> Any:
    """Module ``inference/network.py`` of the official repository (``Painter``)."""
    folder = OFFICIAL / "inference"
    with _paths(folder):
        spec = importlib.util.spec_from_file_location("pt_official_network", folder / "network.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def have_weights() -> bool:
    return (OFFICIAL / "model.pth").exists()
