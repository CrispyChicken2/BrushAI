"""Evaluation saliency maps of the whole evaluation set (T-025): one command, cached outside git, hashes recorded.

    python -m scripts.compute_saliency                 # compute both maps for the 100 prepared images
    python -m scripts.compute_saliency --verify        # recompute and compare with the recorded hashes
    python -m scripts.compute_saliency --sheet         # the small visual sanity sheet (docs/experiments/saliency_sanity.png)

Needs ``python -m scripts.fetch_eval_set`` (prepared 512 x 512 targets) and ``python -m scripts.fetch_inspyrenet``
(pinned model). Maps are float32 ``(512, 512)`` in [0, 1], saved as ``<out>/<method>/<image id>.npy`` (default
``outputs/saliency``, ignored by git; derived data of the dev and test images). Methods: ``inspyrenet`` (the evaluation
model) and ``spectral_residual`` (the model-free control). The record ``docs/experiments/data/saliency/saliency_maps.json``
holds the SHA-256 of every map, summary statistics, the pinned model hashes, the time per image and the peak VRAM.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from brushstroke.eval.dataset import load_manifest
from brushstroke.eval.saliency import (
    CONFIG,
    InSPyReNetSaliency,
    load_spec,
    map_sha256,
    spectral_residual,
)

MANIFEST = Path("configs/eval/eval_set.yaml")
RECORD = Path("docs/experiments/data/saliency/saliency_maps.json")
SHEET = Path("docs/experiments/saliency_sanity.png")
METHODS = ("inspyrenet", "spectral_residual")
SHEET_STRATA = (  # one image of each of these strata (the first of the manifest) in the sanity sheet
    "european_portrait", "european_landscape", "asian", "photo_portrait",
    "photo_animal", "photo_scene", "sdxl_portrait", "sdxl_object",
)  # fmt: skip


def map_path(out: Path, method: str, image_id: str) -> Path:
    return out / method / f"{image_id}.npy"


def load_maps(out: Path, image_id: str) -> dict[str, np.ndarray]:
    """Both cached maps of an image (``inspyrenet``, ``spectral_residual``)."""
    return {m: np.load(map_path(out, m, image_id)) for m in METHODS}


def prepared(root: Path, image_id: str) -> np.ndarray:
    return np.asarray(Image.open(root / "prepared" / f"{image_id}.png").convert("RGB"))


def summary(m: np.ndarray) -> dict[str, float]:
    return {"mean": round(float(m.mean()), 4), "over_half": round(float((m > 0.5).mean()), 4)}


def compute(root: Path, out: Path, device: str) -> dict[str, Any]:
    _, images = load_manifest(MANIFEST)
    model = InSPyReNetSaliency(device)
    use_cuda = torch.device(device).type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    warm = prepared(root, images[0].id)
    model(warm)  # warm-up: kernels, allocator
    t_model: list[float] = []
    t_sr: list[float] = []
    entries: dict[str, Any] = {}
    for im in images:
        pixels = prepared(root, im.id)
        t0 = time.perf_counter()
        a = model(pixels)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        b = spectral_residual(pixels)
        t2 = time.perf_counter()
        t_model.append(t1 - t0)
        t_sr.append(t2 - t1)
        for method, m in zip(METHODS, (a, b), strict=True):
            path = map_path(out, method, im.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, m)
        entries[im.id] = {
            "category": im.category,
            "stratum": im.stratum,
            "inspyrenet": {"sha256": map_sha256(a), **summary(a)},
            "spectral_residual": {"sha256": map_sha256(b), **summary(b)},
            "correlation": round(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]), 3),
        }
    spec = load_spec()
    by_category: dict[str, dict[str, float]] = {}
    for cat in sorted({e["category"] for e in entries.values()}):
        rows = [e for e in entries.values() if e["category"] == cat]
        by_category[cat] = {
            "inspyrenet_mean": round(statistics.fmean(r["inspyrenet"]["mean"] for r in rows), 3),
            "spectral_residual_mean": round(
                statistics.fmean(r["spectral_residual"]["mean"] for r in rows), 3
            ),
            "correlation_mean": round(statistics.fmean(r["correlation"] for r in rows), 3),
        }
    record = {
        "model": {
            "checkpoint_sha256": spec["checkpoint"]["sha256"],
            "code_commit": spec["code"]["commit"],
            "config": str(CONFIG),
        },
        "environment": {
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        },
        "cost": {
            "inspyrenet_ms_per_image": round(1000 * statistics.fmean(t_model), 1),
            "spectral_residual_ms_per_image": round(1000 * statistics.fmean(t_sr), 1),
            "inspyrenet_peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)
            if use_cuda
            else None,
            "inspyrenet_peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)
            if use_cuda
            else None,
            "model_parameters_million": round(
                sum(p.numel() for p in model.model.parameters()) / 1e6, 1
            ),
        },
        "by_category": by_category,
        "images": entries,
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def verify(root: Path, out: Path, device: str) -> int:
    """Recompute the maps and compare them with the record (bitwise) and with the cache; each model run is also repeated."""
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    model = InSPyReNetSaliency(device)
    bad = 0
    worst = 0.0
    for image_id, entry in record["images"].items():
        pixels = prepared(root, image_id)
        a, a2, b = model(pixels), model(pixels), spectral_residual(pixels)
        for method, m in (("inspyrenet", a), ("spectral_residual", b)):
            same = map_sha256(m) == entry[method]["sha256"]
            cached = map_path(out, method, image_id)
            if cached.exists():
                worst = max(worst, float(np.abs(np.load(cached) - m).max()))
            if not same:
                bad += 1
                print(f"{image_id} {method}: differs from the record")
        if map_sha256(a) != map_sha256(a2):
            bad += 1
            print(f"{image_id}: two runs of the model differ")
    print(
        f"{len(record['images'])} images, {bad} difference(s); largest |recomputed - cached| = {worst:.2e}"
    )
    return bad


def sheet(root: Path, out: Path) -> None:
    _, images = load_manifest(MANIFEST)
    picks = []
    for stratum in SHEET_STRATA:
        picks.append(next(im for im in images if im.stratum == stratum))
    tile = 160
    canvas = Image.new("RGB", (3 * tile, len(picks) * tile), "white")
    for row, im in enumerate(picks):
        maps = load_maps(out, im.id)
        tiles = [Image.fromarray(prepared(root, im.id))]
        for method in METHODS:
            tiles.append(
                Image.fromarray((maps[method] * 255).round().astype(np.uint8)).convert("RGB")
            )
        for col, t in enumerate(tiles):
            canvas.paste(t.resize((tile, tile), Image.Resampling.BOX), (col * tile, row * tile))
    canvas = canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(SHEET, optimize=True)
    print(
        f"{SHEET}: {SHEET.stat().st_size} bytes; rows: {[im.id for im in picks]} (image | InSPyReNet | spectral residual)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/eval"))
    parser.add_argument("--out", type=Path, default=Path("outputs/saliency"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--verify", action="store_true", help="recompute and compare with the record"
    )
    parser.add_argument(
        "--sheet", action="store_true", help="write the visual sanity sheet from the cached maps"
    )
    args = parser.parse_args(argv)
    if args.verify:
        raise SystemExit(1 if verify(args.root, args.out, args.device) else 0)
    if args.sheet:
        sheet(args.root, args.out)
        return
    record = compute(args.root, args.out, args.device)
    print(
        json.dumps(
            {k: record[k] for k in ("model", "environment", "cost", "by_category")}, indent=1
        )
    )
    sheet(args.root, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])
