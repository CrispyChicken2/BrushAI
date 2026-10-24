"""Study tools for T-010: run the *official* Paint Transformer code and weights as a reference
point.

Nothing from the official repository is committed. The files (Apache-2.0) are downloaded at a
pinned commit into
``data/raw/pt_official/`` (gitignored) with SHA-256 verification, and the pretrained weights from
the authors' Google Drive
link. **`fetch` followed by `reference`, `datastats` or `trainspeed` downloads and executes
third-party code** (Apache-2.0, pinned by commit
and SHA-256 per file) and loads third-party weights (SHA-256-pinned pickle, loaded with
``weights_only=True``). Usage (repo root)::

    python -m scripts.pt_reference_study fetch [--insecure]     # code at the pinned commit +
    weights
    python -m scripts.pt_reference_study reference              # official inference on the 5
    evaluation images
    python -m scripts.pt_reference_study datastats              # official synthetic-data generator
    statistics
    python -m scripts.pt_reference_study trainspeed             # official training step time and
    VRAM
    python -m scripts.pt_reference_study datagen                # our synthetic generator vs the
    official set_input (speed)

Results are printed and written as JSON in ``outputs/pt_reference/``. The numbers are quoted in
``docs/design/paint_transformer.md``. The weights are a pickle from the internet: they are only
loaded with
``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import ssl
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path
from typing import Any

import torch

COMMIT = "7960b8eef2f6af7171af01933a3d8dbc73190a8d"  # Huage001/PaintTransformer, 2021-10-20 "Update README.md"
RAW = f"https://raw.githubusercontent.com/Huage001/PaintTransformer/{COMMIT}"
ROOT = Path("data/raw/pt_official")
OUT = Path("outputs/pt_reference")
WEIGHTS_ID = "1NDD54BLligyr8tzo8QGI5eihZisXK1nq"
WEIGHTS_SHA256 = "1fbd761ac908c1b53ebc7c9a28e52b8a8afb332f77819c718fe472b7bf6e9dde"
FILES = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "README.md": "be7fef61c64ba2e513ce5e961a1a0a18c5afb22eec5037cd319e99cc882d6d16",
    "inference/inference.py": "952afb8e3b4eb547489d9f054fd9eea30d1d3d856a75d7a377af045690fe2532",
    "inference/network.py": "2bbbadf3fd7739bd53a2dae421cac8d5330ead43256f68109d6e7c574792916c",
    "inference/morphology.py": "763fdf79140ae7fed37a5311ccc75e6b9acb532ab56b29f1ee223004217af154",
    "inference/brush/brush_large_horizontal.png": "d05607378b8ce2efe55a1d35c99d85f9b87b10f696f9481f727acca700ae5bb9",
    "inference/brush/brush_large_vertical.png": "3b2074c1b1f0b32b1b049a3810de544acf61553196d8cf0ce5458a3f64dbc98a",
    "train/train.py": "8e29119bad26181b4745b6004443a33977e3b1892ed62c05b72c67dd46a375c6",
    "train/train.sh": "58a6ece2115b44d38b3143fbdbfeb1d25c41f5b5003a8726ce8dad7b26c095ec",
    "train/models/__init__.py": "d68fc90a5c2aa3ed3bb1833f35d112a47c5afcdb63ca97259c51e5d5629b0684",
    "train/models/base_model.py": "ae160de2311880eb619ea2d62c0e01feeafe036267c6715576ef9d354bf8eae0",
    "train/models/networks.py": "a19646e5eaab80f328767cc8dea1a9df87320075e31f1b0e3e3535f0f93b4bdf",
    "train/models/painter_model.py": "9e44be88314f2b54239a9574d912d2214f3a19f6ca745e910ef2647242019088",
    "train/options/base_options.py": "e7a9831612a34ce96c4252ae05ef4e3352fce7c9fbe7ba316279c54d70043875",
    "train/options/train_options.py": "1230dc7e521dadf14a862c6d8e84797ef9ed3eb18585524009fb3de3bf23a74c",
    "train/options/test_options.py": "6e0df16545bbf8d244442a419de412cfdfee151b0ac50dc3fdabe30d01a8ab3b",
    "train/util/morphology.py": "91d476833b265ec0e959eda56c68e02b980db3f8de68133fbf75f65be8c4bd21",
    "train/util/util.py": "3e2eac39e7c84d31bb1986bc0129d8ec1dec0d99ec80cfbb236c577148e11ffd",
    "train/data/__init__.py": "6824fd37bfce88c4155ac42c4a4bb4a5ec0f6b596066e5e3d80ca11515d2ce89",
    "train/data/base_dataset.py": "8dde22732a9739e0e3a1e81c089ab2096f4ac06bf7781dec3d99acabbf6ebb1d",
    "train/data/null_dataset.py": "d12eabed7a607b0903f47e2a3cc5964895a496f8aad8a153ad44840ae0baddec",
    "train/brush/brush_large_horizontal.png": "9fc5dc4d7d968ad50a1e6b5cfb529e863354843d7c1ff6c8b9547e658d242779",
    "train/brush/brush_large_vertical.png": "7a59df5d865cd7688a44d2e0d7c14b01d8c676178f4e1121e1c19b1f9d1f61c1",
}
IMAGES = ["mona_lisa", "starry_night", "great_wave", "pearl_earring", "earth_apollo17"]
UA = "BrushStrokeResearch/0.1 (research)"


def _get(url: str, insecure: bool) -> bytes:
    ctx = ssl._create_unverified_context() if insecure else None  # noqa: S323
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, context=ctx, timeout=300) as r:  # noqa: S310
        return r.read()  # type: ignore[no-any-return]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(insecure: bool = False) -> None:
    """Download the pinned official files and the pretrained weights; verify the checksums."""
    for rel, digest in FILES.items():
        dest = ROOT / rel
        if not dest.exists() or _sha(dest.read_bytes()) != digest:
            data = _get(f"{RAW}/{rel}", insecure)
            if _sha(data) != digest:
                raise RuntimeError(f"{rel}: checksum mismatch at commit {COMMIT}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
    weights = ROOT / "model.pth"
    if not weights.exists() or _sha(weights.read_bytes()) != WEIGHTS_SHA256:
        # Google Drive serves a "virus scan" page for large files: the confirm token is "t".
        url = (
            f"https://drive.usercontent.google.com/download?id={WEIGHTS_ID}"
            "&export=download&confirm=t"
        )
        data = _get(url, insecure)
        if _sha(data) != WEIGHTS_SHA256:
            raise RuntimeError("pretrained weights: checksum mismatch")
        weights.write_bytes(data)
    print(f"official files at {COMMIT[:8]} and weights ({WEIGHTS_SHA256[:12]}...) are in {ROOT}")


def reference() -> list[dict[str, object]]:
    """Official inference (parallel mode, K=4) on the evaluation images; strokes, time, VRAM and
    our metrics."""
    sys.path.insert(0, str(Path.cwd()))
    from brushstroke.eval.metrics import image_metrics
    from brushstroke.utils.io import center_crop_resize, load_image, save_image

    root = ROOT.resolve()
    img_dir, out_dir = (OUT / "in").resolve(), (OUT / "out").resolve()
    data_dir = Path("data/raw").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "inference"))
    cwd = Path.cwd()
    os.chdir(root / "inference")  # the official script opens brush/*.png relative to its directory
    try:
        import inference as official
        import network

        net = network.Painter(5, 8, 256, 8, 3, 3)
        state = torch.load(root / "model.pth", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        params = sum(p.numel() for p in net.parameters())
        print(f"official Painter: {params} trainable parameters")
        counts: list[tuple[tuple[int, int], int, int]] = []
        original = official.param2img_parallel

        def counting(param, decision, meta_brushes, cur_canvas):  # type: ignore[no-untyped-def]
            counts.append((tuple(decision.shape[1:3]), int(decision.sum()), int(decision.numel())))  # type: ignore[arg-type]
            return original(param, decision, meta_brushes, cur_canvas)

        official.param2img_parallel = counting
        rows: list[dict[str, object]] = []
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for name in IMAGES:
            target = center_crop_resize(load_image(data_dir / f"{name}.jpg"), (512, 512))
            save_image(target, img_dir / f"{name}.png")
            counts.clear()
            if dev.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            official.main(
                str(img_dir / f"{name}.png"), str(root / "model.pth"), str(out_dir), serial=False
            )
            if dev.type == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t0
            painted = load_image(out_dir / f"{name}.png").to(dev)
            metrics = image_metrics(painted, target.to(dev))
            rows.append(
                {
                    "image": name,
                    "seconds_incl_loading": round(seconds, 2),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20)
                    if dev.type == "cuda"
                    else 0,
                    "layers": [list(c) for c in counts],
                    "strokes": sum(c[1] for c in counts),
                    "max_strokes": sum(c[2] for c in counts),
                    **{k: round(v, 4) for k, v in metrics.items()},
                }
            )
            print(json.dumps(rows[-1]))
    finally:
        os.chdir(cwd)
    (OUT / "reference.json").write_text(
        json.dumps({"parameters": params, "rows": rows}, indent=1), encoding="utf-8"
    )
    return rows


def official_paint(
    image_path: str | Path, model_path: str | Path | None = None
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Run the official ``inference.py:main`` (parallel mode) and return its painting as a float
    ``(3, H, W)`` tensor (the official script saves ``(x * 255).astype(uint8)``, which truncates)
    together with the number of painted strokes of each layer.

    The official code uses CUDA when available. Requires ``fetch`` (third-party code, outside git).
    """
    root = ROOT.resolve()
    folder = root / "inference"
    saved_path, cwd = list(sys.path), Path.cwd()
    sys.path.insert(0, str(folder))
    os.chdir(folder)  # the official script opens brush/*.png relative to its directory
    try:
        spec = importlib.util.spec_from_file_location(
            "pt_official_inference_run", folder / "inference.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured: dict[str, torch.Tensor] = {}
        counts: list[tuple[int, int]] = []
        original = module.param2img_parallel

        def counting(param, decision, meta_brushes, cur_canvas):  # type: ignore[no-untyped-def]
            counts.append((int(decision.shape[1]), int(decision.sum())))
            return original(param, decision, meta_brushes, cur_canvas)

        module.param2img_parallel = counting
        module.save_img = lambda img, path: captured.__setitem__("img", img.detach().cpu().clone())
        with tempfile.TemporaryDirectory() as tmp:
            module.main(
                str(Path(image_path).resolve()),
                str(Path(model_path).resolve() if model_path else root / "model.pth"),
                tmp,
                serial=False,
            )
    finally:
        sys.path[:] = saved_path
        os.chdir(cwd)
        for name in ("network", "morphology"):
            sys.modules.pop(name, None)
    return captured["img"], counts


def _official_model(gpu: bool) -> object:
    """The official ``PainterModel`` built with the options of ``train/train.sh`` (batch 64)."""
    train = (ROOT / "train").resolve()
    sys.path.insert(0, str(train))
    os.chdir(train)  # the model reads brush/*.png relative to train/
    from models.painter_model import PainterModel

    opt = types.SimpleNamespace(
        gpu_ids=[0] if gpu else [],
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
    return PainterModel(opt)


def datastats(batches: int = 200) -> dict[str, object]:
    """Statistics of the official synthetic data generator (``set_input``): valid target strokes
    per patch."""
    cwd = Path.cwd()
    try:
        model = _official_model(gpu=False)
        torch.manual_seed(0)
        valid, residual = [], []
        for _ in range(batches):
            model.set_input({"A_paths": ["x"] * 64})  # type: ignore[attr-defined]
            valid.append(model.gt_decision.sum(1))  # type: ignore[attr-defined]
            residual.append(float((model.render - model.old).abs().mean()))  # type: ignore[attr-defined]
        v = torch.cat(valid)
        stats = {
            "patches": int(v.numel()),
            "valid_strokes_mean": float(v.mean()),
            "valid_strokes_std": float(v.std()),
            "histogram_0_to_8": [int((v == k).sum()) for k in range(9)],
            "mean_abs_target_minus_canvas": sum(residual) / len(residual),
        }
    finally:
        os.chdir(cwd)
    print(json.dumps(stats))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datastats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def trainspeed(steps: int = 60) -> dict[str, float]:
    """Time of one official training step (data synthesis + Hungarian matching + backward), batch
    64, fp32, and its phases: ``set_input`` (data), ``forward`` (network + stroke rendering) and
    ``matching`` (the per-sample Hungarian block of ``optimize_parameters``, replicated here and
    timed in isolation). ``rest`` = step - data - forward - matching = loss terms, backward and
    optimizer."""
    from scipy.optimize import linear_sum_assignment

    cwd = Path.cwd()
    try:
        model: Any = _official_model(gpu=True)
        torch.cuda.reset_peak_memory_stats()

        def matching() -> None:
            with torch.no_grad():
                for i in range(model.gt_param.shape[0]):
                    gt = model.gt_param[i, model.gt_decision[i].bool()]
                    pred = model.pred_param[i]
                    l1 = torch.cdist(pred, gt, p=1)
                    pb = pred.unsqueeze(1).repeat(1, gt.shape[0], 1)
                    gb = gt.unsqueeze(0).repeat(pred.shape[0], 1, 1)
                    w = model.gaussian_w_distance(pb, gb)
                    dec = (1 - model.pred_decision[i]).unsqueeze(-1).repeat(1, gt.shape[0])
                    linear_sum_assignment((l1 + w + dec).cpu())

        def clock() -> float:
            torch.cuda.synchronize()
            return time.perf_counter()

        totals = dict.fromkeys(("data", "forward", "matching", "step"), 0.0)
        for i in range(5 + steps):
            t0 = clock()
            model.set_input({"A_paths": ["x"] * 64})
            t1 = clock()
            model.forward()
            t2 = clock()
            matching()
            t3 = clock()
            model.optimize_parameters()
            t4 = clock()
            if i >= 5:
                totals["data"] += t1 - t0
                totals["forward"] += t2 - t1
                totals["matching"] += t3 - t2
                totals["step"] += (t1 - t0) + (t4 - t3)
    finally:
        os.chdir(cwd)
    ms = {f"{k}_ms": 1000 * v / steps for k, v in totals.items()}
    ms["rest_ms"] = ms["step_ms"] - ms["data_ms"] - ms["forward_ms"] - ms["matching_ms"]
    per_step = ms["step_ms"] / 1000
    res = {
        "seconds_per_step": per_step,
        "hours_for_56320_steps": per_step * 56320 / 3600,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        **ms,
    }
    print(json.dumps(res))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trainspeed.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def datagen(batches: int = 30) -> dict[str, dict[str, float]]:
    """Throughput (patches/s) of our synthetic generator vs the official ``set_input`` on the GPU,
    batch 64 / 256."""
    sys.path.insert(0, str(Path.cwd()))
    from brushstroke.renderer.pt import load_pt_brushes
    from brushstroke.stroke_model import SyntheticStrokeGenerator

    cwd = Path.cwd()
    results: dict[str, dict[str, float]] = {}
    ours = SyntheticStrokeGenerator(load_pt_brushes("train", Path("data/raw/pt_brush"), "cuda"))
    for size in (64, 256, 1024):
        g = torch.Generator(device="cuda").manual_seed(0)
        ours.sample(size, g)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batches):
            ours.sample(size, g)
        torch.cuda.synchronize()
        results[f"ours_batch{size}"] = {
            "patches_per_second": size * batches / (time.perf_counter() - t0)
        }
    try:
        model = _official_model(gpu=True)
        model.opt.batch_size = 64  # type: ignore[attr-defined]
        model.set_input({"A_paths": ["x"] * 64})  # type: ignore[attr-defined]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(batches):
            model.set_input({"A_paths": ["x"] * 64})  # type: ignore[attr-defined]
        torch.cuda.synchronize()
        results["official_batch64"] = {
            "patches_per_second": 64 * batches / (time.perf_counter() - t0)
        }
    finally:
        os.chdir(cwd)
    print(json.dumps(results))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "datagen.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "command", choices=["fetch", "reference", "datastats", "trainspeed", "datagen"]
    )
    parser.add_argument(
        "--insecure", action="store_true", help="skip TLS certificate verification (proxy)"
    )
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.insecure)
    elif args.command == "reference":
        reference()
    elif args.command == "datastats":
        datastats()
    elif args.command == "datagen":
        datagen()
    else:
        trainspeed()


if __name__ == "__main__":
    main(sys.argv[1:])
