"""Generate (and verify) the 30 SDXL images of the evaluation set (T-024).

    python -m scripts.generate_eval_sdxl [--update] [--out data/raw/eval] [--only 4 14] [--rerun-dir DIR]

Settings are those of ``configs/eval/sdxl_prompts.yaml`` (library defaults, nothing tuned): SDXL base 1.0 at the pinned
revision, fp16 weights, model CPU offload, 1024 x 1024, default steps and guidance, no negative prompt, no refiner,
``torch.Generator("cpu").manual_seed(prompt index)``. Each image is verified against the manifest:
``sha256`` is the SHA-256 of the uint8 RGB pixels of the 1024 px output, ``prepared_sha256`` that of the 512 px
image. ``--update`` writes the hashes of a first generation into the manifest. A mismatch is reported with the
pixel difference (GPU kernels are not guaranteed bitwise reproducible across drivers and devices) and makes the
command fail unless ``--allow-mismatch`` is given. With ``--rerun-dir`` the images go to that directory instead and are
compared with the ones in ``--out`` (the determinism check of the data card).

The record ``docs/experiments/data/eval_set/sdxl_generation.json`` stores the library versions, the time per image and the
peak VRAM of the run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from brushstroke.eval.dataset import dump_manifest, load_manifest, pixel_sha256, prepare_image

MANIFEST = Path("configs/eval/eval_set.yaml")
PROMPTS = Path("configs/eval/sdxl_prompts.yaml")
RECORD = Path("docs/experiments/data/eval_set/sdxl_generation.json")
CACHE = Path("data/raw/hf")
MAX_RETRIES = 3  # technical failures only (black output)


def load_pipeline(settings: dict[str, Any]) -> Any:
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        settings["model"],
        revision=settings["revision"],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir=str(CACHE),
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def is_technical_failure(pixels: np.ndarray) -> bool:
    """A black or constant output (the known fp16 VAE failure), never an aesthetic judgement."""
    return bool(pixels.std() < 1.0 or pixels.mean() < 2.0)


def generate_one(pipe: Any, prompt: str, seed: int) -> np.ndarray:
    import torch

    generator = torch.Generator("cpu").manual_seed(seed)
    image: Image.Image = pipe(prompt=prompt, generator=generator).images[0]
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/eval"))
    parser.add_argument("--update", action="store_true", help="record the hashes in the manifest")
    parser.add_argument("--only", type=int, nargs="*", help="prompt indices to generate")
    parser.add_argument(
        "--rerun-dir", type=Path, help="write here and compare with --out (determinism check)"
    )
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args(argv)

    import torch

    spec = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
    header, images = load_manifest(MANIFEST)
    entries = {im.seed: im for im in images if im.category == "sdxl"}
    target = args.rerun_dir or args.out
    pipe = load_pipeline(spec["settings"])
    torch.cuda.reset_peak_memory_stats()
    record: dict[str, Any] = {"images": {}}
    mismatches, next_seed = 0, len(spec["prompts"])
    for p in spec["prompts"]:
        idx = p["index"]
        if args.only and idx not in args.only:
            continue
        entry = entries[idx]
        t0 = time.perf_counter()
        seed, tries = idx, 0
        pixels = generate_one(pipe, p["prompt"], seed)
        while (
            is_technical_failure(pixels) and tries < MAX_RETRIES
        ):  # logged, never for aesthetic reasons
            tries += 1
            seed = next_seed  # the next unused seed (0 to 29 are the prompt indices)
            next_seed += 1
            print(
                f"  technical failure for prompt {idx}: regenerating with seed {seed}", flush=True
            )
            pixels = generate_one(pipe, p["prompt"], seed)
        seconds = time.perf_counter() - t0
        prepared = prepare_image(Image.fromarray(pixels))
        sha, psha = pixel_sha256(pixels), pixel_sha256(prepared)
        for sub, arr in (("sdxl", pixels), ("prepared", prepared)):
            path = target / sub / f"{entry.id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(path)
        status = "new"
        if args.rerun_dir:
            ref = np.asarray(
                Image.open(args.out / "sdxl" / f"{entry.id}.png").convert("RGB"), dtype=np.int16
            )
            diff = np.abs(ref - pixels.astype(np.int16))
            status = (
                "identical"
                if not diff.any()
                else f"max |diff| {int(diff.max())}, mean {diff.mean():.4f}"
            )
            mismatches += bool(diff.any())
        elif entry.sha256:
            ok = entry.sha256 == sha and entry.prepared_sha256 == psha
            status = "verified" if ok else "MISMATCH"
            mismatches += not ok
        elif args.update:
            entry.sha256, entry.prepared_sha256 = sha, psha
            entry.retrieved = time.strftime("%Y-%m-%d")
            if tries:
                entry.extra["regenerated"] = {"technical_failures": tries, "seed": seed}
            status = "recorded"
        record["images"][entry.id] = {"seconds": round(seconds, 2), "status": status, "seed": seed}
        print(f"{entry.id}: {status} ({seconds:.1f} s)", flush=True)
    record["peak_vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
    record["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    import diffusers
    import transformers

    record["versions"] = {"torch": torch.__version__, "diffusers": diffusers.__version__, "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0)}  # fmt: skip
    record["mean_seconds"] = round(
        float(np.mean([v["seconds"] for v in record["images"].values()])), 2
    )
    if args.update:
        dump_manifest(
            MANIFEST,
            {**header, "sdxl": {**header.get("sdxl", {}), "versions": record["versions"]}},
            images,
        )
    name = "sdxl_generation_rerun.json" if args.rerun_dir else "sdxl_generation.json"
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    (RECORD.parent / name).write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "images"}))
    if mismatches and not args.allow_mismatch:
        raise SystemExit(f"{mismatches} image(s) differ from the manifest / the first generation")


if __name__ == "__main__":
    main(sys.argv[1:])
