"""Prepare (and, only on request, upload) the final Paint Transformer weights for Hugging Face.

Default = dry run: builds a staging folder ``outputs/hub/<name>/`` with ``pytorch_model.pth`` (the
model ``state_dict`` only, no optimizer or RNG state), ``config.yaml`` (the resolved training
config), ``training_config_sha256.txt`` and a model card ``README.md``; nothing leaves the
machine. ``--push`` uploads that folder with ``huggingface_hub`` (needs ``pip install
huggingface_hub`` and a token in ``HF_TOKEN``); the upload is a later decision of the owner.

    python -m scripts.upload_pt_to_hub --checkpoint outputs/pt_full/seed0/latest.pt \
        --repo-id <user>/brushstroke-painttransformer [--push]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

LICENSE = (
    "mit"  # the licence of this repository; the weights are trained from scratch on synthetic data
)
CARD = """---
license: {license}
library_name: pytorch
tags: [stroke-based-rendering, painting, paint-transformer]
---

# BrushStroke Paint Transformer ({name})

Stroke predictor re-implemented from *Paint Transformer: Feed Forward Neural Painting with Stroke
Prediction* (Liu et al., ICCV 2021) in the BrushStroke project, **trained from scratch** with the
official recipe on the on-the-fly synthetic stroke data of the paper (no image dataset involved, no
weights of the original authors are redistributed).

- Architecture: `Painter(5, 8, 256, 8, 3, 3)`, {parameters:,} parameters, input = target and canvas
  patches `(3, 32, 32)`, output = 8 strokes `(x, y, w, h, theta)` + confidence logit; the stroke
  colour is sampled from the target at the predicted centre.
- Training: {steps} steps x batch 64, Adam lr 1e-4 (betas 0.5, 0.999) with the official decay, seed
  {seed}. Training-config SHA-256: `{config_hash}`.
- Held-out synthetic validation (4096 patches): {metrics}
- License: {license} for the weights and the code (https://github.com/CrispyChicken2/BrushAI); the
  original code is Apache-2.0 (`Huage001/PaintTransformer`).

Load: `PaintTransformer().load_state_dict(torch.load("pytorch_model.pth"))` with
`brushstroke.stroke_model.PaintTransformer`.
"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_staging(checkpoint: Path, out_dir: Path, name: str, metrics: dict[str, float]) -> Path:
    """Write the staging folder for ``checkpoint`` (a ``latest.pt`` of ``train_pt``)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)  # own checkpoint
    config_file = checkpoint.parent / "config.resolved.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state["model"], out_dir / "pytorch_model.pth")
    (out_dir / "config.yaml").write_bytes(config_file.read_bytes())
    config_hash = sha256_file(config_file)
    (out_dir / "training_config_sha256.txt").write_text(config_hash + "\n", encoding="utf-8")
    parameters = sum(v.numel() for v in state["model"].values() if v.dtype.is_floating_point)
    shown = ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()) or "see the project report"
    card = CARD.format(
        license=LICENSE,
        name=name,
        parameters=parameters,
        steps=state["step"],
        seed=state["config"]["seed"],
        config_hash=config_hash,
        metrics=shown,
    )
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", default="<user>/brushstroke-painttransformer")
    parser.add_argument("--name", default="official recipe")
    parser.add_argument(
        "--metrics-json", type=Path, default=None, help="final_eval.json entry file"
    )
    parser.add_argument("--seed-key", default=None, help="entry of the metrics file, e.g. seed0")
    parser.add_argument("--out", type=Path, default=Path("outputs/hub/painttransformer"))
    parser.add_argument("--push", action="store_true", help="really upload (default: dry run)")
    args = parser.parse_args(argv)
    metrics: dict[str, float] = {}
    if args.metrics_json and args.seed_key:
        entry = json.loads(args.metrics_json.read_text(encoding="utf-8"))[args.seed_key]
        metrics = {k: v for k, v in entry.items() if k != "step" and v is not None}
    staging = build_staging(args.checkpoint, args.out, args.name, metrics)
    print(f"staging folder ready: {staging}")
    if not args.push:
        print("dry run: nothing uploaded (use --push to upload)")
        return staging
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(staging), repo_id=args.repo_id)
    return staging


if __name__ == "__main__":
    main()
