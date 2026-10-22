"""Baseline (a): paint an image by progressive stroke optimization.

Usage (repo root)::

    python -m scripts.paint_optim --config configs/baseline_optim.yaml --input data/raw/img.jpg \\
        --out outputs/optim_run [key.sub=value ...]

Writes into ``--out``: ``sequence.npz`` (StrokeSequence), ``final.png``, ``target.png``,
``progress/level_XX.png`` (canvas after each level), ``config.resolved.yaml``, ``loss_curve.json``
and ``metrics.json`` (L1, LPIPS-VGG, stroke count, runtime, peak VRAM).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from brushstroke.baselines.optim.config import PaintOptimConfig
from brushstroke.baselines.optim.losses import lpips_loss
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.config import dump_config, load_config
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image, save_image
from brushstroke.utils.seed import seed_everything


def evaluate(final: torch.Tensor, target: torch.Tensor, seq: StrokeSequence) -> dict[str, float]:
    """Final-image metrics (both images ``(3, H, W)`` in [0, 1] on the same device)."""
    mse = float((final - target).square().mean())
    with torch.no_grad():
        lp = float(lpips_loss(final[None], target[None]))
    return {
        "l1": float((final - target).abs().mean()),
        "psnr": float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12)))),
        "lpips_vgg": lp,
        "num_strokes": float(len(seq)),
        "runtime_seconds": float(seq.meta["runtime_seconds"]),
        "peak_vram_mb": float(seq.meta["peak_vram_mb"]),
    }


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_optim.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="input image")
    parser.add_argument(
        "--out", type=Path, default=None, help="output directory (default: output_dir)"
    )
    parser.add_argument(
        "overrides", nargs="*", help="dot overrides, e.g. optim.iters=50 loss.ot=0.1"
    )
    args = parser.parse_args(argv)

    cfg = load_config(PaintOptimConfig, args.config, args.overrides)
    out = args.out or Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    size = (cfg.optim.height, cfg.optim.width)
    target = center_crop_resize(load_image(args.input), size)
    save_image(target, out / "target.png")

    def on_level(level: int, canvas: torch.Tensor, seq: StrokeSequence) -> None:
        save_image(canvas, out / "progress" / f"level_{level:02d}.png")
        print(
            f"level {level} (grid {cfg.optim.levels[level]}): {len(seq)} strokes, "
            f"L1 {float((canvas.cpu() - target).abs().mean()):.4f}",
            flush=True,
        )

    seq = paint_optim(
        target, cfg.optim, cfg.loss, cfg.brush, cfg.render, cfg.seed, device, callback=on_level
    )
    seq.save(out / "sequence.npz")
    final = seq.render(cfg=cfg.render).to(device)
    save_image(final, out / "final.png")
    metrics = evaluate(final, target.to(device), seq)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "loss_curve.json").write_text(
        json.dumps({"levels": cfg.optim.levels, "loss_per_step": seq.meta["loss_curve"]}),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
