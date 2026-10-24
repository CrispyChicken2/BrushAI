"""Paint an image with the Paint Transformer multi-scale inference (T-015).

Usage (repo root)::

    python -m scripts.paint_pt --config configs/paint_pt.yaml --input data/raw/mona_lisa.jpg \\
        --out outputs/pt_run --checkpoint outputs/pt_long/seed2/latest.pt [key.sub=value ...]

``--checkpoint`` also accepts the released ``model.pth`` (a plain state_dict). Writes into
``--out``: ``sequence.npz`` (StrokeSequence of ``inference.final_layer``), ``final.png`` (same
variant), ``variants/final_{official,gated}.png``, ``target.png``, ``progress/layer_XX.png`` (canvas
after each layer), ``config.resolved.yaml`` and ``metrics.json`` (L1, PSNR, SSIM, LPIPS of both
final-layer variants, stroke counts, runtime, peak VRAM, and how well the sequence reproduces the
pipeline canvas). Enable the per-patch refinement with ``refine.enabled=true``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from brushstroke.eval.metrics import image_metrics
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.inference import PaintPTConfig, PTResult, load_pt_model, paint
from brushstroke.stroke_model.pt_sequence import replay_sequence
from brushstroke.utils.config import dump_config, load_config
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image, save_image
from brushstroke.utils.seed import seed_everything


def sequence_agreement(
    result: PTResult, canvas: torch.Tensor, cfg: PaintPTConfig
) -> dict[str, float]:
    """How well the exported sequence reproduces the pipeline canvas of its variant.

    ``replay_*``: the layer-aware replay (:func:`replay_sequence`); ``generic_*``: the generic PT
    renderer (``StrokeSequence.render``, every stroke at the final resolution, brush resized to
    the canvas, no window clipping).
    """
    brushes = load_pt_brushes("inference", cfg.inference.brush_dir, canvas.device)
    seq = result.sequence.to(canvas.device)
    replay = replay_sequence(seq, brushes)
    generic = seq.render(cfg=cfg.render)
    d_replay, d_generic = (replay - canvas).abs(), (generic - canvas).abs()
    return {
        "replay_max_abs": float(d_replay.max()),
        "replay_mean_abs": float(d_replay.mean()),
        "replay_pixels_over_1_255": float((d_replay.amax(0) > 1 / 255).sum()),
        "generic_mean_abs": float(d_generic.mean()),
        "generic_pixels_over_1_255": float((d_generic.amax(0) > 1 / 255).sum()),
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/paint_pt.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="input image")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="overrides `checkpoint`")
    parser.add_argument("overrides", nargs="*", help="dot overrides, e.g. refine.enabled=true")
    args = parser.parse_args(argv)

    cfg = load_config(PaintPTConfig, args.config, args.overrides)
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint)
    out = args.out or Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    seed_everything(cfg.seed)
    device = get_device(cfg.device)

    image = load_image(args.input)
    if cfg.height > 0 and cfg.width > 0:
        image = center_crop_resize(image, (cfg.height, cfg.width))
    target = image.to(device)
    save_image(image, out / "target.png")

    net, info = load_pt_model(cfg.checkpoint, device)
    brushes = load_pt_brushes("inference", cfg.inference.brush_dir, device)
    result = paint(net, target, brushes, cfg.inference, cfg.refine)

    for level, canvas in enumerate(result.layer_canvases):
        save_image(canvas, out / "progress" / f"layer_{level:02d}.png")
    save_image(result.canvas, out / "variants" / "final_official.png")
    save_image(result.canvas_gated, out / "variants" / "final_gated.png")
    chosen = result.canvas if cfg.inference.final_layer == "official" else result.canvas_gated
    save_image(chosen, out / "final.png")
    result.sequence.save(out / "sequence.npz")

    metrics: dict[str, object] = {
        "checkpoint": cfg.checkpoint,
        "checkpoint_info": info,
        "image_size": list(image.shape[-2:]),
        "refine": cfg.refine.enabled,
        "exported_variant": cfg.inference.final_layer,
        "official": image_metrics(result.canvas, target),
        "gated": image_metrics(result.canvas_gated, target),
        "sequence_vs_canvas": sequence_agreement(result, chosen, cfg),
        **result.stats,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
