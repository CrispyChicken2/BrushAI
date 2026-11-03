"""Benchmark the bounding-box rasterization path (T-027) on the strokes of a real baseline sequence.

Usage (repo root, idle GPU)::

    python -m scripts.bench_bbox --sequence outputs/fresh_clone/outputs/m1/m1_landscape/sequence.npz \\
        --out docs/experiments/renderer_bbox_bench.md

Times forward+backward of a whole sequence at 512^2 (what every step of the last baseline levels
does, ``finetune_all``) for the full-canvas path and for the bounding-box path with several chunk
sizes, with the gradient checkpointing the baseline picks for that many strokes, and reports peak
allocated VRAM and the largest deviation of the image and of the gradient from the full path.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from brushstroke.renderer import RenderConfig, render
from brushstroke.renderer.brushes import get_brush_bank
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.device import get_device

REPEATS = 3
CHUNKS = (128, 64, 32, 16)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run(
    seq: StrokeSequence, n: int, cfg: RenderConfig, dev: torch.device
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """(seconds per forward+backward, peak GB, image, gradient)."""
    params = seq.params[:n][None].to(dev).clone().requires_grad_()
    assert seq.brush_index is not None
    idx = seq.brush_index[:n][None].to(dev)
    bank = get_brush_bank(
        seq.brush.kind,
        seq.brush.bank_size,
        (seq.brush.mask_height, seq.brush.mask_width),
        seq.brush.seed,
    ).to(dev)
    h, w = seq.canvas_size
    target = torch.full((1, 3, h, w), 0.4, device=dev)

    def step() -> torch.Tensor:
        params.grad = None
        img = render(params, bank, h, w, canvas=seq.background, cfg=cfg, brush_index=idx)
        (img - target).abs().mean().backward()  # type: ignore[no-untyped-call]
        return img.detach()

    step()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        img = step()
    sync()
    secs = (time.perf_counter() - t0) / REPEATS
    assert params.grad is not None
    return secs, torch.cuda.max_memory_allocated() / 2**30, img, params.grad.clone()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequence", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--counts", type=int, nargs="*", default=[64, 256, 1024, 2037])
    args = ap.parse_args()
    dev = get_device("auto")
    seq = StrokeSequence.load(args.sequence)
    rows = [
        "| strokes | path | chunk | checkpoint | time / step | peak VRAM | max image dev | max grad dev (rel) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for n in args.counts:
        n = min(n, len(seq))
        ckpt = n * 512 * 512 * 44 > 6e9  # the baseline's rule (checkpoint_bytes)
        base = RenderConfig(chunk_size=64, use_checkpoint=ckpt)
        t0, m0, img0, g0 = run(seq, n, base, dev)
        rows.append(
            f"| {n} | full | 64 | {'yes' if ckpt else 'no'} | {t0 * 1000:.0f} ms | {m0:.2f} GB | 0 | 0 |"
        )
        for chunk in CHUNKS:
            cfg = RenderConfig(chunk_size=64, use_checkpoint=ckpt, bbox=True, bbox_chunk=chunk)
            t, m, img, g = run(seq, n, cfg, dev)
            di = float((img - img0).abs().max())
            dg = float((g - g0).abs().max()) / max(float(g0.abs().max()), 1e-12)
            rows.append(
                f"| {n} | bbox | {chunk} | {'yes' if ckpt else 'no'} | {t * 1000:.0f} ms ({t0 / t:.1f}x) | {m:.2f} GB | {di:.1e} | {dg:.1e} |"
            )
            print(rows[-1], flush=True)
    text = "\n".join(rows)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
