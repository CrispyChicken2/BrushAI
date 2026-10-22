"""Benchmark the stroke renderer: time and peak VRAM, forward and forward+backward.

Usage (repo root): ``python -m scripts.bench_renderer [--out outputs/bench_renderer.md]``.
Sweeps N in {256, 1024, 4096} x resolution in {256, 512, 1024}, fp32, plus a bf16 compositing
comparison (error vs fp32). Cases whose activations would exceed ~6 GB use gradient checkpointing
(and the 1024^2 cases a smaller chunk); the configuration is printed in the table.
An out-of-memory case is reported as OOM.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch

from brushstroke.renderer import BrushBank, RenderConfig, render
from brushstroke.utils.device import get_device, peak_vram_mb, reset_peak_vram
from brushstroke.utils.seed import seed_everything

NUM_STROKES = (256, 1024, 4096)
RESOLUTIONS = (256, 512, 1024)
REPEATS = 3


def random_strokes(n: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(1, n, 9, generator=g)
    p[..., 2:4] = 0.03 + 0.2 * p[..., 2:4]
    return p.to(device)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats: int = REPEATS) -> float:  # type: ignore[no-untyped-def]
    fn()  # warm-up
    sync()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    sync()
    return (time.perf_counter() - t0) / repeats


def bench_case(
    n: int, res: int, cfg: RenderConfig, device: torch.device, brush: BrushBank
) -> dict[str, str]:
    gc.collect()
    torch.cuda.empty_cache()  # a full allocator cache makes WDDM page to shared memory (slow)
    params = random_strokes(n, device)
    out: dict[str, str] = {}

    def fwd() -> None:
        with torch.no_grad():
            render(params, brush, res, res, cfg=cfg)

    def fwd_bwd() -> None:
        q = params.clone().requires_grad_()
        render(q, brush, res, res, cfg=cfg).square().sum().backward()

    for name, fn in (("fwd", fwd), ("fwd_bwd", fwd_bwd)):
        try:
            reset_peak_vram()
            seconds = timed(fn)
            out[name] = f"{seconds * 1000:.0f} ms / {peak_vram_mb() / 1024:.2f} GB"
        except torch.cuda.OutOfMemoryError:
            out[name] = "OOM"
            torch.cuda.empty_cache()
    return out


def bf16_error(device: torch.device, brush: BrushBank) -> str:
    params = random_strokes(512, device, seed=1)
    ref = render(params, brush, 512, 512, cfg=RenderConfig(chunk_size=32))
    cfg = RenderConfig(chunk_size=32, dtype="bfloat16")
    low = render(params, brush, 512, 512, cfg=cfg)
    err = (low - ref).abs()
    t32 = timed(lambda: render(params, brush, 512, 512, cfg=RenderConfig(chunk_size=32)))
    t16 = timed(lambda: render(params, brush, 512, 512, cfg=cfg))
    return (
        f"bf16 compositing, N=512, 512^2: max |err| = {err.max():.4f}, "
        f"mean |err| = {err.mean():.5f}; "
        f"fwd {t16 * 1000:.0f} ms vs fp32 {t32 * 1000:.0f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/bench_renderer.md"))
    args = parser.parse_args()
    device = get_device("auto")
    seed_everything(0)
    brush = BrushBank.build("oil", 8, (64, 128), 0).to(device)
    lines = [
        f"device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}, "
        f"torch {torch.__version__}",
        "",
        "| N | resolution | chunk / checkpoint | forward (time / peak VRAM) | forward+backward |",
        "|---|-----------|--------------------|----------------------------|------------------|",
    ]
    for res in RESOLUTIONS:
        for n in NUM_STROKES:
            # Without checkpointing, forward+backward keeps ~44 bytes per stroke pixel.
            big = n * res * res * 44 > 6e9
            cfg = RenderConfig(chunk_size=16 if res >= 1024 else 32, use_checkpoint=big)
            r = bench_case(n, res, cfg, device, brush)
            lines.append(
                f"| {n} | {res}^2 | {cfg.chunk_size} / {'yes' if big else 'no'} "
                f"| {r['fwd']} | {r['fwd_bwd']} |"
            )
            print(lines[-1], flush=True)
    lines += ["", bf16_error(device, brush)]
    print(lines[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
