"""Render a stroke sequence as a stroke-by-stroke animation (MP4 / WebM / GIF).

Usage (repo root)::

    python -m scripts.animate --sequence outputs/optim_run/sequence.npz --out outputs/anim \
        [--config configs/animation.yaml] [key=value ...]

e.g. ``formats=[mp4,gif] duration=8 gif_width=360``. Writes ``<out>/<name>.<fmt>`` for every format,
``config.resolved.yaml`` and ``timings.json`` (render + encode seconds, peak VRAM).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from brushstroke.animation import AnimationConfig, FrameRenderer, export_frames
from brushstroke.animation.frames import output_size
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.config import dump_config, load_config
from brushstroke.utils.device import peak_vram_mb, reset_peak_vram


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--sequence", type=Path, required=True, help="StrokeSequence .npz")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=Path("configs/animation.yaml"))
    parser.add_argument("--name", default=None, help="output file stem (default: sequence name)")
    parser.add_argument("overrides", nargs="*", help="dot overrides, e.g. fps=24 formats=[gif]")
    args = parser.parse_args(argv)

    cfg = load_config(AnimationConfig, args.config, args.overrides)
    args.out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, args.out)
    seq = StrokeSequence.load(args.sequence)
    renderer = FrameRenderer(seq, cfg)
    h, w = output_size(seq, cfg)
    reset_peak_vram()
    t0 = time.perf_counter()
    files = export_frames(
        renderer.frames(), args.out / (args.name or args.sequence.stem), (w, h), cfg
    )
    seconds = time.perf_counter() - t0
    info: dict[str, object] = {
        "strokes": len(seq),
        "frames": renderer.timeline.num_frames,
        "size": [w, h],
        "render_and_encode_seconds": seconds,
        "peak_vram_mb": peak_vram_mb(),
        "files": {k: str(v) for k, v in files.items()},
        "sizes_bytes": {k: v.stat().st_size for k, v in files.items()},
    }
    (args.out / "timings.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main(sys.argv[1:])
