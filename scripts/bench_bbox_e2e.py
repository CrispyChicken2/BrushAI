"""End-to-end benchmark of the bounding-box renderer on the optimization baseline (T-027).

Usage (repo root, idle GPU)::

    python -m scripts.bench_bbox_e2e --out outputs/bbox_e2e

Paints the M1 landscape demo (``configs/demos/m1_landscape.yaml``: 512^2, 2037 strokes, seed 0,
``cudnn_deterministic``) once with the full-canvas renderer and once with ``render.bbox=true``, and
writes ``<out>/<mode>/{sequence.npz,metrics.json}`` plus ``<out>/summary.json``: runtime, peak VRAM,
sequence hash and final-image metrics of each, and their differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brushstroke.utils.config import load_config
from scripts.fetch_demo_inputs import DEMO_DIR
from scripts.make_m1_demos import DemoConfig, paint_demo, sequence_sha256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("outputs/bbox_e2e"))
    ap.add_argument("--demo", default="m1_landscape")
    ap.add_argument("--input", type=Path, default=Path("data/raw/demo/starry_night.jpg"))
    ap.add_argument("--modes", nargs="*", default=["off", "on"])
    args = ap.parse_args()
    summary: dict[str, dict[str, object]] = {}
    for mode in args.modes:
        cfg = load_config(
            DemoConfig, DEMO_DIR / f"{args.demo}.yaml", [f"paint.render.bbox={mode == 'on'}"]
        )
        out = args.out / mode
        out.mkdir(parents=True, exist_ok=True)
        seq, _, metrics = paint_demo(cfg, args.input, out)
        entry = {
            "sequence_sha256": sequence_sha256(seq),
            "config_hash": seq.meta["config_hash"],
            **metrics,
        }
        (out / "metrics.json").write_text(json.dumps(entry, indent=1), encoding="utf-8")
        summary[mode] = entry
        print(mode, entry, flush=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if "off" in summary and "on" in summary:
        off, on = summary["off"], summary["on"]
        print("speed-up", float(off["runtime_seconds"]) / float(on["runtime_seconds"]))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
