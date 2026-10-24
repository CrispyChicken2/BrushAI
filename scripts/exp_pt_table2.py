"""Table 2 of the Paint Transformer paper on our hardware (T-016): inference time at 128 / 256 / 512 px
and training time. Protocol: ``docs/experiments/pt_reproduction.md``.

    python -m scripts.exp_pt_table2 run      # GPU, idle machine; official code from `pt_reference_study fetch`
    python -m scripts.exp_pt_table2 report   # markdown table from docs/experiments/data/pt_reproduction/table2.json
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

from brushstroke.baselines.optim.config import PaintOptimConfig
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.inference import load_pt_model, paint
from brushstroke.stroke_model.train import TrainConfig, profile_step
from brushstroke.utils.config import load_config
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image, save_image
from scripts.pt_reference_study import official_paint
from scripts.pt_repro_common import (
    IMAGES,
    OFFICIAL_WEIGHTS,
    PAPER_TIME,
    PAPER_TRAIN_HOURS,
    dump,
    load,
    write_block,
)

SIZES = (128, 256, 512)
WARMUP, RUNS = 2, 5
OFFICIAL_TRAIN_LOG = Path("docs/experiments/data/pt_training/official_retrained/train.csv")
OFFICIAL_SPEED = Path("outputs/pt_reference/trainspeed.json")
STEPS = 56_320
WORK = Path("outputs/pt_reproduction")


def inputs(size: int) -> dict[str, Path]:
    paths = {}
    for name in IMAGES:
        path = WORK / "in" / f"{name}_{size}.png"
        if not path.exists():
            save_image(
                center_crop_resize(load_image(Path("data/raw") / f"{name}.jpg"), (size, size)), path
            )
        paths[name] = path
    return paths


def run() -> dict[str, object]:
    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    result: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }
    net, _ = load_pt_model(OFFICIAL_WEIGHTS, device)
    ours: dict[str, list[float]] = {}
    official: dict[str, list[float]] = {}
    for size in SIZES:
        ours[str(size)], official[str(size)] = [], []
        for path in inputs(size).values():
            image = load_image(path).to(device)
            for _ in range(WARMUP):
                paint(net, image, brushes)
            ours[str(size)] += [
                float(paint(net, image, brushes).stats["runtime_seconds"]) for _ in range(RUNS)
            ]  # type: ignore[arg-type]
            for _ in range(WARMUP):
                official_paint(path)
            for _ in range(RUNS):
                timing: dict[str, float] = {}
                official_paint(path, timings=timing)
                official[str(size)].append(timing["seconds"])
        print(
            size,
            "ours",
            statistics.median(ours[str(size)]),
            "official",
            statistics.median(official[str(size)]),
            flush=True,
        )
    result["pt_ours_seconds"] = ours
    result["pt_official_code_seconds"] = official

    cfg = load_config(PaintOptimConfig, "configs/baseline_optim.yaml")
    optim: dict[str, float] = {}
    for size in SIZES:
        cfg.optim.height = cfg.optim.width = size
        target = load_image(inputs(size)["mona_lisa"])
        start = time.perf_counter()
        seq = paint_optim(target, cfg.optim, cfg.loss, cfg.brush, cfg.render, 0, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        optim[str(size)] = time.perf_counter() - start
        print(size, "optim", optim[str(size)], len(seq), flush=True)
    result["optim_seconds"] = optim
    result["optim_strokes"] = 2037

    profile = profile_step(TrainConfig(), steps=50, warmup=10)
    rows = list(csv.DictReader(OFFICIAL_TRAIN_LOG.open(encoding="utf-8")))
    result["training"] = {
        "ours_step_ms": profile["step"],
        "ours_hours": profile["step"] * STEPS / 3.6e6,
        "official_code_hours_retrain": float(rows[-1]["elapsed_s"]) / 3600,
        "official_code_step_ms_retrain": float(rows[-1]["elapsed_s"])
        / int(rows[-1]["step"])
        * 1000,
    }
    if OFFICIAL_SPEED.exists():
        import json

        speed = json.loads(OFFICIAL_SPEED.read_text(encoding="utf-8"))
        result["training"]["official_code_step_ms_t010"] = speed["step_ms"]  # type: ignore[index]
        result["training"]["official_code_hours_t010"] = speed["hours_for_56320_steps"]  # type: ignore[index]
    dump("table2", result)
    return result


def report() -> None:
    data = load("table2")
    med = lambda xs: statistics.median(xs)  # noqa: E731
    pt = [med(data["pt_ours_seconds"][str(s)]) for s in SIZES]
    off = [med(data["pt_official_code_seconds"][str(s)]) for s in SIZES]
    opt = [data["optim_seconds"][str(s)] for s in SIZES]
    paper_pt, paper_opt = PAPER_TIME["PT"], PAPER_TIME["Optim"]
    lines = [
        f"**T2a, inference time (s)** at 128 / 256 / 512 px ({data['gpu']}; paper: 2080 Ti). PT = median of 25 runs (5 images x 5), Optim = one run on Mona Lisa.",
        "",
        "| method | 128 | 256 | 512 | growth 128 -> 512 |",
        "|---|---|---|---|---|",
        f"| paper PT | {paper_pt[0]} | {paper_pt[1]} | {paper_pt[2]} | {paper_pt[2] / paper_pt[0]:.1f}x |",
        f"| paper RL | {PAPER_TIME['RL'][0]} | {PAPER_TIME['RL'][1]} | {PAPER_TIME['RL'][2]} | {PAPER_TIME['RL'][2] / PAPER_TIME['RL'][0]:.1f}x |",
        f"| paper Optim | {paper_opt[0]} | {paper_opt[1]} | {paper_opt[2]} | {paper_opt[2] / paper_opt[0]:.1f}x |",
        f"| **ours PT** (official weights) | {pt[0]:.3f} | {pt[1]:.3f} | {pt[2]:.3f} | {pt[2] / pt[0]:.1f}x |",
        f"| official PT code (same weights) | {off[0]:.3f} | {off[1]:.3f} | {off[2]:.3f} | {off[2] / off[0]:.1f}x |",
        f"| **ours Optim** (M1 baseline, 2,037 strokes) | {opt[0]:.1f} | {opt[1]:.1f} | {opt[2]:.1f} | {opt[2] / opt[0]:.1f}x |",
        "",
        "| ratio Optim / PT | 128 | 256 | 512 |",
        "|---|---|---|---|",
        f"| paper | {paper_opt[0] / paper_pt[0]:.0f} | {paper_opt[1] / paper_pt[1]:.0f} | {paper_opt[2] / paper_pt[2]:.0f} |",
        f"| ours (Optim / our PT) | {opt[0] / pt[0]:.0f} | {opt[1] / pt[1]:.0f} | {opt[2] / pt[2]:.0f} |",
        f"| ours (Optim / official PT code) | {opt[0] / off[0]:.0f} | {opt[1] / off[1]:.0f} | {opt[2] / off[2]:.0f} |",
    ]
    t = data["training"]
    lines += [
        "",
        "**T2b, training time of the stroke predictor** (56,320 steps, batch 64):",
        "",
        "| | step (ms) | total (h) |",
        "|---|---|---|",
        f"| paper PT (2080 Ti) | | {PAPER_TRAIN_HOURS['PT']} |",
        f"| official code, our GPU, T-010 benchmark | {t.get('official_code_step_ms_t010', float('nan')):.0f} | {t.get('official_code_hours_t010', float('nan')):.2f} |",
        f"| official code, our GPU, full retrain (T-014) | {t['official_code_step_ms_retrain']:.0f} | {t['official_code_hours_retrain']:.2f} |",
        f"| **our pipeline**, one run alone on the GPU | {t['ours_step_ms']:.1f} | {t['ours_hours']:.2f} |",
    ]
    write_block("TABLE2", "\n".join(lines))
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["run", "report"])
    args = parser.parse_args(argv)
    run() if args.stage == "run" else report()


if __name__ == "__main__":
    main(sys.argv[1:])
