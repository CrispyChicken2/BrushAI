"""Train the Paint Transformer stroke predictor on on-the-fly synthetic data.

Usage (repo root)::

    python -m scripts.train_pt --config configs/stroke_model/pt.yaml [key.sub=value ...]
    python -m scripts.train_pt --profile precision=bf16       # per-phase step times, no training

Examples of overrides: ``schedule.max_steps=500 val.every=100 resume=auto precision=bf16
loss.preset=paper_weights``. Outputs (``output_dir``, never committed): ``train.csv``, ``val.csv``,
``latest.pt``, ``summary.json``, ``config.resolved.yaml``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brushstroke.stroke_model.train import TrainConfig, profile_step, train
from brushstroke.utils.config import load_config


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stroke_model/pt.yaml"))
    parser.add_argument("--profile", action="store_true", help="time the phases of a step and exit")
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("overrides", nargs="*", help="dot overrides, e.g. schedule.max_steps=500")
    args = parser.parse_args(argv)
    cfg = load_config(TrainConfig, args.config, args.overrides)
    if args.profile:
        result: dict[str, object] = dict(profile_step(cfg, steps=args.profile_steps))
    else:
        result = train(cfg)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
