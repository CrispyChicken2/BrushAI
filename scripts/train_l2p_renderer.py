"""Train the Learning to Paint neural stroke renderer on synthetic strokes (T-028).

Usage (repo root)::

    python -m scripts.train_l2p_renderer [--config configs/l2p/renderer.yaml] [key=value ...]

Resumable: rerunning continues from ``<output_dir>/last.pt``. Writes ``history.jsonl`` (train and
validation error every ``log_every`` steps) and ``renderer.pt`` at the end.
See ``docs/design/learning_to_paint.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brushstroke.baselines.l2p.renderer import RendererTrainConfig, train_renderer
from brushstroke.utils.config import load_config


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/l2p/renderer.yaml"))
    parser.add_argument("overrides", nargs="*", help="dot overrides, e.g. steps=1000")
    args = parser.parse_args(argv)
    cfg = load_config(RendererTrainConfig, args.config, args.overrides)
    return train_renderer(cfg)


if __name__ == "__main__":
    main()
