"""Train the Learning to Paint agent (T-029): model-based DDPG with a WGAN-GP reward on the Met CC0 subset.

Usage (repo root)::

    python -m scripts.train_l2p [--config configs/l2p/train.yaml] [key=value ...]

Needs the training subset (``python -m scripts.build_l2p_trainset fetch`` then ``verify``) and the neural renderer
(``python -m scripts.train_l2p_renderer``). Resumable: rerunning continues from ``<output_dir>/last.pt`` (the replay buffer is
refilled for ``warmup`` steps first). Writes ``history.jsonl``, ``actor.pt`` (loadable by ``l2p.actor.load_actor``) and, at the end,
``final_validation.json``. See ``brushstroke/baselines/l2p/ddpg.py`` and ``docs/experiments/l2p_training.md``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brushstroke.baselines.l2p.ddpg import DDPGConfig, train_agent
from brushstroke.utils.config import load_config


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/l2p/train.yaml"))
    parser.add_argument("overrides", nargs="*", help="dot overrides, e.g. seed=1")
    args = parser.parse_args(argv)
    return train_agent(load_config(DDPGConfig, args.config, args.overrides))


if __name__ == "__main__":
    main()
