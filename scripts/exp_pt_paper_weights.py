"""Loss ablations of Paint Transformer with the paper's loss weights and matching cost (T-021).

Protocol, hypothesis and decision rules (written before the runs): ``docs/experiments/pt_reproduction.md``,
section T-021.

    python -m scripts.exp_pt_paper_weights train      # full model + 4 ablations x 3 seeds (15 runs, official recipe)
    python -m scripts.exp_pt_paper_weights evaluate   # T-016 protocol (strokes, real images) + invariant D_L1 of the 15 runs
    python -m scripts.exp_pt_paper_weights report     # tables and verdicts -> pt_reproduction.md (block PAPER_WEIGHTS)

``train`` resumes interrupted runs (``resume=auto``); ``evaluate`` skips what is already stored.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time

from scripts.pt_repro_common import PW_GROUPS, PW_ROOT

SEEDS = (0, 1, 2)
# the weights of the paper (lambda_r = 8, BCE 1, pixel 1, L1 1, W 10) and its BCE-based matching
# cost (``paper_weights`` preset of T-012); `custom` so that a removed term can be set to 0
PAPER_WEIGHTS = [
    "loss.preset=custom",
    "loss.lambda_pixel=1",
    "loss.lambda_gt=1",
    "loss.lambda_w=10",
    "loss.lambda_decision=1",
    "loss.pos_weight=8",
    "loss.matching_cost=bce",
]
REMOVE = {  # run name -> override that removes the term
    "full": None,
    "no_pixel": "loss.lambda_pixel=0",
    "no_dl1": "loss.lambda_gt=0",
    "no_dw": "loss.lambda_w=0",
    "no_dbce": "loss.lambda_decision=0",
}


def command(name: str, seed: int) -> list[str]:
    """The training command of a run (the removed term, if any, overrides the paper's weight)."""
    removal = REMOVE[name]
    return [
        sys.executable,
        "-m",
        "scripts.train_pt",
        f"seed={seed}",
        *PAPER_WEIGHTS,
        *([removal] if removal else []),
        f"output_dir={(PW_ROOT / name / f'seed{seed}').as_posix()}",
        "resume=auto",
    ]


def train(names: list[str], parallel: int) -> None:
    jobs = list(itertools.product(names, SEEDS))
    running: list[tuple[subprocess.Popen[bytes], tuple[str, int]]] = []
    while jobs or running:
        running = [(p, j) for p, j in running if p.poll() is None]
        while jobs and len(running) < parallel:
            name, seed = jobs.pop(0)
            out = PW_ROOT / name / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            log = (out / "stdout.log").open("ab")
            running.append(
                (
                    subprocess.Popen(command(name, seed), stdout=log, stderr=subprocess.STDOUT),
                    (name, seed),
                )
            )
            print(f"started {name} seed {seed}", flush=True)
        time.sleep(30)
    print("all paper-weights runs finished", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["train", "evaluate", "report"])
    parser.add_argument("--names", nargs="*", default=list(PW_GROUPS.values()), choices=list(REMOVE))
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args(argv)
    if args.stage == "train":
        train(args.names, args.parallel)
    else:
        raise SystemExit(f"stage {args.stage!r} is added with the evaluation code")


if __name__ == "__main__":
    main(sys.argv[1:])
