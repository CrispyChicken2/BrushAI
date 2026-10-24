"""Train the loss ablations of the Paint Transformer paper (Table 1, lower part), T-016.

Four settings (one loss term removed) x 3 seeds, official recipe (56,320 steps), outputs in
``outputs/pt_abl/<name>/seed<k>``. The ``--parallel`` runs share the GPU (about 88 ms/step each
with 3). The runs resume from their checkpoint (``resume=auto``): after an interruption just start
the command again.

    python -m scripts.exp_pt_ablations [--parallel 3] [--names no_pixel no_dl1 ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

TERMS = {  # name -> loss override that removes the term
    "no_pixel": "loss.lambda_pixel=0",
    "no_dl1": "loss.lambda_gt=0",
    "no_dw": "loss.lambda_w=0",
    "no_dbce": "loss.lambda_decision=0",
}
SEEDS = (0, 1, 2)


def command(name: str, seed: int, root: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.train_pt",
        f"seed={seed}",
        "loss.preset=custom",  # keep the fields as given (the defaults are the official weights)
        TERMS[name],
        f"output_dir={root / name / f'seed{seed}'}",
        "resume=auto",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--names", nargs="*", default=list(TERMS), choices=list(TERMS))
    parser.add_argument("--root", type=Path, default=Path("outputs/pt_abl"))
    args = parser.parse_args(argv)
    jobs = [(n, s) for n in args.names for s in SEEDS]
    running: list[tuple[subprocess.Popen[bytes], tuple[str, int]]] = []
    while jobs or running:
        running = [(p, j) for p, j in running if p.poll() is None]
        while jobs and len(running) < args.parallel:
            name, seed = jobs.pop(0)
            out = args.root / name / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            log = (out / "stdout.log").open("ab")
            running.append(
                (
                    subprocess.Popen(
                        command(name, seed, args.root), stdout=log, stderr=subprocess.STDOUT
                    ),
                    (name, seed),
                )
            )
            print(f"started {name} seed {seed}", flush=True)
        time.sleep(30)
    print("all ablation runs finished")


if __name__ == "__main__":
    main(sys.argv[1:])
