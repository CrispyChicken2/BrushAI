"""Parameter-convention experiment of Paint Transformer (T-019).

Protocol and verdict rules: ``docs/experiments/pt_training.md`` (section "T-019"); derivation of the
equivalence classes: ``docs/design/pt_convention.md``.

    python -m scripts.exp_pt_convention symmetry   # render-verify the equivalence classes (GPU or CPU)
    python -m scripts.exp_pt_convention train      # the options A, B, C x 3 seeds, official recipe
    python -m scripts.exp_pt_convention train --names best --recipe long   # stretched recipe, best option
    python -m scripts.exp_pt_convention evaluate   # every checkpoint with the invariant metrics
    python -m scripts.exp_pt_convention report     # tables + verdicts -> pt_training.md, pt_reproduction.md

``train`` resumes interrupted runs (``resume=auto``); ``evaluate`` skips the checkpoints already stored.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from brushstroke.renderer.pt import load_pt_brushes, pt_param2stroke
from brushstroke.stroke_model.convention import variant_specs, variants
from scripts import pt_convention_report

SEEDS = (0, 1, 2)
OPTIONS = {  # name -> config overrides of scripts.train_pt
    "optA_positive": ["model.convention=positive"],
    "optB_invariant_l1": ["loss.l1_invariance=mirror"],
    "optC_sign_penalty": ["loss.lambda_sign=10"],
}
RECIPES = {  # name -> (root, overrides)
    "official": (Path("outputs/pt_conv"), []),
    "long": (Path("outputs/pt_conv_long"), ["schedule.n_epochs=400", "schedule.n_epochs_decay=40"]),
}
SYMMETRY_STROKES, SYMMETRY_SIZE, SYMMETRY_SEED = 4000, 64, 0
DATA = Path("docs/experiments/data/pt_convention")


def dump(name: str, payload: object) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8", newline="\n"
    )


# ----------------------------------------------------------------------------- symmetry


def symmetry() -> None:
    """Rendering error of every parameter variant against the stroke itself (alpha maps)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(SYMMETRY_SEED)
    u = torch.rand(SYMMETRY_STROKES, 5, generator=generator, device=device)
    u[:, :4] = 0.2 + 0.5 * u[:, :4]  # the ground-truth distribution of the synthetic strokes
    out: dict[str, object] = {"strokes": SYMMETRY_STROKES, "size": SYMMETRY_SIZE}
    for variant, mode in (("train", "pt_train"), ("inference", "pt_infer")):
        brushes = load_pt_brushes(variant, "data/raw/pt_brush", device)

        def render(shape: torch.Tensor, b: torch.Tensor = brushes, m: str = mode) -> tuple:
            param8 = torch.cat([shape, torch.ones(len(shape), 3, device=device)], dim=1)
            fg, alpha = pt_param2stroke(param8, SYMMETRY_SIZE, SYMMETRY_SIZE, b, m)
            return alpha[:, 0], fg[:, 0]

        base_alpha, base_fg = render(u)
        rows = []
        for spec, v in zip(variant_specs("swap"), variants(u, "swap"), strict=True):
            alpha, fg = render(v)
            rows.append(
                {
                    "spec": list(spec),
                    "alpha_error": float((alpha - base_alpha).abs().mean()),
                    "alpha_error_relative": float(
                        (alpha - base_alpha).abs().mean() / base_alpha.mean()
                    ),
                    "foreground_error": float((fg - base_fg).abs().mean()),
                }
            )
        out[mode] = {"area": float(base_alpha.mean()), "variants": rows}
    # unrelated strokes of the same distribution, the reference scale of the errors
    other = torch.rand(SYMMETRY_STROKES, 5, generator=generator, device=device)
    other[:, :4] = 0.2 + 0.5 * other[:, :4]
    brushes = load_pt_brushes("train", "data/raw/pt_brush", device)
    a0 = pt_param2stroke(
        torch.cat([u, torch.ones(len(u), 3, device=device)], 1),
        SYMMETRY_SIZE,
        SYMMETRY_SIZE,
        brushes,
        "pt_train",
    )[1][:, 0]
    a1 = pt_param2stroke(
        torch.cat([other, torch.ones(len(u), 3, device=device)], 1),
        SYMMETRY_SIZE,
        SYMMETRY_SIZE,
        brushes,
        "pt_train",
    )[1][:, 0]
    out["unrelated_alpha_error"] = float((a0 - a1).abs().mean())
    dump("symmetry", out)
    print(out["pt_train"])  # type: ignore[index]


# ----------------------------------------------------------------------------- training


def command(option: str, seed: int, recipe: str) -> list[str]:
    root, extra = RECIPES[recipe]
    return [
        sys.executable,
        "-m",
        "scripts.train_pt",
        f"seed={seed}",
        *OPTIONS[option],
        *extra,
        f"output_dir={root / option / f'seed{seed}'}",
        "resume=auto",
    ]


def train(names: list[str], recipe: str, parallel: int) -> None:
    root, _ = RECIPES[recipe]
    jobs = list(itertools.product(names, SEEDS))
    running: list[tuple[subprocess.Popen[bytes], tuple[str, int]]] = []
    while jobs or running:
        running = [(p, j) for p, j in running if p.poll() is None]
        while jobs and len(running) < parallel:
            option, seed = jobs.pop(0)
            out = root / option / f"seed{seed}"
            out.mkdir(parents=True, exist_ok=True)
            log = (out / "stdout.log").open("ab")
            running.append(
                (
                    subprocess.Popen(
                        command(option, seed, recipe), stdout=log, stderr=subprocess.STDOUT
                    ),
                    (option, seed),
                )
            )
            print(f"started {option} seed {seed} ({recipe})", flush=True)
        time.sleep(30)
    print("all convention runs finished", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["symmetry", "train", "evaluate", "report"])
    parser.add_argument(
        "--names",
        nargs="*",
        default=list(OPTIONS),
        choices=[*OPTIONS, "best"],
        help="options; `best` = the option selected by `evaluate`",
    )
    parser.add_argument("--recipe", choices=list(RECIPES), default="official")
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args(argv)
    if args.stage == "symmetry":
        symmetry()
    elif args.stage == "evaluate":
        pt_convention_report.evaluate()
    elif args.stage == "report":
        pt_convention_report.report()
    else:
        names = args.names
        if "best" in names:
            best = pt_convention_report.best_option(pt_convention_report.load_store())
            if best is None:
                raise SystemExit("no finished option run: run `train` then `evaluate` first")
            names = [best if n == "best" else n for n in names]
        train(names, args.recipe, args.parallel)


if __name__ == "__main__":
    main(sys.argv[1:])
