"""Report of the full Paint Transformer training runs (T-014): evaluation, archive, table, figure.

Three stages, all from the command line (repo root)::

    # 1. after the runs (needs the GPU and the checkpoints in outputs/pt_full/seed*/latest.pt):
    python -m scripts.pt_training_report evaluate --runs outputs/pt_full
    #    -> final_eval.json: every seed's final checkpoint and the official pretrained weights
    #       evaluated on one large held-out synthetic set (the same patches for all models)
    # 2. copy the small logs into the repository (val.csv, thinned train.csv, ...):
    python -m scripts.pt_training_report archive --runs outputs/pt_full
    # 3. regenerate the table and the figure from the archived logs (no GPU, no checkpoints):
    python -m scripts.pt_training_report report

``report`` rewrites the block between the ``BEGIN/END GENERATED`` markers of
``docs/experiments/pt_training.md`` and ``docs/media/pt_training_curves.png``.

Plateau criterion: the relative improvement of the mean validation total between the window of
the 10-20 % last steps and the window of the last 10 % is below 1 % (``PLATEAU_TOL``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from typing import Any

ARCHIVE = Path("docs/experiments/data/pt_training")
DOC = Path("docs/experiments/pt_training.md")
FIGURE = Path("docs/media/pt_training_curves.png")
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
PLATEAU_TOL = 0.01  # < 1 % relative improvement over the last 10 % of the steps
THIN_EVERY = 250  # steps between kept rows of the archived train.csv
EVAL_SEED, EVAL_BATCHES = 999, 64  # final evaluation set: 64 x 64 = 4096 patches
TERMS = ("total", "pixel", "gt", "w", "decision", "count_error")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plateau_improvement(steps: list[int], values: list[float]) -> float:
    """Relative improvement of the mean of the last 10 % of the steps over the previous 10 %."""
    last = steps[-1]
    prev = [v for s, v in zip(steps, values, strict=True) if 0.8 * last < s <= 0.9 * last]
    tail = [v for s, v in zip(steps, values, strict=True) if s > 0.9 * last]
    if not prev or not tail:
        raise ValueError("not enough validation points to evaluate the plateau")
    return (statistics.mean(prev) - statistics.mean(tail)) / statistics.mean(prev)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


# ----------------------------------------------------------------------------- stage 1


def evaluate_runs(runs: Path) -> dict[str, Any]:
    """Evaluate each seed's final checkpoint and the official pretrained weights on one set."""
    import torch

    from brushstroke.stroke_model.train import TrainConfig, _Context, evaluate, load_checkpoint
    from brushstroke.utils.config import load_config

    results: dict[str, Any] = {}
    batch_cfg: TrainConfig | None = None
    ctx = None
    for ckpt in sorted(runs.glob("seed*/latest.pt")):
        cfg = load_config(TrainConfig, ckpt.parent / "config.resolved.yaml")
        ctx = _Context(cfg)
        step = load_checkpoint(ctx, ckpt)
        batch_cfg = cfg
        rng = torch.Generator(device=ctx.device).manual_seed(EVAL_SEED)
        data = [ctx.generator.sample(cfg.batch_size, rng) for _ in range(EVAL_BATCHES)]
        results[ckpt.parent.name] = {"step": step, **evaluate(ctx, data)}
    assert ctx is not None and batch_cfg is not None, f"no checkpoint found under {runs}"
    weights = Path("data/raw/pt_official/model.pth")
    if weights.exists():
        state = torch.load(weights, map_location=ctx.device, weights_only=True)
        ctx.net.load_state_dict(state)
        results["official_pretrained"] = {"step": None, **evaluate(ctx, data)}
    out = runs / "final_eval.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return results


# ----------------------------------------------------------------------------- stage 2


def archive_runs(runs: Path) -> None:
    """Copy the small logs of ``runs`` to the repository archive (train.csv thinned)."""
    for seed_dir in sorted(runs.glob("seed*")):
        dest = ARCHIVE / seed_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        rows = read_csv(seed_dir / "train.csv")
        kept = [r for r in rows if int(r["step"]) % THIN_EVERY == 0 or r is rows[-1]]
        with (dest / "train.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(kept)
        for name in ("val.csv", "summary.json", "config.resolved.yaml"):
            shutil.copy(seed_dir / name, dest / name)
    shutil.copy(runs / "final_eval.json", ARCHIVE / "final_eval.json")


# ----------------------------------------------------------------------------- stage 3


def load_archive(root: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for seed_dir in sorted(root.glob("seed*")):
        runs[seed_dir.name] = {
            "train": read_csv(seed_dir / "train.csv"),
            "val": read_csv(seed_dir / "val.csv"),
            "summary": json.loads((seed_dir / "summary.json").read_text(encoding="utf-8")),
        }
    return runs


def build_table(root: Path) -> str:
    """Markdown tables (per seed, mean +- std, comparison with the official weights)."""
    runs = load_archive(root)
    final = json.loads((root / "final_eval.json").read_text(encoding="utf-8"))
    lines = [
        "| seed | steps | time (min) | peak VRAM (MiB) | plateau (rel. improvement, last 10 %) |",
        "|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        val = run["val"]
        steps = [int(r["step"]) for r in val]
        improvement = plateau_improvement(steps, [float(r["total"]) for r in val])
        train = run["train"]
        step_ms = statistics.mean(float(r["step_ms"]) for r in train[1:])
        minutes = step_ms * steps[-1] / 60000
        lines.append(
            f"| {name} | {steps[-1]} | {minutes:.0f} | {run['summary']['peak_vram_mb']:.0f} | "
            f"{100 * improvement:.2f} % {'(met)' if improvement < PLATEAU_TOL else '(NOT met)'} |"
        )
    seeds = [final[k] for k in final if k.startswith("seed")]
    lines += [
        "",
        "Final checkpoints on one held-out synthetic set (4096 patches, identical for all rows):",
        "",
        "| model | total | pixel | L1 | Wasserstein | BCE | count error |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, entry in final.items():
        label = "official pretrained weights" if key == "official_pretrained" else key
        lines.append(f"| {label} | " + " | ".join(f"{entry[t]:.3f}" for t in TERMS) + " |")
    cells = []
    for t in TERMS:
        m, s = mean_std([e[t] for e in seeds])
        cells.append(f"{m:.3f} ± {s:.3f}")
    lines.append("| **ours, mean ± std over seeds** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_figure(root: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_archive(root)
    final = json.loads((root / "final_eval.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), dpi=90)
    for name, run in runs.items():
        val = run["val"]
        steps = [int(r["step"]) for r in val]
        axes[0].plot(steps, [float(r["total"]) for r in val], label=name)
        axes[1].plot(steps, [float(r["total"]) for r in val], label=name)
        axes[2].plot(
            [int(r["step"]) for r in run["train"]],
            [float(r["lr"]) for r in run["train"]],
            label=name,
        )
    axes[0].set_title("validation total (log)", fontsize=9)
    axes[0].set_yscale("log")
    axes[1].set_title("validation total, zoom on the plateau", fontsize=9)
    all_tail = [
        float(r["total"]) for run in runs.values() for r in run["val"][len(run["val"]) // 3 :]
    ]
    axes[1].set_ylim(min(all_tail) * 0.97, max(all_tail) * 1.03)
    if "official_pretrained" in final:
        axes[1].axhline(
            final["official_pretrained"]["total"],
            color="k",
            ls="--",
            lw=1,
            label="official weights",
        )
    axes[2].set_title("learning rate", fontsize=9)
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def write_report(root: Path, doc: Path, figure: Path) -> None:
    table = build_table(root)
    build_figure(root, figure)
    block = f"{BEGIN}\n\n{table}\n\n![training curves](../media/{figure.name})\n\n{END}"
    text = doc.read_text(encoding="utf-8")
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    doc.write_text(text, encoding="utf-8")
    print(table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["evaluate", "archive", "report"])
    parser.add_argument("--runs", type=Path, default=Path("outputs/pt_full"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args(argv)
    if args.stage == "evaluate":
        evaluate_runs(args.runs)
    elif args.stage == "archive":
        archive_runs(args.runs)
    else:
        write_report(args.archive, DOC, FIGURE)


if __name__ == "__main__":
    main()
