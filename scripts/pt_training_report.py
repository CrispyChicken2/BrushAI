"""Report of the full Paint Transformer training runs (T-014): evaluation, archive, table, figure.

Three stages (repo root); a *run set* is a directory ``<runs>/seed*/`` written by
``scripts.train_pt`` and tagged (``official_recipe`` = the 56,320-step recipe, ``long`` = the same
recipe stretched to 112,640 steps)::

    # 1. after the runs (GPU + the checkpoints latest.pt): final evaluation of every seed and of the
    #    official weights on one large held-out synthetic set, written to <runs>/final_eval.json
    python -m scripts.pt_training_report evaluate --runs outputs/pt_full
    # 2. copy the small logs into the repository (val.csv, thinned train.csv, summaries, evaluation)
    python -m scripts.pt_training_report archive --runs outputs/pt_full --tag official_recipe
    # 3. regenerate the tables and figures from the archived logs (no GPU, no checkpoint)
    python -m scripts.pt_training_report report

``report`` rewrites the block between the ``BEGIN/END GENERATED`` markers of
``docs/experiments/pt_training.md`` and ``docs/media/pt_training_<tag>.png``.

Plateau criteria (``PLATEAU_TOL`` = 1 %), all as relative improvement of a *median* (robust to the
eval-mode spikes of the validation curve, see the experiment note):

* **pre-decay**: window [74 %, 83 %) of the steps -> window [83 %, 92 %) (the learning-rate decay
  of the recipe starts at 92.7 %); the loss must be flat *before* the decay is applied. The
  criterion is evaluated on the training loss (25-step means: dense, 240+ points per window); the
  validation gain (points every 1000 steps) is shown for information only, it is too coarse.
* **last 5 %**: windows (95 %, 97.5 %] -> (97.5 %, 100 %] of the validation loss, i.e. the end of
  the decay tail is stable.
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
FIGURES = Path("docs/media")
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
PLATEAU_TOL = 0.01
THIN_EVERY = 250  # steps between kept rows of the archived train.csv
EVAL_SEED, EVAL_BATCHES = 999, 64  # final evaluation set: 64 x 64 = 4096 patches
TERMS = ("total", "pixel", "gt", "w", "decision", "count_error")
TAGS = {
    "official_recipe": "Official recipe (56,320 steps)",
    "long": "Stretched recipe (112,640 steps)",
}
PRE_DECAY = ((0.74, 0.83), (0.83, 0.92))
LAST_5 = ((0.95, 0.975), (0.975, 1.0))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def window_gain(
    steps: list[int],
    values: list[float],
    windows: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    """Relative improvement of the median value from the first to the second window.

    Windows are ``[lo, hi)`` fractions of the last step (a window that would be empty falls back
    on the point at ``hi``).
    """
    last = max(steps)

    def median_in(lo: float, hi: float) -> float:
        picked = [v for s, v in zip(steps, values, strict=True) if lo * last <= s < hi * last]
        picked = picked or [v for s, v in zip(steps, values, strict=True) if s == hi * last]
        if not picked:
            raise ValueError(f"no data in the window [{lo}, {hi}) of the run")
        return statistics.median(picked)

    before, after = median_in(*windows[0]), median_in(*windows[1])
    return (before - after) / before


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


# ----------------------------------------------------------------------------- stage 1


def _score(ctx: Any, data: list[Any]) -> dict[str, float]:
    """Loss terms on ``data`` plus the sign diagnostics of the predicted heights.

    ``h_negative`` = fraction of matched strokes whose predicted ``h`` is negative; ``total_absh`` =
    total loss after replacing ``h`` by ``|h|`` (the Wasserstein term and the PT renderer treat the
    sign of ``h`` as a symmetry, see the experiment note).
    """
    import torch

    from brushstroke.stroke_model.train import evaluate

    scores = evaluate(ctx, data)
    ctx.net.eval()
    negative, total_abs = [], 0.0
    with torch.no_grad():
        for batch in data:
            shape, colour, logits = ctx.forward(batch)
            out = ctx.loss(shape, colour, logits, batch)
            rows, cols = torch.nonzero(batch.gt_decision.bool(), as_tuple=True)
            negative.append((shape[rows, out.assign[rows, cols]][:, 3] < 0).float())
            flipped = shape.clone()
            flipped[..., 3] = flipped[..., 3].abs()
            total_abs += float(ctx.loss(flipped, colour, logits, batch).total) / len(data)
    return {**scores, "h_negative": float(torch.cat(negative).mean()), "total_absh": total_abs}


def evaluate_runs(runs: Path, official_run: Path | None = None) -> dict[str, Any]:
    """Evaluate each seed's final checkpoint, the official pretrained weights and (if present)
    the official code retrained by ``scripts.pt_official_train`` on one common held-out set."""
    import torch

    from brushstroke.stroke_model.train import TrainConfig, _Context, load_checkpoint
    from brushstroke.utils.config import load_config

    results: dict[str, Any] = {}
    ctx = None
    data: list[Any] = []
    for ckpt in sorted(runs.glob("seed*/latest.pt")):
        cfg = load_config(TrainConfig, ckpt.parent / "config.resolved.yaml")
        ctx = _Context(cfg)
        step = load_checkpoint(ctx, ckpt)
        rng = torch.Generator(device=ctx.device).manual_seed(EVAL_SEED)
        data = [ctx.generator.sample(cfg.batch_size, rng) for _ in range(EVAL_BATCHES)]
        results[ckpt.parent.name] = {"step": step, **_score(ctx, data)}
    assert ctx is not None, f"no checkpoint found under {runs}"
    others = {"official_pretrained": Path("data/raw/pt_official/model.pth")}
    if official_run is not None:
        others["official_retrained"] = official_run / "model.pth"
    for key, weights in others.items():
        if weights.exists():
            ctx.net.load_state_dict(torch.load(weights, map_location=ctx.device, weights_only=True))
            results[key] = {"step": None, **_score(ctx, data)}
    (runs / "final_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    return results


# ----------------------------------------------------------------------------- stage 2


def _thin_copy(src: Path, dest: Path) -> None:
    rows = read_csv(src)
    kept = [r for r in rows if int(r["step"]) % THIN_EVERY == 0 or r is rows[-1]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)


def archive_runs(runs: Path, tag: str, official_run: Path | None = None) -> None:
    """Copy the small logs of the run set ``runs`` to ``ARCHIVE/<tag>`` (train.csv thinned)."""
    root = ARCHIVE / tag
    for seed_dir in sorted(runs.glob("seed*")):
        _thin_copy(seed_dir / "train.csv", root / seed_dir.name / "train.csv")
        for name in ("val.csv", "summary.json", "config.resolved.yaml"):
            shutil.copy(seed_dir / name, root / seed_dir.name / name)
    shutil.copy(runs / "final_eval.json", root / "final_eval.json")
    if official_run is not None and (official_run / "train.csv").exists():
        _thin_copy(official_run / "train.csv", ARCHIVE / "official_retrained" / "train.csv")


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


def _column(rows: list[dict[str, str]], key: str) -> tuple[list[int], list[float]]:
    return [int(r["step"]) for r in rows], [float(r[key]) for r in rows]


def build_table(root: Path) -> str:
    """Markdown tables of one run set (per seed, mean +- std, comparison with the official code)."""
    runs = load_archive(root)
    final = json.loads((root / "final_eval.json").read_text(encoding="utf-8"))

    def pct(gain: float, verdict: bool = False) -> str:
        suffix = (" (met)" if gain < PLATEAU_TOL else " (NOT met)") if verdict else ""
        return f"{100 * gain:.2f} %{suffix}"

    lines = [
        "| seed | steps | time (min) | peak VRAM (MiB) | pre-decay gain, train (criterion) | "
        "pre-decay gain, val | last-5 % gain, val |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        v_steps, v_total = _column(run["val"], "total")
        t_steps, t_total = _column(run["train"], "total")
        step_ms = statistics.mean(float(r["step_ms"]) for r in run["train"][1:])
        lines.append(
            f"| {name} | {v_steps[-1]} | {step_ms * v_steps[-1] / 60000:.0f} | "
            f"{run['summary']['peak_vram_mb']:.0f} | "
            f"{pct(window_gain(t_steps, t_total, PRE_DECAY), verdict=True)} | "
            f"{pct(window_gain(v_steps, v_total, PRE_DECAY))} | "
            f"{pct(window_gain(v_steps, v_total, LAST_5))} |"
        )
    seeds = [final[k] for k in final if k.startswith("seed")]
    lines += [
        "",
        "Final checkpoints on one held-out synthetic set (4096 patches, identical for all rows); "
        "`h<0` = fraction of matched strokes with a negative predicted height, `abs(h)` = loss "
        "after replacing `h` by `|h|`:",
        "",
        "| model | total | pixel | L1 | Wasserstein | BCE | count error | h<0 | total, abs(h) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "official_pretrained": "official pretrained weights",
        "official_retrained": "official code retrained here (seed 0)",
    }
    columns = (*TERMS, "h_negative", "total_absh")
    for key, entry in final.items():
        row = " | ".join(f"{entry[c]:.3f}" for c in columns)
        lines.append(f"| {labels.get(key, key)} | {row} |")
    stats: list[str] = []
    for c in columns:
        m, s = mean_std([e[c] for e in seeds])
        stats.append(f"{m:.3f} ± {s:.3f}")
    lines.append("| **ours, mean ± std over seeds** | " + " | ".join(stats) + " |")
    return "\n".join(lines)


def _binned(
    rows: list[dict[str, str]], key: str, width: int = 2000
) -> tuple[list[int], list[float]]:
    bins: dict[int, list[float]] = {}
    for r in rows:
        bins.setdefault(int(r["step"]) // width * width + width // 2, []).append(float(r[key]))
    return list(bins), [statistics.mean(v) for v in bins.values()]


def build_figure(root: Path, out: Path, official_log: Path | None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_archive(root)
    final = json.loads((root / "final_eval.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), dpi=90)
    for name, run in runs.items():
        axes[0].plot(*_column(run["val"], "total"), label=name)
        axes[1].plot(*_column(run["val"], "total"), label=name, lw=0.9)
        axes[2].plot(*_binned(run["train"], "total"), label=name)
    if official_log is not None and official_log.exists():
        axes[2].plot(*_binned(read_csv(official_log), "total"), "k--", label="official code")
    axes[0].set_title("validation total (log scale; eval-mode spikes, see text)", fontsize=8)
    axes[0].set_yscale("log")
    axes[1].set_title("validation total, zoom (dashed: official weights)", fontsize=8)
    lowest = min(float(r["total"]) for run in runs.values() for r in run["val"])
    axes[1].set_ylim(lowest * 0.97, lowest * 1.3)
    if "official_pretrained" in final:
        axes[1].axhline(final["official_pretrained"]["total"], color="k", ls="--", lw=1)
    axes[2].set_title("training total, mean per 2000 steps", fontsize=8)
    axes[2].set_ylim(lowest * 0.9, lowest * 1.3)
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def write_report(archive: Path, doc: Path, figures: Path) -> None:
    parts = []
    for tag, title in TAGS.items():
        root = archive / tag
        if not (root / "final_eval.json").exists():
            continue
        figure = figures / f"pt_training_{tag}.png"
        build_figure(root, figure, archive / "official_retrained" / "train.csv")
        table = build_table(root)
        parts.append(f"### {title}\n\n{table}\n\n![{tag}](../media/{figure.name})")
        print(f"== {title}\n{table}\n")
    block = f"{BEGIN}\n\n" + "\n\n".join(parts) + f"\n\n{END}"
    text = doc.read_text(encoding="utf-8")
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    doc.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["evaluate", "archive", "report"])
    parser.add_argument("--runs", type=Path, default=Path("outputs/pt_full"))
    parser.add_argument("--tag", choices=list(TAGS), default="official_recipe")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--official-run", type=Path, default=Path("outputs/pt_official_train"))
    args = parser.parse_args(argv)
    if args.stage == "evaluate":
        evaluate_runs(args.runs, args.official_run)
    elif args.stage == "archive":
        archive_runs(args.runs, args.tag, args.official_run)
    else:
        write_report(args.archive, DOC, FIGURES)


if __name__ == "__main__":
    main()
