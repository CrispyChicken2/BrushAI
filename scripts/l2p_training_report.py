"""Report of the Learning to Paint agent training (T-029), regenerated from the stored logs.

Usage (repo root)::

    python -m scripts.l2p_training_report --runs outputs/l2p_train/seed0 outputs/l2p_train/seed1 outputs/l2p_train/seed2 \\
        [--cost outputs/l2p_train/cost_fp32 outputs/l2p_train/cost_bf16] \\
        [--released data/raw/l2p_official/actor.pkl --renderer outputs/l2p_renderer/renderer.pt]

Reads ``history.jsonl`` (and ``final_validation.json``) of every run directory and writes

* ``docs/experiments/l2p_training_tables.md``: cost, final quality per training seed, optional comparison with the released actor
  on the held-out Met validation images (128 px, 40 steps), and the cost-measurement runs;
* ``docs/experiments/l2p_training.png``: training l2, validation l2, critic and discriminator curves;
* ``docs/experiments/data/l2p/training_summary.json`` and a copy of each history (``history_<run name>.jsonl``).

Nothing is hand-edited: the narrative ``docs/experiments/l2p_training.md`` only links to these files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

OUT_MD = Path("docs/experiments/l2p_training_tables.md")
OUT_PNG = Path("docs/experiments/l2p_training.png")
DATA = Path("docs/experiments/data/l2p")
FULL_EPISODES = 50_000  # the official 2,000,000 environment steps of 40


def read_history(run: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (run / "history.jsonl").read_text().splitlines() if x.strip()]


def steady_seconds_per_episode(rows: list[dict[str, Any]], skip: int = 20) -> float:
    """Seconds per episode over the rows after the first ``skip`` episodes (warm-up and cudnn autotuning excluded)."""
    rows = [r for r in rows if r["episode"] >= skip]
    if len(rows) < 2:
        return float("nan")
    return (rows[-1]["seconds"] - rows[0]["seconds"]) / (rows[-1]["episode"] - rows[0]["episode"])


def run_summary(run: Path) -> dict[str, Any]:
    rows = read_history(run)
    last = rows[-1]
    final: dict[str, Any] = {}
    if (run / "final_validation.json").exists():
        final = json.loads((run / "final_validation.json").read_text())
    tail = [r["train_l2"] for r in rows[-20:]]
    val = [r for r in rows if "val_l2" in r]
    sec_ep = steady_seconds_per_episode(rows)
    return {
        "run": run.name,
        "episodes": last["episode"],
        "hours": last["seconds"] / 3600,
        "seconds_per_episode": sec_ep,
        "hours_for_full_run": sec_ep * FULL_EPISODES / 3600,
        "peak_vram_gb": max((r.get("peak_vram_gb", 0.0) for r in rows), default=0.0),
        "train_l2_last20_logs": float(np.mean(tail)),
        "val_l2_last": val[-1]["val_l2"] if val else None,
        "val_l2_best": min((r["val_l2"] for r in val), default=None),
        "val_l2_final_2000": final.get("val_l2"),
        "val_l2_blank": final.get("val_l2_blank") or (val[-1]["val_l2_blank"] if val else None),
        "final_q": last.get("q"),
        "final_value_loss": last.get("value_loss"),
    }


def released_comparison(
    runs: list[Path], released: Path, renderer: Path, trainset: Path
) -> dict[str, Any]:
    """l2 of the released actor and of every run's actor on the held-out Met validation images (128 px, 40 steps, BGR)."""
    import torch

    from brushstroke.baselines.l2p.actor import load_actor
    from brushstroke.baselines.l2p.ddpg import evaluate_actor
    from brushstroke.baselines.l2p.renderer import load_renderer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = np.load(trainset / "images.npy", mmap_mode="r")
    val_ids = np.flatnonzero(np.load(trainset / "is_val.npy"))
    net = load_renderer(renderer, dev)
    out = {"released": evaluate_actor(load_actor(released, dev), net, images, val_ids)}
    for run in runs:
        if (run / "actor.pt").exists():
            out[run.name] = evaluate_actor(load_actor(run / "actor.pt", dev), net, images, val_ids)
    out["n_images"] = {"value": float(len(val_ids))}
    return out


def figure(runs: list[Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    panels = [
        ("train_l2", "training l2 (final canvas, 96 images)", True),
        ("val_l2", "validation l2 (held-out Met images)", True),
        ("q", "mean Q + r of the policy batch", False),
        ("value_loss", "critic loss", True),
        ("gan_penalty", "WGAN-GP gradient penalty", False),
        ("reward", "mean step reward (discriminator)", False),
    ]
    for a, (key, title, log) in zip(ax.flat, panels, strict=True):
        for run in runs:
            rows = [r for r in read_history(run) if key in r]
            a.plot([r["episode"] for r in rows], [r[key] for r in rows], label=run.name, lw=1)
        a.set_title(title, fontsize=10)
        a.set_xlabel("episode (40 steps x 96 paintings)")
        if log:
            a.set_yscale("log")
    ax[0, 0].legend(fontsize=8)
    fig.savefig(OUT_PNG, dpi=110)
    plt.close(fig)


def fmt(v: Any, spec: str = ".4g") -> str:
    return "-" if v is None else format(v, spec)


def tables(
    summaries: list[dict[str, Any]], cost: list[dict[str, Any]], released: dict[str, Any] | None
) -> str:
    lines = [
        "# Learning to Paint training: tables (generated by `scripts/l2p_training_report.py`)",
        "",
    ]
    lines += [
        "## Training runs",
        "",
        "| run | episodes | hours | s / episode | full run (h) | peak VRAM (GB) | train l2 (last 20 logs) | val l2 (480 imgs, last) | val l2 (2000 imgs, final) | blank l2 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['run']} | {s['episodes']} | {s['hours']:.2f} | {fmt(s['seconds_per_episode'], '.3f')} | {fmt(s['hours_for_full_run'], '.1f')} | {s['peak_vram_gb']:.2f} | "
            f"{fmt(s['train_l2_last20_logs'])} | {fmt(s['val_l2_last'])} | {fmt(s['val_l2_final_2000'])} | {fmt(s['val_l2_blank'])} |"
        )
    if cost:
        lines += [
            "",
            "## Cost measurements (short runs on the real subset)",
            "",
            "| run | episodes | s / episode | full run (h) | peak VRAM (GB) | train l2 at the end |",
            "|---|---|---|---|---|---|",
        ]
        for s in cost:
            lines.append(
                f"| {s['run']} | {s['episodes']} | {fmt(s['seconds_per_episode'], '.3f')} | {fmt(s['hours_for_full_run'], '.1f')} | {s['peak_vram_gb']:.2f} | {fmt(s['train_l2_last20_logs'])} |"
            )
    if released:
        n = int(released["n_images"]["value"])
        lines += [
            "",
            f"## Released actor against ours (held-out Met validation images, n = {n}, 128 px, {40} steps of 5 strokes, our renderer)",
            "",
            "| actor | l2 after 1 step | 5 steps | 20 steps | 40 steps | blank |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in released.items():
            if name == "n_images":
                continue
            lines.append(
                f"| {name} | {v['l2_step1']:.4f} | {v['l2_step5']:.4f} | {v['l2_step20']:.4f} | {v['l2_step40']:.5f} | {v['l2_blank']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--cost", type=Path, nargs="*", default=[])
    ap.add_argument("--released", type=Path, default=None)
    ap.add_argument("--renderer", type=Path, default=Path("outputs/l2p_renderer/renderer.pt"))
    ap.add_argument("--trainset", type=Path, default=Path("data/raw/l2p_met"))
    args = ap.parse_args(argv)
    runs = [r for r in args.runs if (r / "history.jsonl").exists()]
    summaries = [run_summary(r) for r in runs]
    cost = [run_summary(r) for r in args.cost if (r / "history.jsonl").exists()]
    released = None
    if args.released is not None and args.released.exists():
        released = released_comparison(runs, args.released, args.renderer, args.trainset)
    DATA.mkdir(parents=True, exist_ok=True)
    for r in runs + [c for c in args.cost if (c / "history.jsonl").exists()]:
        shutil.copyfile(r / "history.jsonl", DATA / f"history_{r.name}.jsonl")
    OUT_MD.write_text(tables(summaries, cost, released), encoding="utf-8")
    (DATA / "training_summary.json").write_text(
        json.dumps({"runs": summaries, "cost": cost, "released_comparison": released}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    if runs:
        figure(runs)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
