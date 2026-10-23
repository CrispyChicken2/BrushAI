"""T-009: sweep the optimization baseline and regenerate the results tables.

Usage (repo root)::

    python -m scripts.fetch_data --insecure                  # once: public-domain images -> data/raw
    python -m scripts.exp_optim_sweep --config configs/experiments/optim_sweep.yaml
    python -m scripts.exp_optim_sweep --stage h1_iters       # one stage
    python -m scripts.exp_optim_sweep --worker 0/2           # shard the jobs over two processes
    python -m scripts.exp_optim_sweep --aggregate-only --figures   # tables + figures from stored runs

Every (stage, variant, image, seed) run is stored as ``<out_dir>/runs/<stage>/<variant>/<image>_s<seed>.json``
(resumable: existing runs are skipped unless ``--force``). The tables (mean +- std over the runs of a
variant: images x seeds) go to ``tables_path``. Metrics: L1, PSNR, LPIPS-VGG (also a training-loss term),
LPIPS-AlexNet (held out: never in any loss), runtime, peak VRAM, stroke count.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from brushstroke.baselines.optim.config import PaintOptimConfig
from brushstroke.baselines.optim.init import blur2d
from brushstroke.baselines.optim.losses import lpips_distance
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.utils.config import load_config
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image
from brushstroke.utils.seed import seed_everything

METRICS = (
    "l1",
    "psnr",
    "lpips_vgg",
    "lpips_alex",
    "runtime_seconds",
    "peak_vram_mb",
    "num_strokes",
)


@dataclass
class Schedule:
    levels: list[int]
    strokes_per_patch: list[int]


@dataclass
class Variant:
    name: str
    overrides: list[str] = field(default_factory=list)
    schedule: str | None = None  # overrides the stage schedule


@dataclass
class Stage:
    name: str
    images: list[str]
    seeds: list[int]
    schedule: str
    variants: list[Variant]
    description: str = ""
    delta_reference: str = "base/default"  # 'stage/variant' the paired deltas are computed against


@dataclass
class References:
    blur_sigmas: list[float] = field(default_factory=list)
    resample_factors: list[int] = field(default_factory=list)


@dataclass
class SweepConfig:
    base_config: str = "configs/baseline_optim.yaml"
    images_manifest: str = "configs/experiments/images.yaml"
    image_dir: str = "data/raw"
    out_dir: str = "outputs/exp_optim"
    tables_path: str = "docs/experiments/optim_sweep_tables.md"
    size: int = 512
    schedules: dict[str, Schedule] = field(default_factory=dict)
    references: References = field(default_factory=References)
    stages: list[Stage] = field(default_factory=list)


@dataclass(frozen=True)
class Job:
    stage: str
    variant: str
    image: str
    seed: int
    overrides: tuple[str, ...]

    @property
    def rel_path(self) -> Path:
        return Path("runs") / self.stage / self.variant / f"{self.image}_s{self.seed}.json"


def build_jobs(cfg: SweepConfig, only: list[str] | None = None) -> list[Job]:
    """All runs of the sweep in a deterministic order (stage, variant, image, seed)."""
    jobs: list[Job] = []
    for stage in cfg.stages:
        if only and stage.name not in only:
            continue
        for variant in stage.variants:
            sched = cfg.schedules[variant.schedule or stage.schedule]
            overrides = [
                f"optim.levels={sched.levels}",
                f"optim.strokes_per_patch={sched.strokes_per_patch}",
                *variant.overrides,
            ]
            for image in stage.images:
                for seed in stage.seeds:
                    jobs.append(Job(stage.name, variant.name, image, seed, tuple(overrides)))
    return jobs


def load_target(path: Path, size: int) -> torch.Tensor:
    return center_crop_resize(load_image(path), (size, size))


def image_metrics(final: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """L1, PSNR, LPIPS-VGG and LPIPS-Alex between two ``(3, H, W)`` images on the same device."""
    mse = float((final - target).square().mean())
    with torch.no_grad():
        return {
            "l1": float((final - target).abs().mean()),
            "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "lpips_vgg": float(lpips_distance(final[None], target[None], "vgg")),
            "lpips_alex": float(lpips_distance(final[None], target[None], "alex")),
        }


def run_job(
    job: Job, cfg: SweepConfig, device: torch.device, out_dir: Path, force: bool
) -> dict[str, Any]:
    """Run (or load) one job; returns its stored record."""
    path = out_dir / job.rel_path
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    run_cfg = load_config(
        PaintOptimConfig,
        cfg.base_config,
        [*job.overrides, f"optim.height={cfg.size}", f"optim.width={cfg.size}", f"seed={job.seed}"],
    )
    target = load_target(Path(cfg.image_dir) / f"{job.image}.jpg", cfg.size)
    seed_everything(run_cfg.seed, run_cfg.deterministic)
    seq = paint_optim(
        target, run_cfg.optim, run_cfg.loss, run_cfg.brush, run_cfg.render, run_cfg.seed, device
    )
    final = seq.render(cfg=run_cfg.render).to(device)
    metrics = image_metrics(final, target.to(device))
    metrics |= {
        "runtime_seconds": float(seq.meta["runtime_seconds"]),
        "peak_vram_mb": float(seq.meta["peak_vram_mb"]),
        "num_strokes": float(len(seq)),
    }
    record = {
        "stage": job.stage,
        "variant": job.variant,
        "image": job.image,
        "seed": job.seed,
        "overrides": list(job.overrides),
        "config_hash": seq.meta["config_hash"],
        "metrics": metrics,
        "level_seconds": seq.meta["level_seconds"],
        "level_final_loss": seq.meta["level_final_loss"],
        "loss_curve": seq.meta["loss_curve"],
        "final_pass_curve": seq.meta["final_pass_curve"],
        "levels": list(run_cfg.optim.levels),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record


def run_references(cfg: SweepConfig, device: torch.device, out_dir: Path) -> list[dict[str, Any]]:
    """Reference floors: LPIPS/L1 of blurred and down/up-sampled targets (no strokes involved)."""
    path = out_dir / "references.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    images = sorted({img for st in cfg.stages for img in st.images})
    rows: list[dict[str, Any]] = []
    for image in images:
        target = load_target(Path(cfg.image_dir) / f"{image}.jpg", cfg.size).to(device)
        for sigma in cfg.references.blur_sigmas:
            blurred = blur2d(target.cpu(), sigma).clamp(0, 1).to(device)
            rows.append(
                {"image": image, "reference": f"gaussian_blur_sigma{sigma:g}"}
                | image_metrics(blurred, target)
            )
        for factor in cfg.references.resample_factors:
            small = torch.nn.functional.interpolate(
                target[None], scale_factor=1 / factor, mode="bilinear", antialias=True
            )
            up = torch.nn.functional.interpolate(small, size=target.shape[-2:], mode="bilinear")
            rows.append(
                {"image": image, "reference": f"down_up_x{factor}"}
                | image_metrics(up[0].clamp(0, 1), target)
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Group records by (stage, variant): mean/std of every metric over images x seeds."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["stage"], rec["variant"]), []).append(rec)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, recs in groups.items():
        row: dict[str, Any] = {"n": len(recs), "images": sorted({r["image"] for r in recs})}
        for m in METRICS:
            row[m] = mean_std([r["metrics"][m] for r in recs])
        out[key] = row
    return out


def paired_delta(
    records: list[dict[str, Any]], stage: str, variant: str, reference: str, metric: str
) -> tuple[float, float, int] | None:
    """Mean/std of ``metric(variant) - metric(reference)`` over the (image, seed) pairs both have.

    Returns ``None`` when there is no common pair (or the variant is the reference itself).
    """
    ref_stage, ref_variant = reference.split("/")
    if (stage, variant) == (ref_stage, ref_variant):
        return None
    ref = {
        (r["image"], r["seed"]): r["metrics"][metric]
        for r in records
        if (r["stage"], r["variant"]) == (ref_stage, ref_variant)
    }
    diffs = [
        r["metrics"][metric] - ref[(r["image"], r["seed"])]
        for r in records
        if (r["stage"], r["variant"]) == (stage, variant) and (r["image"], r["seed"]) in ref
    ]
    if not diffs:
        return None
    mean, std = mean_std(diffs)
    return mean, std, len(diffs)


def fmt_delta(d: tuple[float, float, int] | None) -> str:
    return "–" if d is None else f"{d[0]:+.3f}"


def fmt(ms: tuple[float, float], digits: int = 3) -> str:
    mean, std = ms
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def render_tables(
    cfg: SweepConfig, records: list[dict[str, Any]], refs: list[dict[str, Any]]
) -> str:
    """Markdown with one table per stage (+ the reference floors)."""
    agg = aggregate(records)
    lines = [
        "# Optimization-baseline sweep: generated tables",
        "",
        "Generated by `scripts/exp_optim_sweep.py` (do not edit by hand). Every cell is mean ± std over the runs "
        "of the variant (images × seeds, `n`). LPIPS-Alex is the held-out metric (never in a loss); "
        "LPIPS-VGG is part of the default training loss. ΔVGG/ΔAlex are paired differences "
        "(variant − reference, same image and seed; negative = better) against the stage's reference "
        "variant (`base/default` unless stated): more reliable than unpaired means when the images differ a lot.",
        "",
    ]
    for stage in cfg.stages:
        rows = [
            (v.name, agg[(stage.name, v.name)])
            for v in stage.variants
            if (stage.name, v.name) in agg
        ]
        if not rows:
            continue
        lines += [
            f"## {stage.name}: {stage.description}",
            "",
            "| variant | n | strokes | L1 | PSNR (dB) | LPIPS-VGG | LPIPS-Alex | ΔVGG | ΔAlex | time (s) | peak VRAM (GB) |",
            "|---------|---|---------|----|-----------|-----------|------------|------|-------|----------|----------------|",
        ]
        for name, r in rows:
            lines.append(
                f"| {name} | {r['n']} | {r['num_strokes'][0]:.0f} | {fmt(r['l1'], 4)} | {fmt(r['psnr'], 2)} | "
                f"{fmt(r['lpips_vgg'])} | {fmt(r['lpips_alex'])} | "
                f"{fmt_delta(paired_delta(records, stage.name, name, stage.delta_reference, 'lpips_vgg'))} | "
                f"{fmt_delta(paired_delta(records, stage.name, name, stage.delta_reference, 'lpips_alex'))} | "
                f"{fmt(r['runtime_seconds'], 0)} | "
                f"{r['peak_vram_mb'][0] / 1024:.1f} |"
            )
        lines.append("")
        if stage.name.startswith("final"):
            lines += per_image_table(stage, records)
    if refs:
        by_ref: dict[str, list[dict[str, Any]]] = {}
        for r in refs:
            by_ref.setdefault(r["reference"], []).append(r)
        lines += [
            "## references: floors without strokes (blurred / down-up-sampled target vs target)",
            "",
            "| reference | n | L1 | PSNR (dB) | LPIPS-VGG | LPIPS-Alex |",
            "|---|---|---|---|---|---|",
        ]
        for name, rs in by_ref.items():
            cells = [
                fmt(
                    mean_std([x[m] for x in rs]),
                    3 if m.startswith("lpips") else 4 if m == "l1" else 2,
                )
                for m in ("l1", "psnr", "lpips_vgg", "lpips_alex")
            ]
            lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def per_image_table(stage: Stage, records: list[dict[str, Any]]) -> list[str]:
    """Per-image breakdown (mean ± std over seeds) of a confirmation stage."""
    names = [v.name for v in stage.variants]
    lines = [
        f"### {stage.name}: per image (mean ± std over seeds)",
        "",
        "| image | " + " | ".join(f"{n} L1 | {n} LPIPS-VGG | {n} LPIPS-Alex" for n in names) + " |",
        "|---|" + "---|---|---|" * len(names),
    ]
    for image in stage.images:
        cells = []
        for n in names:
            recs = [
                r["metrics"]
                for r in records
                if (r["stage"], r["variant"], r["image"]) == (stage.name, n, image)
            ]
            if not recs:
                cells += ["–", "–", "–"]
                continue
            cells += [
                fmt(mean_std([m[k] for m in recs]), 4 if k == "l1" else 3)
                for k in ("l1", "lpips_vgg", "lpips_alex")
            ]
        lines.append(f"| {image} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def make_figures(cfg: SweepConfig, records: list[dict[str, Any]], out: Path) -> list[Path]:
    """Loss curves of the iteration sweep and LPIPS bars per stage (PNG, small)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)
    # 1. convergence: training loss over the steps of the finest level, per iteration budget
    iters = [r for r in records if r["stage"] in ("h1_iters", "base") and r["image"] == "mona_lisa"]
    if iters:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in sorted(iters, key=lambda r: len(r["loss_curve"][-1])):
            label = r["variant"]
            axes[0].plot(r["loss_curve"][-1], label=label)
            axes[1].plot(r["loss_curve"][-2], label=label)
        axes[0].set_title("finest level (16x16): training loss per step")
        axes[1].set_title("level 12x12: training loss per step")
        for ax in axes:
            ax.set_xlabel("step")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        path = out / "optim_sweep_convergence.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    # 2. LPIPS-Alex (held out) and LPIPS-VGG per variant, all stages
    agg = aggregate(records)
    keys = [k for k in agg if k[0] != "references"]
    if keys:
        fig, ax = plt.subplots(figsize=(11, 0.28 * len(keys) + 1.5))
        names = [f"{s}/{v}" for s, v in keys][::-1]
        ys = range(len(keys))
        alex = [agg[k]["lpips_alex"][0] for k in keys][::-1]
        vgg = [agg[k]["lpips_vgg"][0] for k in keys][::-1]
        ax.barh([y + 0.2 for y in ys], vgg, height=0.4, label="LPIPS-VGG (in loss)")
        ax.barh([y - 0.2 for y in ys], alex, height=0.4, label="LPIPS-Alex (held out)")
        ax.set_yticks(list(ys))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("LPIPS (lower is better)")
        ax.legend(fontsize=7)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        path = out / "optim_sweep_lpips.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(path)
    return written


def load_records(
    cfg: SweepConfig, out_dir: Path, stages: list[str] | None = None
) -> list[dict[str, Any]]:
    recs = []
    for job in build_jobs(cfg, stages):
        path = out_dir / job.rel_path
        if path.exists():
            recs.append(json.loads(path.read_text(encoding="utf-8")))
    return recs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/optim_sweep.yaml"))
    parser.add_argument(
        "--stage", action="append", help="run/aggregate only this stage (repeatable)"
    )
    parser.add_argument(
        "--worker", default="0/1", help="K/N: process the jobs with index %% N == K"
    )
    parser.add_argument("--force", action="store_true", help="recompute existing runs")
    parser.add_argument("--aggregate-only", action="store_true", help="do not run anything")
    parser.add_argument("--figures", action="store_true", help="also write the figures")
    parser.add_argument("--list", action="store_true", help="list the jobs and exit")
    args = parser.parse_args(argv)

    cfg = load_config(SweepConfig, args.config)
    out_dir = Path(cfg.out_dir)
    jobs = build_jobs(cfg, args.stage)
    k, n = (int(x) for x in args.worker.split("/"))
    if args.list:
        for job in jobs:
            print(job.stage, job.variant, job.image, job.seed, job.overrides)
        print(f"{len(jobs)} jobs")
        return
    device = get_device("auto")
    if not args.aggregate_only:
        mine = [j for i, j in enumerate(jobs) if i % n == k]
        t0 = time.perf_counter()
        for i, job in enumerate(mine):
            existed = (out_dir / job.rel_path).exists() and not args.force
            rec = run_job(job, cfg, device, out_dir, args.force)
            m = rec["metrics"]
            print(
                f"[{i + 1}/{len(mine)}] {'cached ' if existed else ''}{job.stage}/{job.variant} {job.image} "
                f"s{job.seed}: L1 {m['l1']:.4f} LPIPS-VGG {m['lpips_vgg']:.3f} "
                f"Alex {m['lpips_alex']:.3f} {m['runtime_seconds']:.0f}s",
                flush=True,
            )
        print(f"worker {k}/{n} done in {time.perf_counter() - t0:.0f}s")
    if n == 1 or args.aggregate_only:
        refs = run_references(cfg, device, out_dir)
        records = load_records(cfg, out_dir, args.stage)
        tables = render_tables(cfg, records, refs)
        Path(cfg.tables_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.tables_path).write_text(tables, encoding="utf-8")
        print(f"wrote {cfg.tables_path}")
        if args.figures:
            for p in make_figures(cfg, records, Path(cfg.tables_path).parent):
                print(f"wrote {p}")


if __name__ == "__main__":
    main(sys.argv[1:])
