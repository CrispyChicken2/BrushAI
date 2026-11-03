"""Tables and figure of the evaluation harness (T-026): ``docs/design/evaluation.md`` sections 2 to 6.

Reads the records of a run (:mod:`brushstroke.eval.runner`) and builds, as markdown:

* ``B1`` final metrics at the four budgets (mean +/- std over images, 95 % bootstrap interval of the mean, replicate noise;
  a dagger marks the metrics that are in a method's own loss), ``controls`` (blank, blur, sequence render);
* ``B2`` anytime AUC at the four budgets (+ log-spaced and late variants), ``B3`` saliency-weighted LPIPS (evaluator, control,
  lambda sensitivity, AUC), ``B4`` strokes-to-target with right-censoring, ``B5`` runtime and VRAM;
* the paired statistics: the confirmatory family F1 with Holm correction, and exploratory comparisons (uncorrected, labelled);
* the anytime-curve figure and its companion, the fraction of images that reach the target as a function of the stroke count.

The unit of analysis is the image: a per-image value is the mean over the replicates of the method. A method that cannot reach a
budget (fewer strokes) is shown as ``n/a (max n = ...)`` and is excluded from the comparisons at that budget.
"""

from __future__ import annotations

import itertools
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from brushstroke.eval import curves, stats
from brushstroke.eval.protocol import Protocol

Extract = Callable[[dict[str, Any]], float | None]

METRIC_LABEL = {
    "lpips_alex": "LPIPS-Alex",
    "lpips_vgg": "LPIPS-VGG",
    "l1": "L1",
    "ssim": "SSIM",
    "psnr": "PSNR (dB)",
    "lpips_alex_128": "LPIPS-Alex at 128 px",
}


def load_records(out: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((out / "records").glob("*/*.json"))
    ]


def load_controls(out: Path) -> dict[str, dict[str, Any]]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((out / "controls").glob("*.json"))
    }


class Results:
    """The records of a run, indexed by method and image, with the per-image values the tables need."""

    def __init__(
        self, records: list[dict[str, Any]], controls: dict[str, dict[str, Any]], proto: Protocol
    ) -> None:
        self.proto = proto
        self.controls = controls
        self.records = records
        self.ok: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        self.failed: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            if r["status"] == "ok":
                self.ok[r["method"]][r["image_id"]].append(r)
            else:
                self.failed[r["method"]].append(r)
        self.methods = sorted(self.ok)

    def images(self, method: str) -> list[str]:
        return sorted(self.ok[method])

    def common_images(self, a: str, b: str) -> list[str]:
        return sorted(set(self.ok[a]) & set(self.ok[b]))

    # -- values of one record ----------------------------------------------------------------------------------------

    def at_budget(self, rec: dict[str, Any], key: str, budget: int) -> float | None:
        if budget not in rec["checkpoints"] or key not in rec["metrics"]:
            return None
        return float(rec["metrics"][key][rec["checkpoints"].index(budget)])

    def auc(self, rec: dict[str, Any], key: str, budget: int, kind: str = "norm") -> float | None:
        entry = rec["auc"].get(str(budget), {}).get(key)
        return None if entry is None or entry.get(kind) is None else float(entry[kind])

    def strokes(self, rec: dict[str, Any], tau: float) -> curves.StrokesToTarget:
        return curves.strokes_to_target(
            rec["checkpoints"], rec["metrics"]["lpips_alex"], tau, self.proto.max_stroke_count
        )

    # -- per image over replicates -----------------------------------------------------------------------------------

    def per_image(
        self, method: str, extract: Callable[[dict[str, Any]], float | None]
    ) -> dict[str, tuple[float, float]]:
        """``{image: (mean over replicates, std over replicates)}``; an image counts only if every replicate has a value."""
        out: dict[str, tuple[float, float]] = {}
        for image, recs in self.ok[method].items():
            values = [extract(r) for r in recs]
            if all(v is not None for v in values):
                out[image] = stats.per_image_summary([float(v) for v in values if v is not None])
        return out

    def metric(self, key: str, budget: int) -> Extract:
        """A per-record extractor of ``key`` at ``budget`` strokes."""
        return lambda rec: self.at_budget(rec, key, budget)

    def auc_of(self, key: str, budget: int, kind: str = "norm") -> Extract:
        """A per-record extractor of the AUC of ``key`` (``norm``, ``raw``, ``log`` or ``late``) at ``budget``."""
        return lambda rec: self.auc(rec, key, budget, kind)

    def is_warmup(self, rec: dict[str, Any]) -> bool:
        """True for the warm-up image of a (method, seed): flagged by the runner or, for records written before the flag
        existed, the first image in sorted id order (the order a run processes them in)."""
        if "warmup" in rec:
            return bool(rec["warmup"])
        ids = [
            i
            for i, rs in self.ok[rec["method"]].items()
            if any(r["seed"] == rec["seed"] for r in rs)
        ]
        return bool(rec["image_id"] == min(ids))

    def max_n(self, method: str) -> int:
        return min((r["n_total"] for rs in self.ok[method].values() for r in rs), default=0)

    def reachable(self, method: str, budget: int) -> bool:
        return all(
            curves.reaches(r["checkpoints"], budget) for rs in self.ok[method].values() for r in rs
        )


def describe_cell(values: dict[str, tuple[float, float]], proto: Protocol, digits: int = 3) -> str:
    """``mean +/- std [95 % CI] (noise)`` of per-image values."""
    if not values:
        return "n/a"
    means = [m for m, _ in values.values()]
    d = stats.describe(
        means,
        [s for _, s in values.values()],
        proto.bootstrap["resamples"],
        proto.bootstrap["seed"],
        proto.bootstrap["confidence"],
    )
    cell = (
        f"{d.mean:.{digits}f} ± {d.std:.{digits}f} [{d.ci_low:.{digits}f}, {d.ci_high:.{digits}f}]"
    )
    return cell + (f" ({d.replicate_noise:.{digits}f})" if d.replicate_noise else "")


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def na(res: Results, method: str, budget: int) -> str:
    return f"n/a (max n = {res.max_n(method)})"


# ------------------------------------------------------------------------------------------------------------- tables


def table_b1(res: Results) -> str:
    """Final metrics at the budgets."""
    rows = []
    for method in res.methods:
        dagger = set(res.ok[method][res.images(method)[0]][0]["dagger_metrics"])
        for key, label in METRIC_LABEL.items():
            cells = []
            for budget in res.proto.budgets:
                if not res.reachable(method, budget):
                    cells.append(na(res, method, budget))
                    continue
                cells.append(
                    describe_cell(
                        res.per_image(method, res.metric(key, budget)),
                        res.proto,
                    )
                )
            rows.append([method, label + (" †" if key in dagger else ""), *cells])
    return table(["method", "metric", *[f"n = {b}" for b in res.proto.budgets]], rows)


def table_controls(res: Results) -> str:
    """Reference canvases of section 2.2 and the sequence-render control."""
    rows = []
    ids = sorted({r["image_id"] for r in res.records})
    for name in ("blank_black", "blank_white", "blur"):
        cells = []
        for key in ("lpips_alex", "lpips_vgg", "l1", "ssim"):
            vals = {
                i: (res.controls[i]["controls"][name][key], 0.0) for i in ids if i in res.controls
            }
            cells.append(describe_cell(vals, res.proto))
        rows.append([name, *cells])
    for method in res.methods:
        diffs = {
            i: (rs[0]["control"].get("pipeline_vs_sequence_mean_abs"), 0.0)
            for i, rs in res.ok[method].items()
            if rs[0].get("control")
        }
        diffs = {i: (float(v[0]), 0.0) for i, v in diffs.items() if v[0] is not None}
        if diffs:
            rows.append(
                [
                    f"{method}: pipeline canvas vs sequence render (mean abs diff)",
                    describe_cell(diffs, res.proto, 4),
                    "",
                    "",
                    "",
                ]
            )
        bg = sorted({tuple(r["background"]) for rs in res.ok[method].values() for r in rs})
        rows.append(
            [
                f"{method}: native blank canvas",
                ", ".join(str(tuple(round(c, 2) for c in b)) for b in bg),
                "",
                "",
                "",
            ]
        )
    return table(["reference", "LPIPS-Alex", "LPIPS-VGG", "L1", "SSIM"], rows)


def table_b2(res: Results) -> str:
    """Anytime AUC."""
    rows = []
    for method in res.methods:
        cells = []
        for budget in res.proto.budgets:
            if not res.reachable(method, budget):
                cells.append(na(res, method, budget))
            else:
                cells.append(
                    describe_cell(
                        res.per_image(method, res.auc_of("lpips_alex", budget)),
                        res.proto,
                    )
                )
        rows.append([method, *cells])
    out = table(["method", *[f"AUC@{b}" for b in res.proto.budgets]], rows)
    top = res.proto.headline_budget
    rows = []
    for method in res.methods:
        if not res.reachable(method, top):
            rows.append([method, na(res, method, top), "", ""])
            continue
        rows.append(
            [
                method,
                describe_cell(
                    res.per_image(method, res.auc_of("lpips_alex", top, "raw")), res.proto
                ),
                describe_cell(
                    res.per_image(method, res.auc_of("lpips_alex", top, "log")), res.proto
                ),
                describe_cell(
                    res.per_image(method, res.auc_of("lpips_alex", top, "late")),
                    res.proto,
                ),
            ]
        )
    return (
        out
        + f"\n\nAUC@{top}: raw integral, log-spaced variant (early strokes), and n >= {res.proto.auc_late_start} only (robustness to the canvas colour, 3.3):\n\n"
        + table(["method", "raw", "log-spaced", f"n >= {res.proto.auc_late_start}"], rows)
    )


def table_b3(res: Results) -> str:
    """Saliency-weighted LPIPS: evaluator, control, lambda sensitivity, AUC."""
    lam = res.proto.headline_lambda
    ev, ctl = res.proto.saliency_methods
    rows = []
    for method in res.methods:
        cells = []
        for budget in res.proto.budgets:
            if not res.reachable(method, budget):
                cells.append(na(res, method, budget))
            else:
                cells.append(
                    describe_cell(
                        res.per_image(method, res.metric(f"sw_{ev}_{lam:g}", budget)),
                        res.proto,
                    )
                )
        rows.append([method, *cells])
    out = table(["method", *[f"SW-LPIPS (λ={lam:g}) n = {b}" for b in res.proto.budgets]], rows)
    ref = 1000
    rows = []
    for method in res.methods:
        if not res.reachable(method, ref):
            rows.append([method, *[na(res, method, ref)] * 4])
            continue
        cells = [
            describe_cell(
                res.per_image(method, res.metric(f"sw_{ev}_{x:g}", ref)),
                res.proto,
            )
            for x in res.proto.sw_lambdas
        ]
        cells.append(
            describe_cell(
                res.per_image(method, res.metric(f"sw_{ctl}_{lam:g}", ref)),
                res.proto,
            )
        )
        cells.append(describe_cell(res.per_image(method, res.metric("lpips_alex", ref)), res.proto))
        rows.append([method, *cells])
    out += f"\n\nSensitivity at n = {ref}: λ of the evaluator ({ev}), the control map ({ctl}, λ={lam:g}), and the unweighted LPIPS-Alex:\n\n"
    out += table(
        [
            "method",
            *[f"{ev} λ={x:g}" for x in res.proto.sw_lambdas],
            f"{ctl} λ={lam:g}",
            "LPIPS-Alex",
        ],
        rows,
    )
    top = res.proto.headline_budget
    rows = []
    for method in res.methods:
        rows.append(
            [method, na(res, method, top)]
            if not res.reachable(method, top)
            else [
                method,
                describe_cell(
                    res.per_image(method, res.auc_of(f"sw_{ev}_{lam:g}", top)), res.proto
                ),
            ]
        )
    return (
        out
        + f"\n\nSW-LPIPS AUC@{top} (normalized by the per-image blank-canvas SW-LPIPS):\n\n"
        + table(["method", f"SW-AUC@{top}"], rows)
    )


def table_b4(res: Results) -> str:
    """Strokes-to-target with right-censoring."""
    rows = []
    notes = []
    for budget in (500, 1000, 2000):
        tau = res.proto.tau_value(budget)
        if tau is None:
            notes.append(f"tau_{budget} is not set (`python -m scripts.eval_run tau`)")
            continue
        for method in res.methods:
            n_star, reached = [], []
            for _image, recs in res.ok[method].items():
                results = [res.strokes(r, tau) for r in recs]
                n_star.append(statistics.fmean(x.n_star for x in results))
                reached.append(statistics.fmean(0.0 if x.censored else 1.0 for x in results))
            rows.append(
                [
                    f"tau_{budget} = {tau:.4f}",
                    method,
                    f"{100 * statistics.fmean(reached):.0f} % ({sum(1 for x in reached if x > 0.5)}/{len(reached)} images)",
                    f"{statistics.median(n_star):.0f}"
                    + (
                        " (> 2000: censored)"
                        if statistics.median(n_star) >= res.proto.max_stroke_count
                        else ""
                    ),
                ]
            )
    out = (
        table(
            [
                "target",
                "method",
                "reach the target within 2000",
                "median n* (censored kept at 2000)",
            ],
            rows,
        )
        if rows
        else ""
    )
    return out + ("\n\n" + "; ".join(notes) if notes else "")


def table_b5(res: Results) -> str:
    """Runtime and peak VRAM."""
    rows = []
    for method in res.methods:
        recs = [r for rs in res.ok[method].values() for r in rs]
        timed = [
            r for r in recs if not res.is_warmup(r)
        ]  # protocol 2.3: the warm-up image is not in the statistic
        run = [r["runtime_s"] for r in timed] or [float("nan")]
        vram = [r["peak_vram_gb"] for r in timed] or [float("nan")]
        res_gb = [r["peak_vram_reserved_gb"] for r in timed] or [float("nan")]

        def iqr(x: list[float]) -> str:
            q = np.quantile(x, [0.25, 0.5, 0.75])
            return f"{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"

        rows.append(
            [
                method,
                f"{len(timed)} of {len(recs)}",
                iqr(run),
                iqr(vram),
                iqr(res_gb),
                f"{res.max_n(method)}",
            ]
        )
    return table(
        [
            "method",
            "timed runs (warm-up excluded)",
            "runtime s, median [IQR]",
            "peak VRAM GB (allocated)",
            "reserved GB",
            "min strokes emitted",
        ],
        rows,
    )


def paired_rows(
    res: Results, endpoints: dict[str, Callable[[dict[str, Any]], float | None]], a: str, b: str
) -> list[tuple[str, stats.PairedResult | None]]:
    out = []
    for name, extract in endpoints.items():
        va, vb = res.per_image(a, extract), res.per_image(b, extract)
        common = sorted(set(va) & set(vb))
        out.append(
            (
                name,
                stats.paired_test(
                    [va[i][0] for i in common],
                    [vb[i][0] for i in common],
                    res.proto.bootstrap["resamples"],
                    res.proto.bootstrap["seed"],
                    res.proto.bootstrap["confidence"],
                )
                if len(common) >= 2
                else None,
            )
        )
    return out


def stats_tables(res: Results) -> str:
    """The confirmatory family F1 (Holm) and the exploratory comparisons (uncorrected)."""
    lam, ev, ctl = res.proto.headline_lambda, *res.proto.saliency_methods
    top = res.proto.headline_budget
    p1, p3 = 1000, 1000
    endpoints = {
        f"P1 LPIPS-Alex@{p1}": res.metric("lpips_alex", p1),
        f"P2 AUC@{top}": res.auc_of("lpips_alex", top),
        f"P3 SW-LPIPS@{p3}": res.metric(f"sw_{ev}_{lam:g}", p3),
    }
    control = {f"P3 SW-LPIPS@{p3}": res.metric(f"sw_{ctl}_{lam:g}", p3)}
    pairs = list(itertools.combinations(res.methods, 2))
    results = [(a, b, name, r) for a, b in pairs for name, r in paired_rows(res, endpoints, a, b)]
    valid = [t for t in results if t[3] is not None]
    adjusted = stats.holm([t[3].p for t in valid])  # type: ignore[union-attr]
    adj = {id(t): p for t, p in zip(valid, adjusted, strict=True)}
    rows = []
    for t in results:
        a, b, name, r = t
        if r is None:
            rows.append([f"{a} vs {b}", name, "n/a", "", "", "", ""])
            continue
        claim = adj[id(t)] < 0.05 and r.effect_ok(res.proto.effect_threshold)
        sign = ""
        if name in control:
            va, vb = res.per_image(a, control[name]), res.per_image(b, control[name])
            common = sorted(set(va) & set(vb))
            if common:
                md = float(np.median([va[i][0] - vb[i][0] for i in common]))
                sign = (
                    "same sign"
                    if (md < 0) == (r.median_diff < 0)
                    else "OPPOSITE sign (depends on the evaluation model)"
                )
        rows.append(
            [
                f"{a} vs {b}",
                name,
                f"{r.p:.3g} / {adj[id(t)]:.3g}",
                f"{r.r:+.2f}",
                f"{r.median_diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                "yes" if claim else "no",
                sign,
            ]
        )
    out = "**F1 (confirmatory, Holm inside the family)**: a difference `a - b` is negative when `a` is better (lower).\n\n"
    out += table(
        [
            "pair",
            "endpoint",
            "p / p Holm",
            "rank-biserial r",
            "median diff [95 % CI]",
            "claim (p Holm < 0.05, CI excludes 0, |r| ≥ 0.3)",
            "control map",
        ],
        rows,
    )
    exp_rows = []
    secondary: dict[str, Callable[[dict[str, Any]], float | None]] = {}
    for bud in res.proto.budgets:
        secondary[f"LPIPS-Alex@{bud}"] = res.metric("lpips_alex", bud)
        secondary[f"AUC@{bud}"] = res.auc_of("lpips_alex", bud)
    for key in ("lpips_vgg", "l1", "ssim"):
        secondary[f"{METRIC_LABEL[key]}@1000"] = res.metric(key, 1000)
    for a, b in pairs:
        for name, r in paired_rows(res, secondary, a, b):
            if r is not None:
                exp_rows.append(
                    [
                        f"{a} vs {b}",
                        name,
                        f"{r.p:.3g}",
                        f"{r.r:+.2f}",
                        f"{r.median_diff:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
                    ]
                )
    out += "\n\n**Exploratory / secondary (uncorrected p-values, never a headline claim)**:\n\n"
    out += table(["pair", "endpoint", "p (uncorrected)", "r", "median diff [95 % CI]"], exp_rows)
    return out


def summary_lines(res: Results, env: dict[str, Any]) -> str:
    rows = []
    for method in res.methods:
        n_img = len(res.ok[method])
        seeds = sorted({r["seed"] for rs in res.ok[method].values() for r in rs})
        rows.append(
            [
                method,
                str(n_img),
                str(seeds),
                res.ok[method][res.images(method)[0]][0]["replicate_kind"],
                str(len(res.failed[method])),
            ]
        )
    head = table(["method", "images", "replicates (seeds)", "replicate kind", "failures"], rows)
    return (
        head
        + f"\n\nGPU {env.get('gpu')}, driver {env.get('driver')}, torch {env.get('torch')}, commit {str(env.get('git_commit'))[:10]}"
        + (" (dirty tree)" if env.get("git_dirty") else "")
    )


# ------------------------------------------------------------------------------------------------------------- figure


def figure(res: Results, path: Path) -> Path:
    """Anytime curves (mean LPIPS-Alex over images with a 95 % bootstrap band) and the reaching fraction."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = dict(zip(res.methods, ["#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad"], strict=False))
    taus: list[tuple[int, float]] = [
        (b, t) for b in (1000,) if (t := res.proto.tau_value(b)) is not None
    ]
    fig, axes = plt.subplots(1, 3 if taus else 2, figsize=(13 if taus else 9, 3.8), dpi=100)
    for ax, logx in zip(axes[:2], (False, True), strict=True):
        for method in res.methods:
            cps = sorted(
                {c for rs in res.ok[method].values() for r in rs for c in r["checkpoints"]}
            )
            cps = [c for c in cps if res.reachable(method, c)]
            means, lo, hi = [], [], []
            for c in cps:
                v = [m for m, _ in res.per_image(method, res.metric("lpips_alex", c)).values()]
                means.append(np.mean(v))
                a, b = stats.bootstrap_ci(v, "mean", 2000, 22)
                lo.append(a)
                hi.append(b)
            x = np.asarray(cps, dtype=float)
            ym, ylo, yhi = np.asarray(means), np.asarray(lo), np.asarray(hi)
            if logx:
                keep = x > 0
                x, ym, ylo, yhi = x[keep], ym[keep], ylo[keep], yhi[keep]
            ax.plot(x, ym, marker="o", ms=3, color=colours[method], label=method)
            ax.fill_between(x, ylo, yhi, color=colours[method], alpha=0.2)
        ax.set(
            xlabel="strokes n" + (" (log)" if logx else ""),
            ylabel="LPIPS-Alex",
            title="anytime curve" + (", log axis" if logx else ""),
        )
        if logx:
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if taus:
        budget, tau = taus[0]
        ax = axes[2]
        grid = np.arange(0, res.proto.max_stroke_count + 1, 10)
        for method in res.methods:
            n_star = []
            for recs in res.ok[method].values():
                n_star.append(
                    np.mean(
                        [
                            res.strokes(r, tau).n_star
                            if not res.strokes(r, tau).censored
                            else np.inf
                            for r in recs
                        ]
                    )
                )
            frac = [np.mean([x <= g for x in n_star]) for g in grid]
            ax.plot(grid, frac, color=colours[method], label=method)
        ax.set(
            xlabel="strokes n",
            ylabel="fraction of images that reached the target",
            title=f"tau_{budget} = {tau:.3f}",
            ylim=(0, 1.02),
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
