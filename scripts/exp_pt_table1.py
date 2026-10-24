"""Table 1 of the Paint Transformer paper and its loss ablations on our data (T-016).

Protocol and tolerances: ``docs/experiments/pt_reproduction.md`` (written before the runs).

    python -m scripts.exp_pt_table1 strokes   # stroke metrics on 4096 synthesized patches (+ the Optim row)
    python -m scripts.exp_pt_table1 real      # L_pixel / L_pcpt / LPIPS-Alex of the PT paintings of the 5 images
    python -m scripts.exp_pt_table1 optim     # the optimization baseline with the same number of strokes (slow)
    python -m scripts.exp_pt_table1 report    # markdown tables from docs/experiments/data/pt_reproduction/

``strokes`` and ``real`` evaluate every model they find (the official weights, the T-014 runs and the
ablation runs under ``outputs/pt_abl/<name>/seed<k>/latest.pt``) and skip those already stored, so they
can be re-run when more trainings have finished.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import torch

from brushstroke.baselines.optim.config import PaintOptimConfig
from brushstroke.baselines.optim.optimize import paint_optim
from brushstroke.eval.metrics import image_metrics
from brushstroke.eval.pcpt import pcpt_loss
from brushstroke.eval.pt_strokes import StrokeScores, network_scores, optim_strokes, unweighted_loss
from brushstroke.renderer.pipeline import RenderConfig
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.inference import load_pt_model, paint
from brushstroke.stroke_model.refine import RefineConfig
from brushstroke.stroke_model.synthetic import SyntheticConfig, SyntheticStrokeGenerator
from brushstroke.utils.config import load_config
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image, save_image
from scripts.pt_repro_common import (
    IMAGES,
    OFFICIAL_WEIGHTS,
    PAPER_ABLATIONS,
    PAPER_REAL,
    PAPER_STROKES,
    dump,
    fmt_ms,
    load,
    mean_std,
    write_block,
)

EVAL_SEED, EVAL_BATCHES, BATCH = 999, 64, 64  # the T-014 evaluation set: 4096 patches
SEEDS = (0, 1, 2)
OPTIM_ITERS = 300
ABLATIONS = {
    "no_pixel": "w/o L_pixel",
    "no_dl1": "w/o D_L1",
    "no_dw": "w/o D_W",
    "no_dbce": "w/o D_bce",
}
GROUPS = {  # group -> list of (key, checkpoint)
    "official": [("official", OFFICIAL_WEIGHTS)],
    "ours": [(f"seed{s}", Path(f"outputs/pt_full/seed{s}/latest.pt")) for s in SEEDS],
    "ours_long": [(f"seed{s}", Path(f"outputs/pt_long/seed{s}/latest.pt")) for s in SEEDS],
    **{
        f"abl_{name}": [
            (f"seed{s}", Path(f"outputs/pt_abl/{name}/seed{s}/latest.pt")) for s in SEEDS
        ]
        for name in ABLATIONS
    },
}
OPTIM_STROKES = (
    1,
    1,
    1,
    3,
    8,
    5,
)  # strokes per patch of the levels (1, 2, 4, 8, 16, 20): 4,261 strokes
WORK = Path("outputs/pt_reproduction")


def _finished(path: Path, group: str) -> bool:
    """A training checkpoint is evaluated only once the run is complete (``latest.pt`` is also written
    during the run)."""
    if not path.exists():
        return False
    if group == "official":
        return True
    steps = 112_640 if group == "ours_long" else 56_320
    return load_pt_model(path)[1]["step"] == steps


def _stored(name: str) -> dict[str, dict[str, object]]:
    try:
        return load(name)  # type: ignore[no-any-return]
    except FileNotFoundError:
        return {}


# ----------------------------------------------------------------------------- strokes


def strokes() -> None:
    device = get_device("auto")
    brushes = load_pt_brushes("train", "data/raw/pt_brush", device)
    loss = unweighted_loss(brushes)
    rng = torch.Generator(device=device).manual_seed(EVAL_SEED)
    generator = SyntheticStrokeGenerator(brushes, SyntheticConfig())
    data = [generator.sample(BATCH, rng) for _ in range(EVAL_BATCHES)]
    store = _stored("table1_strokes")
    for group, items in GROUPS.items():
        for key, path in items:
            if not _finished(path, group) or "abs_wh" in store.get(group, {}).get(key, {}):
                continue
            net, info = load_pt_model(path, device)
            store.setdefault(group, {})[key] = {
                "step": info["step"],
                "raw": network_scores(net, loss, data, abs_h=False),
                "abs_h": network_scores(net, loss, data, abs_h=True),
                "abs_wh": network_scores(net, loss, data, abs_h=True, abs_w=True),
            }
            print(group, key, store[group][key], flush=True)
            dump("table1_strokes", store)
    if "optim" not in store and OFFICIAL_WEIGHTS.exists():
        net, _ = load_pt_model(OFFICIAL_WEIGHTS, device)
        scores = StrokeScores()
        cfg = RefineConfig(enabled=True, iters=OPTIM_ITERS, lr=0.05, keep_if_better=False)
        for i, batch in enumerate(data):
            with torch.no_grad():
                _, _, logits = net.strokes(batch.target, batch.canvas)
            shape, colour, logits = optim_strokes(batch, logits > 0, brushes, cfg, seed=i)
            scores.add(loss, shape, colour, logits, batch)
        store["optim"] = {"optim": {"abs_h": scores.result(), "raw": scores.result()}}
        print("optim", store["optim"], flush=True)
        dump("table1_strokes", store)


# ----------------------------------------------------------------------------- real images


def _images(device: torch.device) -> dict[str, torch.Tensor]:
    out = {}
    for name in IMAGES:
        path = WORK / "in" / f"{name}_512.png"
        if not path.exists():
            save_image(
                center_crop_resize(load_image(Path("data/raw") / f"{name}.jpg"), (512, 512)), path
            )
        out[name] = load_image(path).to(device)
    return out


def _real_row(image: torch.Tensor, painting: torch.Tensor, strokes: int) -> dict[str, float]:
    m = image_metrics(painting, image)
    return {
        "l_pixel": m["l1"],
        "l_pcpt": pcpt_loss(painting, image),
        "lpips_alex": m["lpips_alex"],
        "lpips_vgg": m["lpips_vgg"],
        "strokes": float(strokes),
    }


def real() -> None:
    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    images = _images(device)
    store = _stored("table1_real")
    for group, items in GROUPS.items():
        for key, path in items:
            if not _finished(path, group) or "abs_wh" in store.get(group, {}).get(key, {}):
                continue
            net, _ = load_pt_model(path, device)
            rows = {}
            for name, image in images.items():
                result = paint(net, image, brushes)
                rows[name] = _real_row(image, result.canvas, int(result.stats["strokes_official"]))  # type: ignore[call-overload]
            store.setdefault(group, {})[key] = rows
            print(
                group,
                key,
                {
                    k: round(sum(r[k] for r in rows.values()) / len(rows), 4)
                    for k in ("l_pixel", "l_pcpt")
                },
                flush=True,
            )
            dump("table1_real", store)


def optim() -> None:
    """The M1 optimization baseline with about as many strokes as PT, with and without the LPIPS term."""
    device = get_device("auto")
    images = _images(device)
    store = _stored("table1_optim")
    for variant, lpips in (("default", 0.1), ("l1_only", 0.0)):
        for name, image in images.items():
            if name in store.get(variant, {}):
                continue
            cfg = load_config(PaintOptimConfig, "configs/baseline_optim.yaml")
            cfg.optim.strokes_per_patch = list(OPTIM_STROKES)
            cfg.loss.lpips = lpips
            seq = paint_optim(image.cpu(), cfg.optim, cfg.loss, cfg.brush, cfg.render, 0, device)
            painting = seq.render(cfg=RenderConfig(**dataclasses.asdict(cfg.render))).to(device)
            save_image(painting, WORK / "optim" / variant / f"{name}.png")
            store.setdefault(variant, {})[name] = {
                **_real_row(image, painting, len(seq)),
                "runtime_seconds": seq.meta["runtime_seconds"],
            }
            print(variant, name, store[variant][name], flush=True)
            dump("table1_optim", store)


# ----------------------------------------------------------------------------- report


def _seed_means(rows_by_seed: dict[str, dict[str, dict[str, float]]], key: str) -> list[float]:
    """Per seed, the mean over the images of ``key``."""
    return [sum(r[key] for r in rows.values()) / len(rows) for rows in rows_by_seed.values()]


def _group_real(real_data: dict[str, dict[str, object]], group: str, key: str) -> list[float]:
    return _seed_means(real_data.get(group, {}), key)  # type: ignore[arg-type]


def _group_strokes(
    data: dict[str, dict[str, object]], group: str, mode: str, key: str
) -> list[float]:
    return [v[mode][key] for v in data.get(group, {}).values()]  # type: ignore[index]


def build_real_table(real_data: dict, optim_data: dict) -> str:
    lines = [
        "| row | L_pixel | L_pcpt | LPIPS-Alex | strokes |",
        "|---|---|---|---|---|",
    ]
    for name in ("RL", "Optim", "PT"):
        p = PAPER_REAL[name]
        lines.append(f"| paper {name} (own datasets) | {p[0]} | {p[1]} | n/a | n/a |")
    labels = {
        "official": "official model (pretrained weights)",
        "ours": "**ours**, official recipe, 3 seeds",
        "ours_long": "ours, stretched recipe (112,640 steps), 3 seeds",
    }
    for group, label in labels.items():
        if group not in real_data:
            continue
        cols = [
            _group_real(real_data, group, k) for k in ("l_pixel", "l_pcpt", "lpips_alex", "strokes")
        ]
        lines.append(
            f"| {label} | {fmt_ms(cols[0], 4)} | {fmt_ms(cols[1], 3)} | {fmt_ms(cols[2], 3)} | {fmt_ms(cols[3], 0)} |"
        )
    for variant, label in (
        ("default", "ours Optim (M1 baseline, L1 + 0.1 LPIPS), 4,261 strokes"),
        ("l1_only", "ours Optim, L1 only, 4,261 strokes"),
    ):
        rows = optim_data.get(variant)
        if rows:
            vals = {
                k: sum(r[k] for r in rows.values()) / len(rows)
                for k in ("l_pixel", "l_pcpt", "lpips_alex", "strokes")
            }
            lines.append(
                f"| {label} | {vals['l_pixel']:.4f} | {vals['l_pcpt']:.3f} | {vals['lpips_alex']:.3f} | {vals['strokes']:.0f} |"
            )
    return "\n".join(lines)


def build_stroke_table(data: dict) -> str:
    lines = [
        "| row | L_pixel | D_L1 | D_W | painted strokes / patch | matched strokes with h < 0 | with w < 0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("RL", "Optim", "PT"):
        p = PAPER_STROKES[name]
        lines.append(
            f"| paper {name} | {p[0]} | {p[1] if p[1] is not None else 'n/a'} | {p[2] if p[2] is not None else 'n/a'} | n/a | n/a | n/a |"
        )
    labels = {
        "official": "official model",
        "ours": "**ours**, official recipe, 3 seeds",
        "ours_long": "ours, stretched recipe, 3 seeds",
    }
    for group, label in labels.items():
        if group not in data:
            continue
        for mode, suffix in (
            ("abs_wh", " (`|w|`, `|h|`)"),
            ("abs_h", " (`|h|`)"),
            ("raw", " (raw)"),
        ):
            cols = [
                _group_strokes(data, group, mode, k)
                for k in ("l_pixel", "d_l1", "d_w", "painted_per_patch")
            ]
            cols.append(_group_strokes(data, group, "raw", "h_negative"))  # raw sign in all rows
            cols.append(_group_strokes(data, group, "raw", "w_negative"))
            lines.append(
                f"| {label}{suffix} | {fmt_ms(cols[0], 4)} | {fmt_ms(cols[1], 4)} | {fmt_ms(cols[2], 4)} | {fmt_ms(cols[3], 2)} | {fmt_ms([100 * x for x in cols[4]], 1)} % | {fmt_ms([100 * x for x in cols[5]], 1)} % |"
            )
    if "optim" in data:
        o = data["optim"]["optim"]["abs_h"]  # type: ignore[index]
        lines.append(
            f"| **ours Optim** (random init, 300 it., same count as the official PT) | {o['l_pixel']:.4f} | {o['d_l1']:.4f} | {o['d_w']:.4f} | {o['painted_per_patch']:.2f} | n/a | n/a |"
        )
    return "\n".join(lines)


def build_ablation_table(real_data: dict, stroke_data: dict) -> str:
    cols = ("L_pixel (real)", "L_pcpt (real)", "L_pixel (strokes)", "D_L1", "D_W")
    lines = [
        "| model | " + " | ".join(cols) + " |",
        "|---|---|---|---|---|---|",
    ]
    ours: dict[str, list[float]] = {}
    sets = {"full": "ours"} | {label: f"abl_{name}" for name, label in ABLATIONS.items()}
    for label, group in sets.items():
        if group not in real_data or group not in stroke_data:
            continue
        values = [
            _group_real(real_data, group, "l_pixel"),
            _group_real(real_data, group, "l_pcpt"),
            _group_strokes(stroke_data, group, "abs_h", "l_pixel"),
            _group_strokes(stroke_data, group, "abs_h", "d_l1"),
            _group_strokes(stroke_data, group, "abs_h", "d_w"),
        ]
        ours[label] = [mean_std(v)[0] for v in values]
        paper = PAPER_ABLATIONS[label]
        lines.append(f"| paper {label} | " + " | ".join(str(x) for x in paper) + " |")
        lines.append(
            f"| **ours** {label} (n = {len(values[0])}) | "
            + " | ".join(fmt_ms(v, 3) for v in values)
            + " |"
        )
    lines += [
        "",
        "Effect of removing a term (ablated / full; > 1 = worse):",
        "",
        "| removed term | " + " | ".join(cols) + " |",
        "|---|---|---|---|---|---|",
    ]
    if "full" in ours:
        for label in (name for name in ABLATIONS.values() if name in ours):
            paper = [
                a / b for a, b in zip(PAPER_ABLATIONS[label], PAPER_ABLATIONS["full"], strict=True)
            ]
            mine = [a / b for a, b in zip(ours[label], ours["full"], strict=True)]
            lines.append(
                f"| {label} | "
                + " | ".join(
                    f"paper {p:.2f}, ours {m:.2f}" for p, m in zip(paper, mine, strict=True)
                )
                + " |"
            )
    return "\n".join(lines)


def build_symmetry_table(stroke_data: dict) -> str:
    """D_L1 per seed with the sign of ``h`` folded, then of ``w`` and ``h``: the renderer (and
    ``D_W``) cannot see these signs, only ``D_L1`` pins the parameter convention."""
    lines = [
        "| model | seed 0 | seed 1 | seed 2 |",
        "|---|---|---|---|",
    ]
    sets = {"full": "ours"} | {label: f"abl_{name}" for name, label in ABLATIONS.items()}
    for label, group in sets.items():
        runs = stroke_data.get(group, {})
        if not runs:
            continue
        cells = []
        for key in sorted(runs):
            run = runs[key]
            cells.append(
                f"{run['abs_h']['d_l1']:.3f} -> {run['abs_wh']['d_l1']:.3f}"
                f" (w < 0: {100 * run['raw']['w_negative']:.0f} %)"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report() -> None:
    real_data, stroke_data, optim_data = (
        _stored("table1_real"),
        _stored("table1_strokes"),
        _stored("table1_optim"),
    )
    text = "\n\n".join(
        [
            "**T1a, real images** (five public-domain images at 512 x 512; paper rows are on different datasets: orderings only; our rows are mean ± std over seeds of the mean over the images):",
            build_real_table(real_data, optim_data),
            "**T1b, synthesized strokes** (4,096 held-out patches of the official training distribution):",
            build_stroke_table(stroke_data),
            "**T1c, loss ablations** (official recipe, `|h|`, mean ± std over seeds):",
            build_ablation_table(real_data, stroke_data),
            "**D_L1 per seed, `|h|` -> `|w|`, `|h|`** (the sign of `w` and `h` is invisible to the renderer and to `D_W`; `w < 0` = share of matched strokes with a negative width, raw):",
            build_symmetry_table(stroke_data),
        ]
    )
    write_block("TABLE1", text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["strokes", "real", "optim", "report"])
    args = parser.parse_args(argv)
    {"strokes": strokes, "real": real, "optim": optim, "report": report}[args.stage]()


if __name__ == "__main__":
    main(sys.argv[1:])
