"""Evaluation and report of the parameter-convention experiment (T-019); see ``exp_pt_convention``.

``evaluate``: every available checkpoint (the released weights, the T-014 runs, the T-016 ablation
runs and the T-019 option runs) on the T-014 held-out set (4096 synthesized patches, seed 999) with
the unweighted stroke metrics (raw, with ``|h|`` as in T-016, and the convention-invariant ``D_L1``
of every class level) and the three totals (official loss with raw L1, after the ``|h|`` fold, and
with the invariant ``L1``). Stored in ``docs/experiments/data/pt_convention/eval.json``.

``report``: tables and verdicts (rules fixed in ``docs/experiments/pt_training.md`` before the runs)
written into the generated blocks of ``pt_training.md`` and ``pt_reproduction.md``.
"""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path
from typing import Any

import torch

from brushstroke.eval.pt_strokes import INVARIANT_LEVELS, network_scores, unweighted_loss
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.convention import variant_specs
from brushstroke.stroke_model.inference import load_pt_model
from brushstroke.stroke_model.losses import PaintTransformerLoss, PTLossConfig
from brushstroke.stroke_model.synthetic import SyntheticConfig, SyntheticStrokeGenerator
from brushstroke.utils.device import get_device
from scripts.exp_pt_ablation_verdict import T_CRITICAL, welch_t
from scripts.pt_repro_common import PW_GROUPS, PW_ROOT, fmt_ms, mean_std, write_block

DATA = Path("docs/experiments/data/pt_convention")
DOC_TRAINING = Path("docs/experiments/pt_training.md")
DOC_REPRO = Path("docs/experiments/pt_reproduction.md")
EVAL_SEED, EVAL_BATCHES, BATCH = 999, 64, 64  # the T-014 evaluation set: 4096 patches
SEEDS = (0, 1, 2)
OFFICIAL_WEIGHTS = Path("data/raw/pt_official/model.pth")
STEPS = {"official": 56_320, "long": 112_640}
OPTIONS = {
    "optA_positive": "A: positive `w`, `h` (softplus), `theta` wrapped",
    "optB_invariant_l1": "B: class-invariant L1 and matching (mirror class)",
    "optC_sign_penalty": "C: penalty `relu(-w) + relu(-h)` (weight 10)",
}
ABLATIONS = ("no_pixel", "no_dl1", "no_dw", "no_dbce")
# Rules fixed before the runs (docs/experiments/pt_training.md, section T-019)
OUTLIER_SHARE = (
    0.5  # a seed is an outlier when more than this share of matched strokes has w or h < 0
)
CV_MAX = 0.05  # H2: coefficient of variation of D_L1 across seeds at most 5 %
CV_GAIN = 3.0  # ... and at least 3 times lower than the raw D_L1's
STAT_KEYS = (
    "l_pixel",
    "d_l1",
    "d_w",
    "h_negative",
    "w_negative",
    *[f"d_l1_{k}" for k in INVARIANT_LEVELS],
)


def groups() -> dict[str, list[tuple[str, Path, int | None]]]:
    """group -> [(key, checkpoint, expected final step or None)]."""
    out: dict[str, list[tuple[str, Path, int | None]]] = {
        "official": [("official", OFFICIAL_WEIGHTS, None)],
        "ours": [
            (f"seed{s}", Path(f"outputs/pt_full/seed{s}/latest.pt"), STEPS["official"])
            for s in SEEDS
        ],
        "ours_long": [
            (f"seed{s}", Path(f"outputs/pt_long/seed{s}/latest.pt"), STEPS["long"]) for s in SEEDS
        ],
    }
    for name in ABLATIONS:
        out[f"abl_{name}"] = [
            (f"seed{s}", Path(f"outputs/pt_abl/{name}/seed{s}/latest.pt"), STEPS["official"])
            for s in SEEDS
        ]
    for group, name in PW_GROUPS.items():  # T-021
        out[group] = [
            (f"seed{s}", PW_ROOT / name / f"seed{s}" / "latest.pt", STEPS["official"])
            for s in SEEDS
        ]
    for option in OPTIONS:
        out[option] = [
            (f"seed{s}", Path(f"outputs/pt_conv/{option}/seed{s}/latest.pt"), STEPS["official"])
            for s in SEEDS
        ]
        out[f"{option}_long"] = [
            (f"seed{s}", Path(f"outputs/pt_conv_long/{option}/seed{s}/latest.pt"), STEPS["long"])
            for s in SEEDS
        ]
    return out


def load_store() -> dict[str, Any]:
    path = DATA / "eval.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_store(store: dict[str, Any]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "eval.json").write_text(json.dumps(store, indent=1), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------- evaluate


@torch.no_grad()
def held_out_totals(net: Any, data: list[Any], brushes: torch.Tensor) -> dict[str, float]:
    """Mean official total over the batches: ``raw``; ``abs_h`` (``h`` replaced by ``|h|``, the T-014
    protocol); ``inv`` (official weights, L1 and matching invariant over the mirror class)."""
    official = PaintTransformerLoss(brushes, PTLossConfig())
    invariant = PaintTransformerLoss(brushes, PTLossConfig(l1_invariance="mirror"))
    totals = dict.fromkeys(("raw", "abs_h", "inv"), 0.0)
    net.eval()
    for batch in data:
        shape, colour, logits = net.strokes(batch.target, batch.canvas)
        folded = shape.clone()
        folded[..., 3] = folded[..., 3].abs()
        totals["raw"] += float(official(shape, colour, logits, batch).total)
        totals["abs_h"] += float(official(folded, colour, logits, batch).total)
        totals["inv"] += float(invariant(shape, colour, logits, batch).total)
    return {k: v / len(data) for k, v in totals.items()}


def evaluate() -> None:
    device = get_device("auto")
    brushes = load_pt_brushes("train", "data/raw/pt_brush", device)
    loss = unweighted_loss(brushes)
    rng = torch.Generator(device=device).manual_seed(EVAL_SEED)
    generator = SyntheticStrokeGenerator(brushes, SyntheticConfig())
    data = [generator.sample(BATCH, rng) for _ in range(EVAL_BATCHES)]
    store = load_store()
    for group, items in groups().items():
        for key, path, steps in items:
            if key in store.get(group, {}) or not path.exists():
                continue
            net, info = load_pt_model(path, device)
            if steps is not None and info["step"] != steps:
                continue  # the run is not finished
            store.setdefault(group, {})[key] = {
                "step": info["step"],
                "raw": network_scores(net, loss, data, abs_h=False),
                "abs_h": network_scores(net, loss, data, abs_h=True),
                "totals": held_out_totals(net, data, brushes),
            }
            print(group, key, store[group][key]["totals"], store[group][key]["raw"], flush=True)
            save_store(store)
    archive_logs()


def archive_logs() -> None:
    """Copy the small logs of the option runs (summary, validation curve, resolved config)."""
    for root in (Path("outputs/pt_conv"), Path("outputs/pt_conv_long")):
        for run in sorted(root.glob("*/seed*")):
            if not (run / "summary.json").exists():
                continue
            dest = DATA / "runs" / root.name / run.parent.name / run.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("summary.json", "val.csv", "config.resolved.yaml"):
                shutil.copy(run / name, dest / name)


# ----------------------------------------------------------------------------- report helpers


def _values(store: dict[str, Any], group: str, mode: str, key: str) -> list[float]:
    return [v[mode][key] for v in store.get(group, {}).values()]


def _totals(store: dict[str, Any], group: str, key: str) -> list[float]:
    return [v["totals"][key] for v in store.get(group, {}).values()]


def _cv(values: list[float]) -> float:
    m, s = mean_std(values)
    return s / m if m else float("inf")


def _outliers(store: dict[str, Any], group: str) -> int:
    """Seeds with more than ``OUTLIER_SHARE`` of the matched strokes with ``w < 0`` or ``h < 0``."""
    return sum(
        1
        for v in store.get(group, {}).values()
        if max(v["raw"]["h_negative"], v["raw"]["w_negative"]) > OUTLIER_SHARE
    )


def _row(store: dict[str, Any], group: str, label: str, mode: str = "raw") -> str:
    cells = []
    for key in ("l_pixel", "d_l1", "d_l1_exact", "d_l1_mirror", "d_l1_rot180", "d_l1_swap", "d_w"):
        cells.append(fmt_ms(_values(store, group, mode, key), 4))
    neg_h = fmt_ms([100 * x for x in _values(store, group, "raw", "h_negative")], 1)
    neg_w = fmt_ms([100 * x for x in _values(store, group, "raw", "w_negative")], 1)
    return (
        f"| {label} | "
        + " | ".join(cells)
        + f" | {neg_h} % | {neg_w} % | {_outliers(store, group)} / {len(store[group])} |"
    )


STROKE_HEADER = (
    "| model | L_pixel | D_L1 | D_L1 exact | D_L1 mirror | D_L1 rot180 | D_L1 swap | D_W | h < 0 | w < 0 | outlier seeds |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


# ----------------------------------------------------------------------------- report: symmetry


def symmetry_table() -> str:
    path = DATA / "symmetry.json"
    if not path.exists():
        return "_(run `python -m scripts.exp_pt_convention symmetry`)_"
    sym = json.loads(path.read_text(encoding="utf-8"))
    train, infer = sym["pt_train"], sym["pt_infer"]
    classes = {
        tuple(spec): name
        for name, specs in (
            ("identity", variant_specs("none")),
            ("exact", variant_specs("exact")[1:]),
            ("mirror", variant_specs("mirror")[2:]),
        )
        for spec in specs
    }
    lines = [
        f"Mean absolute alpha difference between the stroke and each parameter variant, {sym['strokes']} random strokes of the training distribution ({sym['size']} x {sym['size']} pixels; the mean alpha is {train['area']:.3f}); `error / area` is the share of the stroke that changes. Unrelated strokes of the same distribution differ by {sym['unrelated_alpha_error']:.4f} ({sym['unrelated_alpha_error'] / train['area']:.2f} of the area).",
        "",
        "| variant `(sign w, sign h, theta shift, swap)` | class | alpha error (train) | error / area | foreground error (train) | alpha error (inference) |",
        "|---|---|---|---|---|---|",
    ]
    for a, b in zip(train["variants"][1:], infer["variants"][1:], strict=True):
        spec = tuple(a["spec"])
        name = classes.get(spec, "rot180 (diagnostic)" if not spec[3] else "swap (diagnostic)")
        lines.append(
            f"| `({spec[0]:+.0f}, {spec[1]:+.0f}, {spec[2]:+.1f}, {'swap' if spec[3] else '-'})` | {name} | {a['alpha_error']:.5f} | {a['alpha_error_relative']:.3f} | {a['foreground_error']:.5f} | {b['alpha_error']:.5f} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- report: H2


def h2_section(store: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    lines = [STROKE_HEADER]
    labels = {
        "official": "official weights",
        "ours": "ours, official recipe (T-014)",
        "ours_long": "ours, stretched recipe (T-014)",
        **{f"abl_{n}": f"ablation w/o {n[3:]} (T-016)" for n in ABLATIONS},
    }
    for group, label in labels.items():
        if group in store:
            lines.append(_row(store, group, label))
    lines += [
        "",
        "**Spread of `D_L1` across the 3 seeds (coefficient of variation = std / mean).**",
        "",
    ]
    lines += [
        "| group | raw | `\\|h\\|` fold | exact | mirror | rot180 | swap |",
        "|---|---|---|---|---|---|---|",
    ]
    holds: dict[str, bool] = {}
    for group, label in labels.items():
        if group not in store or len(store[group]) < 3:
            continue
        cvs = {
            "raw": _cv(_values(store, group, "raw", "d_l1")),
            "abs_h": _cv(_values(store, group, "abs_h", "d_l1")),
            **{k: _cv(_values(store, group, "raw", f"d_l1_{k}")) for k in INVARIANT_LEVELS},
        }
        for level in INVARIANT_LEVELS:
            ok = cvs[level] <= CV_MAX and cvs[level] * CV_GAIN <= cvs["raw"]
            holds[f"{group}/{level}"] = ok
        lines.append(
            f"| {label} | "
            + " | ".join(f"{100 * cvs[k]:.1f} %" for k in ("raw", "abs_h", *INVARIANT_LEVELS))
            + " |"
        )
    return "\n".join(lines), holds


def clean_bias(store: dict[str, Any]) -> str:
    """Mean relative decrease of ``D_L1`` caused by each class level on the models without any negative
    sign (their convention is already the ground truth's: the minimum over the class should not
    change them)."""
    clean = [
        v["raw"]
        for group in store.values()
        for v in group.values()
        if max(v["raw"]["h_negative"], v["raw"]["w_negative"]) == 0
    ]
    cells = [
        100 * statistics.mean(1 - r[f"d_l1_{k}"] / r["d_l1"] for r in clean)
        for k in INVARIANT_LEVELS
    ]
    return (
        f"Bias of the invariant metric on the {len(clean)} models with no negative `w` or `h` (mean decrease of `D_L1` "
        "relative to the raw value; a level that lowers clean models also lowers wrong ones for the wrong reason): "
        + ", ".join(f"`{k}` {c:.1f} %" for k, c in zip(INVARIANT_LEVELS, cells, strict=True))
        + "."
    )


def h2_verdict(holds: dict[str, bool]) -> str:
    primary = ("ours", "ours_long")
    lines = []
    for level in INVARIANT_LEVELS:
        ok = [holds.get(f"{g}/{level}") for g in primary]
        ablation = [holds.get(f"abl_{n}/{level}") for n in ABLATIONS]
        lines.append(
            f"- `{level}`: T-014 groups {'both' if all(ok) else 'not both'} within the rule ({', '.join('yes' if o else 'no' for o in ok)}); ablation groups {sum(bool(a) for a in ablation)} / {len([a for a in ablation if a is not None])}."
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- report: H1


def option_rows(
    store: dict[str, Any], suffix: str
) -> tuple[list[str], dict[str, dict[str, object]]]:
    """Table rows and verdict inputs of the three options (``suffix`` '' = official recipe)."""
    base_group = "ours_long" if suffix else "ours"
    base_inv = _totals(store, base_group, "inv")
    verdicts: dict[str, dict[str, object]] = {}
    rows = []
    for option, label in OPTIONS.items():
        group = option + suffix
        if group not in store or len(store[group]) < 3:
            continue
        total_inv = _totals(store, group, "inv")
        diff = statistics.mean(total_inv) - statistics.mean(base_inv)
        t = welch_t(total_inv, base_inv)
        outliers = _outliers(store, group)
        worse = diff > 0 and t >= T_CRITICAL
        verdicts[option] = {
            "outliers": outliers,
            "mean": statistics.mean(total_inv),
            "diff": diff,
            "t": t,
            "worse": worse,
            "h1": outliers == 0 and not worse,
        }
        rows.append(
            f"| {label} | {fmt_ms(_totals(store, group, 'raw'), 3)} | {fmt_ms(_totals(store, group, 'abs_h'), 3)} | {fmt_ms(total_inv, 3)} | {diff:+.3f} (t = {t:+.2f}) | {fmt_ms([100 * x for x in _values(store, group, 'raw', 'h_negative')], 1)} % | {fmt_ms([100 * x for x in _values(store, group, 'raw', 'w_negative')], 1)} % | {outliers} / 3 |"
        )
    return rows, verdicts


TOTAL_HEADER = (
    "| option | total (raw) | total (`\\|h\\|`) | total (invariant L1) | invariant total vs baseline | h < 0 | w < 0 | outlier seeds |\n"
    "|---|---|---|---|---|---|---|---|"
)


def best_option(store: dict[str, Any]) -> str | None:
    """The option that meets H1 with the lowest invariant total, else the one with the fewest outlier
    seeds (ties: lowest invariant total); ``None`` if no option has finished."""
    _, verdicts = option_rows(store, "")
    if not verdicts:
        return None
    return min(
        verdicts,
        key=lambda o: (not verdicts[o]["h1"], verdicts[o]["outliers"], verdicts[o]["mean"]),
    )


def report() -> None:
    store = load_store()
    parts = ["### Equivalence classes verified by rendering", "", symmetry_table(), ""]
    h2_table, holds = h2_section(store)
    parts += [
        "### H2: convention-invariant D_L1 on the existing checkpoints (T-014, T-016)",
        "",
        h2_table,
        "",
    ]
    parts += [clean_bias(store), ""]
    parts += [
        "Rule (H2 holds at a level): coefficient of variation across seeds at most "
        f"{100 * CV_MAX:.0f} % and at least {CV_GAIN:.0f} times lower than the raw `D_L1`'s.",
        "",
        h2_verdict(holds),
        "",
    ]
    baseline = [
        "| baseline (T-014) | "
        + fmt_ms(_totals(store, g, "raw"), 3)
        + " | "
        + fmt_ms(_totals(store, g, "abs_h"), 3)
        + " | "
        + fmt_ms(_totals(store, g, "inv"), 3)
        + " | - | "
        + fmt_ms([100 * x for x in _values(store, g, "raw", "h_negative")], 1)
        + " % | "
        + fmt_ms([100 * x for x in _values(store, g, "raw", "w_negative")], 1)
        + f" % | {_outliers(store, g)} / 3 |"
        for g in ("ours",)
        if g in store
    ]
    rows, verdicts = option_rows(store, "")
    if rows:
        parts += [
            "### H1: the three options, official recipe (56,320 steps), 3 seeds",
            "",
            TOTAL_HEADER,
            *baseline,
            *rows,
            "",
        ]
        stroke = [STROKE_HEADER, _row(store, "ours", "baseline, raw")]
        stroke += [_row(store, o, OPTIONS[o]) for o in OPTIONS if o in store]
        parts += ["Stroke metrics of the same models:", "", *stroke, ""]
        for option, v in verdicts.items():
            parts.append(
                f"- **{OPTIONS[option]}**: {v['outliers']} / 3 outlier seeds; invariant total {v['diff']:+.3f} vs baseline (Welch t = {v['t']:+.2f}, critical {T_CRITICAL}): {'significantly worse' if v['worse'] else 'not worse'}; **H1 {'met' if v['h1'] else 'not met'}**."
            )
        parts.append("")
    best = best_option(store)
    long_rows, long_verdicts = option_rows(store, "_long")
    if long_rows and best is not None:
        parts += [
            f"### Stretched recipe (112,640 steps) for the best option ({OPTIONS[best]})",
            "",
            TOTAL_HEADER,
            "| baseline (T-014, stretched) | "
            + fmt_ms(_totals(store, "ours_long", "raw"), 3)
            + " | "
            + fmt_ms(_totals(store, "ours_long", "abs_h"), 3)
            + " | "
            + fmt_ms(_totals(store, "ours_long", "inv"), 3)
            + " | - | "
            + fmt_ms([100 * x for x in _values(store, "ours_long", "raw", "h_negative")], 1)
            + " % | "
            + fmt_ms([100 * x for x in _values(store, "ours_long", "raw", "w_negative")], 1)
            + f" % | {_outliers(store, 'ours_long')} / 3 |",
            *long_rows,
            "",
        ]
        stroke = [STROKE_HEADER, _row(store, "ours_long", "baseline, raw")]
        stroke += [_row(store, o + "_long", OPTIONS[o]) for o in OPTIONS if o + "_long" in store]
        parts += ["Stroke metrics of the same models:", "", *stroke, ""]
        for option, v in long_verdicts.items():
            parts.append(
                f"- **{OPTIONS[option]}** (stretched): {v['outliers']} / 3 outlier seeds; invariant total {v['diff']:+.3f} vs baseline (t = {v['t']:+.2f}): {'significantly worse' if v['worse'] else 'not worse'}; **H1 {'met' if v['h1'] else 'not met'}**."
            )
        parts.append("")
    text = "\n".join(parts)
    write_block("CONVENTION", text, DOC_TRAINING)
    write_block("CONVENTION_NOTE", repro_note(store, holds), DOC_REPRO)
    print(text)


def repro_note(store: dict[str, Any], holds: dict[str, bool]) -> str:
    """Short note for ``pt_reproduction.md``: the D_L1 column of Table 1 under the invariant metric."""
    lines = [
        "**T-019 update of the `D_L1` column (item 3 above).** The convention outliers are real (the classes are derived and verified by rendering in `docs/design/pt_convention.md`): `D_L1` of the full model, stretched recipe, 3 seeds, by equivalence class of the prediction:",
        "",
        "| `D_L1` | raw | `\\|h\\|` fold | exact class | mirror class | rot180 | swap |",
        "|---|---|---|---|---|---|---|",
    ]
    for group, label in (
        ("ours", "ours, official recipe"),
        ("ours_long", "ours, stretched recipe"),
    ):
        if group in store:
            lines.append(
                f"| {label} | {fmt_ms(_values(store, group, 'raw', 'd_l1'), 4)} | {fmt_ms(_values(store, group, 'abs_h', 'd_l1'), 4)} | "
                + " | ".join(
                    fmt_ms(_values(store, group, "raw", f"d_l1_{k}"), 4) for k in INVARIANT_LEVELS
                )
                + " |"
            )
    lines += [
        "",
        "Details, per-model values and the options that fix the convention: `docs/experiments/pt_training.md`, section T-019.",
    ]
    return "\n".join(lines)
