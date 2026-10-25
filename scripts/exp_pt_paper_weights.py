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
import statistics
import subprocess
import sys
import time

from scripts import exp_pt_table1, pt_convention_report
from scripts.exp_pt_ablation_verdict import T_CRITICAL, columns, verdict, welch_t
from scripts.exp_pt_table1 import ABLATIONS, _stored
from scripts.pt_repro_common import (
    PAPER_ABLATIONS,
    PW_GROUPS,
    PW_ROOT,
    PW_TERMS,
    fmt_ms,
    write_block,
)

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
COLUMNS = ("L_pixel (real)", "L_pcpt (real)", "L_pixel (strokes)", "D_L1", "D_W")
CONFIRMED_MARGIN = 3  # H confirmed with at least this many more cells than the code weights
SIDES = {  # weights -> (full-model group, group prefix of the ablations)
    "code": ("ours", "abl_"),
    "paper": ("pw_full", "pw_abl_"),
}
VARIANTS = {  # D_L1 flavour of the 4th column -> label
    "fold": "`\\|w\\|`, `\\|h\\|` folded (primary, the T-016 counting)",
    "abs_h": "`\\|h\\|` (T-016 protocol numbers)",
    "mirror": "mirror-invariant (T-019)",
}

# ----------------------------------------------------------------------------- training


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


# ----------------------------------------------------------------------------- evaluation


def evaluate() -> None:
    """T-016 protocol (synthesized strokes, real images) and the invariant metrics of every
    checkpoint found (the 15 new runs; the stored ones are skipped)."""
    exp_pt_table1.strokes()
    exp_pt_table1.real()
    pt_convention_report.evaluate()


# ----------------------------------------------------------------------------- report


def cell_columns(
    real: dict, strokes: dict, conv: dict, group: str, variant: str
) -> list[list[float]]:
    """The five per-seed value lists of a group; ``variant`` selects the ``D_L1`` flavour."""
    cols = columns(real, strokes, group, fold=variant == "fold")
    if variant == "mirror":
        cols[3] = [v["raw"]["d_l1_mirror"] for v in conv.get(group, {}).values()]
    return cols


def ratios(
    real: dict, strokes: dict, conv: dict, side: str, variant: str
) -> dict[str, dict[str, list[float]]]:
    """Per removed term: the five ablated / full ratios and the five Welch t statistics."""
    full_group, prefix = SIDES[side]
    full = cell_columns(real, strokes, conv, full_group, variant)
    out: dict[str, dict[str, list[float]]] = {}
    for term in PW_TERMS:
        abl = cell_columns(real, strokes, conv, f"{prefix}{term}", variant)
        if not all(abl) or not all(full):
            continue
        out[term] = {
            "ratio": [
                statistics.mean(a) / statistics.mean(f) for a, f in zip(abl, full, strict=True)
            ],
            "t": [welch_t(a, f) for a, f in zip(abl, full, strict=True)],
        }
    return out


def direction(ratio: float) -> bool:
    """The paper's direction (worse without the term): the ratio rounded to 2 decimals is above 1."""
    return round(ratio, 2) > 1.0


def significant(ratio: float, t: float) -> bool:
    return ratio > 1 and t >= T_CRITICAL


def counts(table: dict[str, dict[str, list[float]]]) -> tuple[int, int]:
    """(cells in the paper's direction, cells significantly worse by the T-017 rule)."""
    cells = [(r, t) for v in table.values() for r, t in zip(v["ratio"], v["t"], strict=True)]
    return sum(direction(r) for r, _ in cells), sum(significant(r, t) for r, t in cells)


def hypothesis(code: int, paper: int) -> str:
    gain = paper - code
    if gain >= CONFIRMED_MARGIN:
        return "confirmed"
    if gain <= 0:
        return "refuted"
    return "weak (within noise)"


def paper_ratio(term: str) -> list[float]:
    return [
        a / b
        for a, b in zip(PAPER_ABLATIONS[ABLATIONS[term]], PAPER_ABLATIONS["full"], strict=True)
    ]


def absolute_table(real: dict, strokes: dict, conv: dict) -> list[str]:
    lines = [
        "| model | " + " | ".join(COLUMNS) + " | D_L1 mirror | h < 0 | w < 0 |",
        "|---|" + "---|" * (len(COLUMNS) + 3),
    ]
    sets = [("full", "ours", "pw_full")] + [(t, f"abl_{t}", f"pw_abl_{t}") for t in PW_TERMS]
    for term, code_group, pw_group in sets:
        label = "full" if term == "full" else ABLATIONS[term]
        paper = " | ".join(str(x) for x in PAPER_ABLATIONS[label])
        lines.append(f"| paper {label} | {paper} | | | |")
        for tag, group in (("code weights", code_group), ("**paper weights**", pw_group)):
            if group not in real or group not in strokes:
                continue
            cols = cell_columns(real, strokes, conv, group, "abs_h")
            runs = conv.get(group, {}).values()
            mirror = [v["raw"]["d_l1_mirror"] for v in runs]
            neg_h = [100 * v["raw"]["h_negative"] for v in runs]
            neg_w = [100 * v["raw"]["w_negative"] for v in runs]
            lines.append(
                f"| {tag} {label} (n = {len(cols[0])}) | "
                + " | ".join(fmt_ms(c, 3) for c in cols)
                + f" | {fmt_ms(mirror, 3)} | {fmt_ms(neg_h, 0)} % | {fmt_ms(neg_w, 0)} % |"
            )
    return lines


def build_report(real: dict, strokes: dict, conv: dict) -> str:
    tables = {
        (side, variant): ratios(real, strokes, conv, side, variant)
        for side in SIDES
        for variant in VARIANTS
    }
    complete = len(tables[("paper", "fold")]) == len(PW_TERMS)
    parts = [
        "Same protocol as T1c (`\\|h\\|`, real images and 4,096 synthesized patches, mean over 3 seeds); the *code weights* side is the "
        "stored T-016 / T-014 measurement, the *paper weights* side the 15 runs of T-021. Per-run values: "
        "`docs/experiments/data/pt_reproduction/` and `docs/experiments/data/pt_convention/eval.json`.",
        "",
        "**Absolute values** (mean ± std over seeds; `D_L1` with `\\|h\\|`; `D_L1 mirror` = mirror-invariant, raw predictions; `h < 0`, `w < 0` = share of matched strokes with a negative sign):",
        "",
        *absolute_table(real, strokes, conv),
        "",
        "**Effect of removing a term** (ablated / full; `*` = significantly worse by the T-017 rule; `D_L1` with `\\|w\\|`, `\\|h\\|` folded):",
        "",
        "| removed term | weights | " + " | ".join(COLUMNS) + " | cells in the paper's direction |",
        "|---|---|" + "---|" * (len(COLUMNS) + 1),
    ]
    for term in PW_TERMS:
        paper = " | ".join(f"{x:.2f}" for x in paper_ratio(term))
        parts.append(f"| {ABLATIONS[term]} | paper | {paper} | 5 / 5 |")
        for side in SIDES:
            row = tables[(side, "fold")].get(term)
            if row is None:
                continue
            cells = [
                f"{r:.2f}{'*' if significant(r, t) else ''}"
                for r, t in zip(row["ratio"], row["t"], strict=True)
            ]
            n = sum(direction(r) for r in row["ratio"])
            parts.append(f"| | {side} weights | " + " | ".join(cells) + f" | {n} / 5 |")
    parts += [
        "",
        "**Cells in the paper's direction (of 20)** and cells significantly worse:",
        "",
        "| `D_L1` flavour | code weights | paper weights | significantly worse (code / paper) |",
        "|---|---|---|---|",
    ]
    primary = {"code": 0, "paper": 0}
    for variant, label in VARIANTS.items():
        c = counts(tables[("code", variant)])
        p = counts(tables[("paper", variant)])
        if variant == "fold":
            primary = {"code": c[0], "paper": p[0]}
        parts.append(f"| {label} | {c[0]} | {p[0]} | {c[1]} / {p[1]} |")
    parts += [
        "",
        "**Per-term verdicts** (T-017 rule: *matters* = significantly worse on at least 3 of 5 columns; `\\|w\\|`, `\\|h\\|` fold):",
        "",
        "| removed term | code weights | paper weights | paper |",
        "|---|---|---|---|",
    ]
    for term in PW_TERMS:
        cells = []
        for side in SIDES:
            row = tables[(side, "fold")].get(term)
            if row is None:
                cells.append("n/a")
                continue
            worse = sum(significant(r, t) for r, t in zip(row["ratio"], row["t"], strict=True))
            cells.append(f"{verdict(worse)} ({worse} / 5)")
        parts.append(f"| {ABLATIONS[term]} | {cells[0]} | {cells[1]} | matters |")
    if complete:
        parts += [
            "",
            f"**Verdict on H** (decision rule of the protocol, primary count): code weights {primary['code']} of 20, "
            f"paper weights {primary['paper']} of 20 -> **{hypothesis(primary['code'], primary['paper'])}**.",
        ]
        if primary["code"] != 7:
            parts.append(
                f"(The counting rule gives {primary['code']} cells for the code weights, T-016 reported 7.)"
            )
    else:
        parts += ["", "*Incomplete: not every paper-weights group is evaluated yet.*"]
    return "\n".join(parts)


def report() -> None:
    text = build_report(
        _stored("table1_real"), _stored("table1_strokes"), pt_convention_report.load_store()
    )
    write_block("PAPER_WEIGHTS", text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["train", "evaluate", "report"])
    parser.add_argument(
        "--names", nargs="*", default=list(PW_GROUPS.values()), choices=list(REMOVE)
    )
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args(argv)
    {"train": lambda: train(args.names, args.parallel), "evaluate": evaluate, "report": report}[
        args.stage
    ]()


if __name__ == "__main__":
    main(sys.argv[1:])
