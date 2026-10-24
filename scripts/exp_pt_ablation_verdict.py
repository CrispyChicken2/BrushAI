"""Ordering of the loss terms and per-term verdicts of the ablations (T-017).

Applies the rules written in ``docs/experiments/pt_reproduction.md`` (section "T-017") to the stored
measurements of the T-016 protocol; no GPU. The block ``ABLATION_VERDICT`` of the doc is rewritten.

    python -m scripts.exp_pt_ablation_verdict report
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import sys

from scripts.exp_pt_table1 import ABLATIONS, _group_real, _group_strokes, _stored
from scripts.pt_repro_common import PAPER_ABLATIONS, fmt_ms, write_block

COLUMNS = ("L_pixel (real)", "L_pcpt (real)", "L_pixel (strokes)", "D_L1", "D_W")
T_CRITICAL = 2.13  # one-sided 5 % level, about 4 degrees of freedom (3 + 3 seeds)
TIE = 0.05  # importances closer than this are reported as tied
MATTERS = 3  # worse on at least this many columns = the term matters


def welch_t(ablated: list[float], full: list[float]) -> float:
    """Welch t statistic of ``mean(ablated) - mean(full)`` (``inf`` / ``-inf`` / 0 when both have
    no spread)."""
    diff = statistics.mean(ablated) - statistics.mean(full)
    var = statistics.variance(ablated) / len(ablated) + statistics.variance(full) / len(full)
    if var == 0:
        return math.copysign(math.inf, diff) if diff else 0.0
    return diff / math.sqrt(var)


def verdict(worse: int) -> str:
    return "matters" if worse >= MATTERS else ("partly" if worse >= 1 else "not reproduced")


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall tau-a between two score vectors (1 = same ordering, -1 = reversed)."""
    pairs = list(itertools.combinations(range(len(a)), 2))
    score = sum(
        math.copysign(1, (a[i] - a[j]) * (b[i] - b[j])) if (a[i] - a[j]) * (b[i] - b[j]) else 0
        for i, j in pairs
    )
    return score / len(pairs)


def columns(real: dict, strokes: dict, group: str, fold: bool) -> list[list[float]]:
    """The five per-seed value lists of a group; ``fold``: D_L1 with ``|w|``, ``|h|`` folded."""
    d_l1_mode = "abs_wh" if fold else "abs_h"
    return [
        _group_real(real, group, "l_pixel"),
        _group_real(real, group, "l_pcpt"),
        _group_strokes(strokes, group, "abs_h", "l_pixel"),
        _group_strokes(strokes, group, d_l1_mode, "d_l1"),
        _group_strokes(strokes, group, "abs_h", "d_w"),
    ]


def analyse(real: dict, strokes: dict, fold: bool) -> dict[str, dict[str, object]]:
    """Per term: ratios, worse flags, importance, verdict (ours) and the paper's ratios."""
    full = columns(real, strokes, "ours", fold)
    paper_full = PAPER_ABLATIONS["full"]
    out: dict[str, dict[str, object]] = {}
    for name, label in ABLATIONS.items():
        group = f"abl_{name}"
        if group not in real or group not in strokes:
            continue
        ablated = columns(real, strokes, group, fold)
        ratios = [
            statistics.mean(a) / statistics.mean(f) for a, f in zip(ablated, full, strict=True)
        ]
        t_stats = [welch_t(a, f) for a, f in zip(ablated, full, strict=True)]
        worse = [r > 1 and t >= T_CRITICAL for r, t in zip(ratios, t_stats, strict=True)]
        better = [r < 1 and t <= -T_CRITICAL for r, t in zip(ratios, t_stats, strict=True)]
        paper = [p / q for p, q in zip(PAPER_ABLATIONS[label], paper_full, strict=True)]
        out[label] = {
            "values": ablated,
            "ratios": ratios,
            "worse": worse,
            "better": better,
            "importance": statistics.mean(ratios),
            "verdict": verdict(sum(worse)),
            "paper_ratios": paper,
            "paper_importance": statistics.mean(paper),
        }
    return out


def ordering(importance: dict[str, float]) -> str:
    """Terms sorted by decreasing importance; terms within ``TIE`` of the previous one are joined
    with ``~`` instead of ``>``."""
    items = sorted(importance.items(), key=lambda kv: -kv[1])
    text = items[0][0]
    for (_, prev), (name, value) in itertools.pairwise(items):
        text += f" {'~' if prev - value < TIE else '>'} {name}"
    return text


def build_report(real: dict, strokes: dict) -> str:
    blocks = []
    for fold, title in (
        (False, "protocol numbers (`|h|`)"),
        (True, "D_L1 with `|w|`, `|h|` folded"),
    ):
        result = analyse(real, strokes, fold)
        if not result:
            continue
        lines = [
            f"**Per-term verdicts, {title}.** Cells: ablated / full ratio (`*` = significantly worse by the rule, `†` = significantly better without the term), paper ratio in brackets.",
            "",
            "| removed term | "
            + " | ".join(COLUMNS)
            + " | worse on | importance (ours / paper) | verdict (ours / paper) |",
            "|---|" + "---|" * (len(COLUMNS) + 3),
        ]
        for label, r in result.items():
            cells = [
                f"{x:.2f}{'*' if w else ('†' if b else '')} ({p:.2f})"
                for x, w, b, p in zip(  # type: ignore[call-overload]
                    r["ratios"], r["worse"], r["better"], r["paper_ratios"], strict=True
                )
            ]
            lines.append(
                f"| {label} | "
                + " | ".join(cells)
                + f" | {sum(r['worse'])} / 5 | {r['importance']:.2f} / {r['paper_importance']:.2f} | {r['verdict']} / matters |"  # type: ignore[arg-type]
            )
        ours = {k: v["importance"] for k, v in result.items()}
        paper = {k: v["paper_importance"] for k, v in result.items()}
        tau = kendall_tau(list(ours.values()), list(paper.values()))  # type: ignore[arg-type]
        top_ours = max(ours, key=ours.get)  # type: ignore[arg-type]
        top_paper = max(paper, key=paper.get)  # type: ignore[arg-type]
        lines += [
            "",
            f"- ordering, ours: {ordering(ours)}  (importances {', '.join(f'{k} {v:.2f}' for k, v in ours.items())})",  # type: ignore[misc]
            f"- ordering, paper: {ordering(paper)}",  # type: ignore[arg-type]
            f"- Kendall tau between the two importance vectors: {tau:+.2f}; most important term: ours {top_ours}, paper {top_paper} ({'same' if top_ours == top_paper else 'different'}).",
        ]
        blocks.append("\n".join(lines))
    spread = ", ".join(
        f"{label}: {fmt_ms(columns(real, strokes, f'abl_{name}', False)[0], 4)}"
        for name, label in ABLATIONS.items()
        if f"abl_{name}" in real
    )
    blocks.append(f"Real-image L_pixel per ablation (mean ± std over seeds): {spread}.")
    return "\n\n".join(blocks)


def report() -> None:
    text = build_report(_stored("table1_real"), _stored("table1_strokes"))
    write_block("ABLATION_VERDICT", text)
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["report"])
    parser.parse_args(argv)
    report()


if __name__ == "__main__":
    main(sys.argv[1:])
