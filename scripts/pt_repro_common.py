"""Shared helpers of the reproduction scripts of T-016 (paper numbers, JSON store, report blocks)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

DATA = Path("docs/experiments/data/pt_reproduction")
DOC = Path("docs/experiments/pt_reproduction.md")
FIGURES = Path("docs/media")
IMAGES = ["mona_lisa", "starry_night", "great_wave", "pearl_earring", "earth_apollo17"]
OFFICIAL_WEIGHTS = Path("data/raw/pt_official/model.pth")

# Paper numbers (Liu et al., ICCV 2021, Tables 1 and 2) as extracted in docs/design/paint_transformer.md
# section 7. Table 1, real images: (L_pixel, L_pcpt); strokes: (L_pixel, D_L1, D_W).
PAPER_REAL = {"RL": (0.040, 0.737), "Optim": (0.059, 0.856), "PT": (0.056, 0.807)}
PAPER_STROKES = {
    "RL": (0.058, None, None),
    "Optim": (0.073, 0.137, 0.057),
    "PT": (0.042, 0.083, 0.018),
}
PAPER_ABLATIONS = {  # real L_pixel, real L_pcpt, stroke L_pixel, D_L1, D_W
    "full": (0.056, 0.807, 0.042, 0.083, 0.018),
    "w/o L_pixel": (0.081, 1.012, 0.068, 0.241, 0.024),
    "w/o D_L1": (0.074, 0.941, 0.077, 0.267, 0.019),
    "w/o D_W": (0.069, 0.947, 0.046, 0.113, 0.034),
    "w/o D_bce": (0.071, 0.928, 0.052, 0.093, 0.021),
}
# T-021: the ablations re-run with the paper's loss weights; group "pw_full" / "pw_abl_<term>",
# checkpoints outputs/pt_pw/<name>/seed<k>/latest.pt (name "full" = no term removed)
PW_ROOT = Path("outputs/pt_pw")
PW_TERMS = ("no_pixel", "no_dl1", "no_dw", "no_dbce")
PW_GROUPS = {"pw_full": "full", **{f"pw_abl_{t}": t for t in PW_TERMS}}  # group -> run name
PAPER_TIME = {  # seconds at 128 / 256 / 512
    "PT": (0.055, 0.124, 0.304),
    "RL": (0.242, 0.266, 0.322),
    "Optim": (76.5, 161.6, 521.4),
}
PAPER_TRAIN_HOURS = {"PT": 3.79, "RL": 40.0, "Optim": 10.16}


def dump(name: str, payload: object) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8", newline="\n"
    )


def load(name: str) -> Any:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def fmt_ms(values: list[float], digits: int = 3) -> str:
    """``mean ± std`` of a list (``mean`` alone for one value)."""
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}" if len(values) > 1 else f"{m:.{digits}f}"


def write_block(name: str, text: str, doc: Path = DOC) -> None:
    """Replace (or append) the block ``<!-- BEGIN name --> ... <!-- END name -->`` of ``doc``."""
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    block = f"{begin}\n\n{text.strip()}\n\n{end}"
    body = doc.read_text(encoding="utf-8")
    if begin in body:
        head, rest = body.split(begin, 1)
        body = head + block + rest.split(end, 1)[1]
    else:
        body = body.rstrip("\n") + "\n\n" + block + "\n"
    doc.write_text(body, encoding="utf-8", newline="\n")
