"""The circularity rule of the evaluation protocol (docs/design/evaluation.md 4.2).

The evaluation saliency model (InSPyReNet) may only be used by the evaluation code and its
scripts, never by the method, the renderer, the stroke models, the contribution (M4)
or the animation."""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = re.compile(r"inspyrenet|eval\.saliency|eval import saliency", re.IGNORECASE)
METHOD_PACKAGES = (
    "baselines", "contribution", "renderer", "stroke_model", "animation", "generation", "utils",
)  # fmt: skip
ALLOWED_SCRIPTS = {"fetch_inspyrenet.py", "compute_saliency.py"}


def test_no_method_code_uses_the_evaluation_saliency_model() -> None:
    offenders = []
    for package in METHOD_PACKAGES:
        for path in Path("brushstroke", package).rglob("*.py"):
            if FORBIDDEN.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path))
    assert offenders == [], f"the evaluation saliency model is reserved for evaluation: {offenders}"


def test_only_the_saliency_scripts_mention_the_evaluation_saliency_model() -> None:
    offenders = []
    for path in Path("scripts").glob("*.py"):
        if path.name not in ALLOWED_SCRIPTS and FORBIDDEN.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == [], (
        f"scripts outside the saliency tooling use the evaluation model: {offenders}"
    )
