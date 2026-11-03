"""The constants of the evaluation protocol, loaded from ``configs/eval/protocol.yaml`` (T-026).

Spec: ``docs/design/evaluation.md`` sections 3 to 6. Every number lives in the YAML file, none in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROTOCOL = Path("configs/eval/protocol.yaml")


@dataclass(frozen=True)
class Protocol:
    checkpoints: tuple[int, ...]
    budgets: tuple[int, ...]
    headline_budget: int
    auc_log_start: int
    auc_late_start: int
    sw_lambdas: tuple[float, ...]
    headline_lambda: float
    saliency_methods: tuple[str, ...]
    saliency_dir: str
    eval_set: str
    eval_root: str
    size: int
    low_res_size: int
    vgg_chunk: int
    blur_sigma: float
    seeds: tuple[int, ...]
    bootstrap: dict[str, Any]
    effect_threshold: float
    gpu_contention_gb: float
    endpoints: dict[str, str]
    tau: dict[str, Any]
    producers: dict[str, Any]
    max_stroke_count: int  # the last checkpoint: censoring point of the strokes-to-target

    def tau_value(self, budget: int) -> float | None:
        value = self.tau["values"].get(budget)
        return None if value is None else float(value)


def load_protocol(path: Path = PROTOCOL) -> Protocol:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cps = tuple(int(c) for c in raw["checkpoints"])
    budgets = tuple(int(b) for b in raw["budgets"])
    if list(cps) != sorted(set(cps)) or cps[0] != 0:
        raise ValueError("checkpoints must be strictly ascending and start at 0")
    if not set(budgets) <= set(cps):
        raise ValueError(f"budgets {budgets} must belong to the checkpoint grid")
    return Protocol(
        checkpoints=cps,
        budgets=budgets,
        headline_budget=int(raw["headline_budget"]),
        auc_log_start=int(raw["auc_log_start"]),
        auc_late_start=int(raw["auc_late_start"]),
        sw_lambdas=tuple(float(x) for x in raw["sw_lambdas"]),
        headline_lambda=float(raw["headline_lambda"]),
        saliency_methods=tuple(raw["saliency_methods"]),
        saliency_dir=str(raw["saliency_dir"]),
        eval_set=str(raw["eval_set"]),
        eval_root=str(raw["eval_root"]),
        size=int(raw["size"]),
        low_res_size=int(raw["low_res_size"]),
        vgg_chunk=int(raw["vgg_chunk"]),
        blur_sigma=float(raw["blur_sigma"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        bootstrap=dict(raw["bootstrap"]),
        effect_threshold=float(raw["effect_threshold"]),
        gpu_contention_gb=float(raw["gpu_contention_gb"]),
        endpoints=dict(raw["endpoints"]),
        tau=dict(raw["tau"]),
        producers=dict(raw["producers"]),
        max_stroke_count=cps[-1],
    )
