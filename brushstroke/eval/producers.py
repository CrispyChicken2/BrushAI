"""The producer interface of the evaluation harness (T-026) and the adapters of the baselines.

Spec: ``docs/design/evaluation.md`` section 7. The harness never renders a stroke for a non-sequence producer: a producer
hands over **canvases** (the painting after exactly ``n`` strokes, ``n`` in the protocol grid), so that any stroke model is
evaluated by the same code. A producer receives the 512 x 512 target and a seed, nothing else (no image id, category, dev flag
or evaluation saliency map).

Adapters here: :class:`OptimProducer` (the optimization baseline, replicate = initialization seed) and :class:`PTProducer`
(the Paint Transformer, replicate = training seed, one checkpoint per seed, inference is deterministic). Both build a
:class:`~brushstroke.strokes.StrokeSequence` and take the canvases with ``render_progressive`` (the prefix of the sequence).
"""

from __future__ import annotations

import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch

from brushstroke.strokes import StrokeSequence


@dataclass
class PaintResult:
    """What a producer returns (section 7).

    ``checkpoints`` are the stroke counts of the canvases actually produced: the protocol grid cut at ``n_total`` (a
    producer cannot reach a budget above the number of strokes it emitted). ``canvases`` is ``(len(checkpoints), 3, 512, 512)``
    float32 in [0, 1], the canvas after exactly ``n`` strokes. ``runtime_s`` and the VRAM figures are measured by the harness.
    """

    checkpoints: list[int]
    canvases: torch.Tensor
    n_total: int
    background: tuple[float, float, float]
    runtime_s: float = 0.0
    peak_vram_bytes: int = 0
    peak_vram_reserved_bytes: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class Producer(Protocol):
    """A stroke model seen by the harness."""

    name: str
    stochastic: bool
    replicate_kind: str  # "init seed" | "training seed" | "none"
    dagger_metrics: tuple[
        str, ...
    ]  # metrics that are in the method's own loss or selection (section 2.1)

    def load(self, seed: int) -> None:
        """Load whatever the replicate ``seed`` needs (weights...). Called before the timed region, never timed."""

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> PaintResult:
        """Paint the 512 x 512 ``target`` (``(3, 512, 512)`` in [0, 1], on the producer's device)."""

    def control(self) -> dict[str, Any]:
        """Optional extra facts about the last painting, computed after the timed region (section 2.2 control rows)."""


def usable_checkpoints(grid: Sequence[int], n_total: int) -> list[int]:
    """The protocol checkpoints a producer that emitted ``n_total`` strokes can reach."""
    return [int(c) for c in grid if c <= n_total]


def prefix_canvases(
    sequence: StrokeSequence, grid: Sequence[int], render_cfg: Any
) -> tuple[list[int], torch.Tensor]:
    """Canvases after the first ``n`` strokes for the reachable checkpoints, in one pass (``render_progressive``).

    The canvas at ``n`` is exactly ``sequence[:n].render()`` (checked by a test): nothing is re-optimized or recomposed.
    """
    cps = usable_checkpoints(grid, len(sequence))
    return cps, sequence.render_progressive(cps, cfg=render_cfg).clamp(0.0, 1.0)


def other_gpu_memory_gb() -> float | None:
    """VRAM used by compute processes other than this one, in GB (``nvidia-smi``), or ``None`` if it cannot be told.

    Windows (WDDM) often does not report per-process memory; the harness then records that the check was unavailable.
    """
    import os

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total_mb = 0.0
    for line in out.strip().splitlines():
        pid, _, mem = line.partition(",")
        if not pid.strip().isdigit() or int(pid) == os.getpid():
            continue
        if not mem.strip().replace(".", "", 1).isdigit():
            return None
        total_mb += float(mem)
    return total_mb / 1024.0


def measured_paint(
    producer: Producer, target: torch.Tensor, grid: Sequence[int], seed: int
) -> PaintResult:
    """Run a producer under the measurement rules of section 2.3 and validate what it returns.

    ``torch.cuda.synchronize()`` before and after, peak memory reset before; the producer's ``load`` is called first and is not
    timed; metric computation is outside. Raises ``ValueError`` for a result that breaks the contract (a NaN canvas, a wrong shape).
    """
    cuda = target.device.type == "cuda"
    producer.load(seed)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = producer.paint(target, grid, seed)
    if cuda:
        torch.cuda.synchronize()
    result.runtime_s = time.perf_counter() - t0
    if cuda:
        result.peak_vram_bytes = int(torch.cuda.max_memory_allocated())
        result.peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved())
    validate_result(result, grid, target.shape[-1])
    return result


def validate_result(result: PaintResult, grid: Sequence[int], size: int) -> None:
    """The contract of section 7: shapes, the grid, no NaN, ascending checkpoints starting at 0, enough strokes."""
    cps = result.checkpoints
    if cps != usable_checkpoints(grid, result.n_total) or not cps or cps[0] != 0:
        raise ValueError(
            f"checkpoints {cps} are not the protocol grid cut at n_total = {result.n_total}"
        )
    if tuple(result.canvases.shape) != (len(cps), 3, size, size):
        raise ValueError(
            f"canvases have shape {tuple(result.canvases.shape)}, expected {(len(cps), 3, size, size)}"
        )
    if not torch.isfinite(result.canvases).all():
        raise ValueError("a canvas holds NaN or infinite values")
    if float(result.canvases.min()) < -1e-4 or float(result.canvases.max()) > 1.0 + 1e-4:
        raise ValueError("canvases must lie in [0, 1]")


class OptimProducer:
    """Baseline (a): optimization-only progressive stroke decomposition (``configs/baseline_optim.yaml``)."""

    name = "optim"
    stochastic = True
    replicate_kind = "init seed"
    dagger_metrics = (
        "l1",
        "lpips_vgg",
    )  # the loss of the baseline: 1.0 L1 + 0.1 LPIPS-VGG (section 2.1)

    def __init__(self, config: str | Path, device: str | torch.device = "cuda") -> None:
        from brushstroke.baselines.optim.config import PaintOptimConfig
        from brushstroke.utils.config import load_config

        self.cfg = load_config(PaintOptimConfig, Path(config), [])
        self.device = torch.device(device)
        self._last: StrokeSequence | None = None

    def load(self, seed: int) -> None:
        """Nothing to load: the brush bank is built inside the run."""

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> PaintResult:
        from brushstroke.baselines.optim.optimize import paint_optim
        from brushstroke.utils.seed import seed_everything

        cfg = self.cfg
        seed_everything(seed, cfg.deterministic, cfg.cudnn_deterministic)
        seq = paint_optim(target, cfg.optim, cfg.loss, cfg.brush, cfg.render, seed, self.device)
        self._last = seq
        cps, canvases = prefix_canvases(seq, checkpoints, cfg.render)
        return PaintResult(
            checkpoints=cps,
            canvases=canvases.to(target.device),
            n_total=len(seq),
            background=tuple(float(c) for c in cfg.optim.background),  # type: ignore[arg-type]
            meta={
                "config_hash": seq.meta.get("config_hash"),
                "seed": seed,
                "replicate_kind": self.replicate_kind,
                "native_resolution": [cfg.optim.height, cfg.optim.width],
                "upsampled": False,
                "internal_runtime_s": seq.meta.get("runtime_seconds"),
                "internal_peak_vram_mb": seq.meta.get("peak_vram_mb"),
            },
        )

    def control(self) -> dict[str, Any]:
        return {}


class PTProducer:
    """Paint Transformer inference on a trained checkpoint (one per training seed), evaluated as its StrokeSequence."""

    name = "pt"
    stochastic = False  # inference is deterministic; the replicate is the training seed
    replicate_kind = "training seed"
    dagger_metrics: tuple[str, ...] = ()

    def __init__(
        self,
        checkpoints: dict[int, str],
        config: str | Path = "configs/paint_pt.yaml",
        device: str | torch.device = "cuda",
    ) -> None:
        from brushstroke.stroke_model.inference import PaintPTConfig
        from brushstroke.utils.config import load_config

        self.cfg = load_config(PaintPTConfig, Path(config), [])
        self.checkpoints = {int(k): str(v) for k, v in checkpoints.items()}
        self.device = torch.device(device)
        self._net: Any = None
        self._loaded_seed: int | None = None
        self._brushes: torch.Tensor | None = None
        self._last: tuple[Any, torch.Tensor] | None = None

    def load(self, seed: int) -> None:
        from brushstroke.renderer.pt import load_pt_brushes
        from brushstroke.stroke_model.inference import load_pt_model

        if seed not in self.checkpoints:
            raise KeyError(f"no Paint Transformer checkpoint for the training seed {seed}")
        if self._loaded_seed != seed:
            self._net, _ = load_pt_model(self.checkpoints[seed], self.device)
            self._brushes = load_pt_brushes("inference", self.cfg.inference.brush_dir, self.device)
            self._loaded_seed = seed

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> PaintResult:
        from brushstroke.stroke_model.inference import paint

        assert self._brushes is not None
        result = paint(self._net, target, self._brushes, self.cfg.inference, self.cfg.refine)
        seq = result.sequence
        cps, canvases = prefix_canvases(seq, checkpoints, self.cfg.render)
        self._last = (result, canvases[-1:] if len(cps) else canvases)
        return PaintResult(
            checkpoints=cps,
            canvases=canvases.to(target.device),
            n_total=len(seq),
            background=tuple(float(c) for c in seq.background),  # type: ignore[arg-type]
            meta={
                "checkpoint": self.checkpoints[seed],
                "seed": seed,
                "replicate_kind": self.replicate_kind,
                "native_resolution": [512, 512],
                "upsampled": False,
                "final_layer": self.cfg.inference.final_layer,
                "refine": bool(self.cfg.refine.enabled),
            },
        )

    def control(self) -> dict[str, Any]:
        """The sequence-render control (section 2.2): the official window pipeline's canvas against the generic render of the sequence."""
        if self._last is None:
            return {}
        result, _ = self._last
        generic = result.sequence.render(cfg=self.cfg.render).to(result.canvas.device)
        diff = (generic - result.canvas).abs()
        return {
            "pipeline_vs_sequence_mean_abs": float(diff.mean()),
            "pipeline_vs_sequence_max_abs": float(diff.max()),
        }


def nan_safe(x: float) -> float | None:
    """JSON-friendly float: ``None`` for NaN or infinity."""
    return float(x) if math.isfinite(x) else None
