"""The evaluation runner (T-026): one record per (method, image, seed), controls per image.

Spec: ``docs/design/evaluation.md`` sections 2 to 7. For every image the runner computes once the controls (blank black, blank
white, blur) and the blank-canvas reference ``q_ref`` of section 3.2; for every (method, image, seed) it runs the producer under
the measurement rules (:func:`brushstroke.eval.producers.measured_paint`), scores every canvas with every protocol metric
(:func:`brushstroke.eval.metrics.canvas_metrics`) and derives the AUCs. Records are JSON files, written atomically and never
overwritten unless asked (a run can be resumed). A failure (an exception, a NaN canvas) is recorded as such, never skipped.
Images are processed in sorted id order and every random source is seeded; a record's ``content_hash`` covers everything but
the timings, so two runs of a deterministic producer have the same hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from brushstroke.eval import curves
from brushstroke.eval.metrics import canvas_metrics
from brushstroke.eval.producers import Producer, measured_paint, other_gpu_memory_gb
from brushstroke.eval.protocol import Protocol

SCHEMA_VERSION = 1


@dataclass
class ImageContext:
    """One evaluation image: the target and the saliency maps of the target (never given to a producer)."""

    id: str
    split: str
    category: str
    stratum: str
    target: torch.Tensor  # (3, 512, 512) in [0, 1]
    saliency: dict[str, torch.Tensor] = field(default_factory=dict)


def blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian blur of ``(3, H, W)``, kernel radius ``ceil(3 sigma)``, reflect padding: the "no painting at all" control (2.2)."""
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    ax = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    k = torch.exp(-(ax**2) / (2 * sigma**2))
    k = k / k.sum()
    x = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return x[0]


def image_controls(ctx: ImageContext, proto: Protocol) -> dict[str, Any]:
    """The reference canvases of section 2.2 and the references of section 3.2 for one image.

    ``q_ref = max(LPIPS-Alex(black), LPIPS-Alex(white))``; ``sw_ref_<map>_<lambda>`` is the same for the saliency-weighted LPIPS
    (same weights), used to normalize the SW-LPIPS AUC (an implementation detail the protocol leaves open, see the amendment).
    """
    t = ctx.target
    canvases = torch.stack([torch.zeros_like(t), torch.ones_like(t), blur(t, proto.blur_sigma)])
    m = canvas_metrics(
        canvases, t, ctx.saliency, proto.sw_lambdas, proto.vgg_chunk, proto.low_res_size
    )
    controls = {
        name: {k: v[i] for k, v in m.items()}
        for i, name in enumerate(("blank_black", "blank_white", "blur"))
    }
    refs = {"q_ref": max(m["lpips_alex"][0], m["lpips_alex"][1])}
    for key in m:
        if key.startswith("sw_"):
            refs["ref_" + key] = max(m[key][0], m[key][1])
    return {"image_id": ctx.id, "controls": controls, "references": refs}


def environment() -> dict[str, Any]:
    """Facts stored in every record (section 6.8): commit, versions, GPU."""

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=20, check=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    gpu = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "cpu"
    return {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_dirty": run("git", "status", "--porcelain") != "",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "gpu": gpu,
        "driver": run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        if gpu != "cpu"
        else "none",
    }


def derive_auc(
    cps: Sequence[int], metrics: dict[str, list[float]], refs: dict[str, float], proto: Protocol
) -> dict[str, Any]:
    """AUCs of section 3 for the LPIPS-Alex curve and the SW-LPIPS curves, at every reachable budget.

    ``in_range`` records the check ``0 <= AUC <= 1`` (the harness logs a violation and never clips).
    """
    out: dict[str, Any] = {}
    for budget in proto.budgets:
        if not curves.reaches(cps, budget):
            continue
        entry: dict[str, Any] = {}
        for key, ref_key in [("lpips_alex", "q_ref")] + [
            (k, "ref_" + k)
            for k in metrics
            if k.startswith("sw_") and k.endswith(f"_{proto.headline_lambda:g}")
        ]:
            q = metrics[key]
            ref = refs[ref_key]
            norm = curves.auc_normalized(cps, q, budget, ref)
            entry[key] = {
                "raw": curves.auc_raw(cps, q, budget),
                "norm": norm,
                "log": curves.auc_log(cps, q, budget, proto.auc_log_start, ref),
                "late": curves.auc_late(cps, q, budget, proto.auc_late_start, ref)
                if budget > proto.auc_late_start
                else None,
                "in_range": bool(0.0 <= norm <= 1.0),
            }
        out[str(budget)] = entry
    return out


def content_hash(record: dict[str, Any]) -> str:
    """SHA-256 of everything in a record except the timings, the environment and the hash itself."""
    skip = {
        "runtime_s",
        "peak_vram_gb",
        "peak_vram_reserved_gb",
        "env",
        "content_hash",
        "contended_gpu_gb",
    }
    blob = json.dumps(
        {k: v for k, v in record.items() if k not in skip}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def run_one(
    producer: Producer,
    ctx: ImageContext,
    seed: int,
    proto: Protocol,
    controls: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    """One record: paint, score every canvas, derive the AUCs. Failures are recorded, not raised."""
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "method": producer.name,
        "image_id": ctx.id,
        "split": ctx.split,
        "seed": seed,
        "replicate_kind": producer.replicate_kind,
        "dagger_metrics": list(producer.dagger_metrics),
        "env": env,
        "status": "ok",
    }
    try:
        result = measured_paint(producer, ctx.target, proto.checkpoints, seed)
        metrics = canvas_metrics(
            result.canvases.to(ctx.target.device),
            ctx.target,
            ctx.saliency,
            proto.sw_lambdas,
            proto.vgg_chunk,
            proto.low_res_size,
        )
        refs = controls["references"]
        record.update(
            checkpoints=result.checkpoints,
            n_total=result.n_total,
            background=list(result.background),
            runtime_s=result.runtime_s,
            peak_vram_gb=result.peak_vram_bytes / 2**30,
            peak_vram_reserved_gb=result.peak_vram_reserved_bytes / 2**30,
            meta=result.meta,
            control=producer.control(),
            metrics=metrics,
            auc=derive_auc(result.checkpoints, metrics, refs, proto),
        )
    except (
        Exception
    ) as err:  # a failure is a result: recorded and counted, never skipped silently (section 7.7)
        record.update(status="failed", error=f"{type(err).__name__}: {err}")
    record["content_hash"] = content_hash(record)
    return record


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write (a crash never leaves a half record)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def record_path(out: Path, method: str, image_id: str, seed: int) -> Path:
    return out / "records" / method / f"{image_id}__s{seed}.json"


def run_harness(
    producers: Sequence[Producer],
    images: Iterable[ImageContext],
    seeds: dict[str, Sequence[int]],
    proto: Protocol,
    out: Path,
    force: bool = False,
    log: Callable[[str], None] = print,
    timed: bool = True,
) -> dict[str, int]:
    """Run every producer on every image for its seeds; resumable. Returns counts ``{"ran", "skipped", "failed"}``.

    Images are processed in sorted id order. When ``timed`` the run refuses to start if another process uses more than
    ``proto.gpu_contention_gb`` of VRAM (when the driver can tell); the check result is stored in ``out/environment.json``.
    """
    env = environment()
    other = other_gpu_memory_gb() if timed and torch.cuda.is_available() else None
    if other is not None and other > proto.gpu_contention_gb:
        raise RuntimeError(
            f"another process uses {other:.1f} GB of VRAM (limit {proto.gpu_contention_gb} GB): a timed run needs an idle GPU"
        )
    env["other_gpu_processes_gb"] = other
    write_json(out / "environment.json", env)
    counts = {"ran": 0, "skipped": 0, "failed": 0}
    for ctx in sorted(images, key=lambda c: c.id):
        controls_path = out / "controls" / f"{ctx.id}.json"
        if controls_path.exists() and not force:
            controls = json.loads(controls_path.read_text(encoding="utf-8"))
        else:
            controls = image_controls(ctx, proto)
            write_json(controls_path, controls)
        for producer in producers:
            for seed in seeds[producer.name]:
                path = record_path(out, producer.name, ctx.id, seed)
                if path.exists() and not force:
                    counts["skipped"] += 1
                    continue
                record = run_one(producer, ctx, seed, proto, controls, env)
                write_json(path, record)
                counts["ran" if record["status"] == "ok" else "failed"] += 1
                log(
                    f"{producer.name} {ctx.id} seed {seed}: {record['status']}"
                    + (
                        f" ({record['runtime_s']:.1f} s)"
                        if record["status"] == "ok"
                        else f" {record['error']}"
                    )
                )
    return counts
