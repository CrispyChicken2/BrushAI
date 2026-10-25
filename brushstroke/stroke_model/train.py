"""Training pipeline of the Paint Transformer stroke predictor (study note sections 2, 4, 5).

The schedule follows the official ``train.sh`` (Liu et al., ICCV 2021, repository
``Huage001/PaintTransformer`` at commit 7960b8e): Adam (lr 1e-4, betas (0.5, 0.999)), batch 64, 220
epochs of 256 steps (56,320 steps = 3.60 M samples), learning rate constant for 204 epochs and then
multiplied by 0.3 every 5 epochs. Data are synthesized on the GPU at each step
(:class:`~brushstroke.stroke_model.synthetic.SyntheticStrokeGenerator`).

Beyond the official recipe: a fixed-seed synthetic validation set, CSV logs (optional TensorBoard),
optional bf16 autocast, seeded checkpointing and *exact* resume (model, optimizer, scheduler, the
data generator, global RNG states and the step are restored), and a per-phase step profiler.

The training stream is determined by ``seed``; the validation set by ``val.seed`` only.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.losses import PaintTransformerLoss, PTLossConfig
from brushstroke.stroke_model.model import (
    PaintTransformer,
    PTModelConfig,
    count_parameters,
    init_official_weights,
)
from brushstroke.stroke_model.synthetic import PatchBatch, SyntheticConfig, SyntheticStrokeGenerator
from brushstroke.utils.config import dump_config
from brushstroke.utils.device import get_device, peak_vram_mb, reset_peak_vram
from brushstroke.utils.seed import seed_everything

LOSS_TERMS = ("total", "pixel", "gt", "w", "decision")
PRECISIONS = ("fp32", "bf16")
BN_MODES = ("running", "recalibrate", "batch")


@dataclass
class OptimConfig:
    """Adam settings (official: lr 1e-4, betas (0.5, 0.999), no weight decay, no clipping)."""

    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # max gradient norm; 0 = off (official)


@dataclass
class ScheduleConfig:
    """Official lr policy: constant for `n_epochs`, then x `decay_gamma` every `decay_every`."""

    steps_per_epoch: int = 256  # max_dataset_size 16384 / batch 64
    n_epochs: int = 200
    n_epochs_decay: int = 20
    decay_every: int = 5  # epochs
    decay_gamma: float = 0.3
    max_steps: int = (
        0  # stop after this many steps (0 = the full schedule); does not alter the lr curve
    )

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * (self.n_epochs + self.n_epochs_decay)

    def lr_factor(self, step: int) -> float:
        """Multiplier of the base lr for the optimizer step number ``step`` (0-based)."""
        epoch = step // self.steps_per_epoch  # 0-based; official epoch number is epoch + 1
        return float(self.decay_gamma ** max(0, (epoch + 1 - self.n_epochs) // self.decay_every))


@dataclass
class ValConfig:
    """Held-out synthetic validation batches (generated once from ``seed``)."""

    seed: int = 12345
    batches: int = 8
    every: int = 1000  # steps; 0 = only at the start and the end
    # BatchNorm statistics used to score the validation set (T-020, design note section "BatchNorm
    # in validation"): running = the running averages of training (eval mode; they can lag behind
    # the weights and make isolated spikes), recalibrate = statistics re-estimated on
    # `bn_batches` fixed batches of the training distribution, batch = the statistics of each
    # validation batch. The last two work on a copy of the network.
    bn: str = "running"
    bn_batches: int = 20
    bn_seed: int = 54321  # seed of the calibration batches (a stream apart from the training one)


@dataclass
class TrainConfig:
    """Root config of ``scripts/train_pt.py`` (``configs/stroke_model/pt.yaml``)."""

    seed: int = 0
    deterministic: bool = False  # force deterministic cuDNN kernels (slower)
    device: str = "auto"
    output_dir: str = "outputs/pt_train"
    brush_dir: str = "data/raw/pt_brush"
    batch_size: int = 64
    precision: str = "fp32"  # fp32 | bf16 (autocast of the network; losses stay in fp32)
    log_every: int = 25
    checkpoint_every: int = 2000
    resume: str = ""  # "" = fresh run, "auto" = <output_dir>/latest.pt if present, or a path
    tensorboard: bool = False
    model: PTModelConfig = field(default_factory=PTModelConfig)
    loss: PTLossConfig = field(default_factory=PTLossConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    val: ValConfig = field(default_factory=ValConfig)


class _Context:
    """Everything a run needs: device, brushes, generators, network, loss, optimizer."""

    def __init__(self, cfg: TrainConfig) -> None:
        if cfg.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {cfg.precision!r}")
        if cfg.val.bn not in BN_MODES:
            raise ValueError(f"val.bn must be one of {BN_MODES}, got {cfg.val.bn!r}")
        if cfg.batch_size % 4 != 0:
            raise ValueError("batch_size must be a multiple of 4 (synthetic data)")
        self.cfg = cfg
        self.device = get_device(cfg.device)
        seed_everything(cfg.seed, cudnn_deterministic=cfg.deterministic)
        brushes = load_pt_brushes("train", cfg.brush_dir, self.device)
        self.generator = SyntheticStrokeGenerator(brushes, cfg.synthetic)
        self.data_rng = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.net = PaintTransformer(cfg.model).to(self.device)
        init_official_weights(self.net)
        self.loss = PaintTransformerLoss(brushes, cfg.loss, cfg.model.colour_mode)
        o = cfg.optim
        self.optimizer = Adam(
            self.net.parameters(),
            lr=o.lr,
            betas=(o.beta1, o.beta2),
            weight_decay=o.weight_decay,
        )
        self.scheduler = LambdaLR(self.optimizer, cfg.schedule.lr_factor)
        self.step = 0

    def forward(
        self, batch: PatchBatch, net: PaintTransformer | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Network predictions in fp32 (autocast of the network only when ``precision: bf16``);
        ``net`` replaces the training network (a validation copy)."""
        enabled = self.cfg.precision == "bf16"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=enabled):
            shape, colour, logits = (net or self.net).strokes(batch.target, batch.canvas)
        return shape.float(), colour.float(), logits.float()

    def train_step(self, batch: PatchBatch) -> dict[str, float]:
        """One optimization step; returns the detached loss terms."""
        self.net.train()
        out = self.loss(*self.forward(batch), batch)
        self.optimizer.zero_grad(set_to_none=True)
        out.total.backward()  # type: ignore[no-untyped-call]
        if self.cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.optim.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {k: float(getattr(out, k).detach()) for k in LOSS_TERMS}


# ----------------------------------------------------------------------------- validation


def make_validation_set(ctx: _Context) -> list[PatchBatch]:
    """Fixed held-out batches, a function of ``val.seed`` only (never of the training stream)."""
    rng = torch.Generator(device=ctx.device).manual_seed(ctx.cfg.val.seed)
    return [ctx.generator.sample(ctx.cfg.batch_size, rng) for _ in range(ctx.cfg.val.batches)]


def make_calibration_set(ctx: _Context) -> list[PatchBatch]:
    """``val.bn_batches`` fixed batches of the training distribution for the BatchNorm
    recalibration, a function of ``val.bn_seed`` only: neither the training stream nor any global
    random state is consumed."""
    v = ctx.cfg.val
    rng = torch.Generator(device=ctx.device).manual_seed(v.bn_seed)
    return [ctx.generator.sample(ctx.cfg.batch_size, rng) for _ in range(v.bn_batches)]


def _batch_norms(net: torch.nn.Module) -> list[torch.nn.modules.batchnorm._BatchNorm]:
    return [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def validation_network(
    ctx: _Context, mode: str, calibration: list[PatchBatch] | None = None
) -> PaintTransformer:
    """The network scored on the validation set, in eval mode.

    ``running``: the training network itself. ``recalibrate`` and ``batch``: a deep copy (the
    training state, running statistics included, is never touched) in which only the BatchNorm
    layers use batch statistics (dropout stays off); ``recalibrate`` first resets them and
    averages the statistics of the ``calibration`` batches (cumulative average, ``momentum=None``)
    and then uses those averages, ``batch`` uses the statistics of each batch it is applied to.
    """
    if mode not in BN_MODES:
        raise ValueError(f"mode must be one of {BN_MODES}, got {mode!r}")
    ctx.net.eval()
    if mode == "running":
        return ctx.net
    net = copy.deepcopy(ctx.net).eval()
    norms = _batch_norms(net)
    if mode == "batch":
        for m in norms:
            m.train()
            m.track_running_stats = False  # normalize with the batch, update nothing
        return net
    if not calibration:
        raise ValueError("mode 'recalibrate' needs calibration batches")
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the calibration batches
        m.train()
    for batch in calibration:
        ctx.forward(batch, net)
    for m in norms:
        m.eval()
    return net


@torch.no_grad()
def evaluate(
    ctx: _Context,
    batches: list[PatchBatch],
    mode: str | None = None,
    calibration: list[PatchBatch] | None = None,
) -> dict[str, float]:
    """Mean loss terms over the validation batches (eval mode, BatchNorm as ``mode``, default
    ``val.bn``) and the mean absolute error of the number of confident strokes (``logit > 0``)
    against the number of valid ground-truth strokes."""
    was_training = ctx.net.training
    mode = mode or ctx.cfg.val.bn
    if mode == "recalibrate" and calibration is None:
        calibration = make_calibration_set(ctx)
    net = validation_network(ctx, mode, calibration)
    sums = dict.fromkeys((*LOSS_TERMS, "count_error"), 0.0)
    for batch in batches:
        shape, colour, logits = ctx.forward(batch, net)
        out = ctx.loss(shape, colour, logits, batch)
        for k in LOSS_TERMS:
            sums[k] += float(getattr(out, k))
        count = (logits > 0).sum(dim=1).float()
        sums["count_error"] += float((count - batch.gt_decision.sum(dim=1)).abs().mean())
    ctx.net.train(was_training)
    return {k: v / len(batches) for k, v in sums.items()}


# ----------------------------------------------------------------------------- checkpoints


def rng_state(ctx: _Context) -> dict[str, Any]:
    """All random state that influences the rest of the run."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data": ctx.data_rng.get_state(),
    }


def set_rng_state(ctx: _Context, state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    ctx.data_rng.set_state(state["data"])


def save_checkpoint(ctx: _Context, path: Path) -> None:
    """Atomically write the full training state to ``path``."""
    state = {
        "step": ctx.step,
        "model": ctx.net.state_dict(),
        "optimizer": ctx.optimizer.state_dict(),
        "scheduler": ctx.scheduler.state_dict(),
        "rng": rng_state(ctx),
        "config": asdict(ctx.cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(ctx: _Context, path: Path) -> int:
    """Restore the training state; returns the step."""
    # own checkpoints only: they hold RNG states and the config, which are not plain tensors
    state = torch.load(path, map_location=ctx.device, weights_only=False)
    ctx.net.load_state_dict(state["model"])
    ctx.optimizer.load_state_dict(state["optimizer"])
    ctx.scheduler.load_state_dict(state["scheduler"])
    rng = state["rng"]
    rng = {**rng, "torch": rng["torch"].cpu(), "cuda": [s.cpu() for s in rng["cuda"]]}
    rng["data"] = rng["data"].cpu()
    set_rng_state(ctx, rng)
    ctx.step = int(state["step"])
    return ctx.step


# ----------------------------------------------------------------------------- logging


class _CsvLog:
    """Append-only CSV; on resume, rows after the restored step are dropped."""

    def __init__(self, path: Path, fields: list[str], keep_up_to: int | None) -> None:
        self.path, self.fields = path, fields
        rows: list[dict[str, str]] = []
        if keep_up_to is not None and path.exists():
            with path.open(newline="", encoding="utf-8") as fh:
                rows = [r for r in csv.DictReader(fh) if int(r["step"]) <= keep_up_to]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, row: dict[str, float | int]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fields).writerow(row)


@contextmanager
def _tensorboard(cfg: TrainConfig, out: Path) -> Iterator[Any]:
    if not cfg.tensorboard:
        yield None
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("tensorboard: true needs the `tensorboard` package") from exc
    writer = SummaryWriter(str(out / "tb"))
    try:
        yield writer
    finally:
        writer.close()


# ----------------------------------------------------------------------------- main loop


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run (or resume) a training; returns the final summary (also written to ``summary.json``).

    Outputs in ``cfg.output_dir``: ``config.resolved.yaml``, ``train.csv``, ``val.csv``,
    ``latest.pt`` (checkpoint, overwritten), ``summary.json``.
    """
    ctx = _Context(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, out)
    resume = out / "latest.pt" if cfg.resume == "auto" else Path(cfg.resume)
    resumed = bool(cfg.resume) and resume.exists()
    start = load_checkpoint(ctx, resume) if resumed else 0
    keep = start if resumed else None
    train_log = _CsvLog(
        out / "train.csv",
        ["step", "epoch", "lr", *LOSS_TERMS, "step_ms", "peak_vram_mb"],
        keep,
    )
    val_log = _CsvLog(out / "val.csv", ["step", *LOSS_TERMS, "count_error"], keep)
    val_set = make_validation_set(ctx)
    calibration = make_calibration_set(ctx) if cfg.val.bn == "recalibrate" else None
    last_step = min(cfg.schedule.total_steps, cfg.schedule.max_steps or cfg.schedule.total_steps)
    sch = cfg.schedule
    reset_peak_vram()
    history: list[dict[str, float]] = []
    window: list[dict[str, float]] = []
    window_start = time.perf_counter()

    def validate(tb: Any) -> None:
        metrics = evaluate(ctx, val_set, calibration=calibration)
        val_log.write({"step": ctx.step, **metrics})
        history.append({"step": float(ctx.step), **metrics})
        if tb is not None:
            for k, v in metrics.items():
                tb.add_scalar(f"val/{k}", v, ctx.step)

    with _tensorboard(cfg, out) as tb:
        if not resumed:
            validate(tb)
        while ctx.step < last_step:
            batch = ctx.generator.sample(cfg.batch_size, ctx.data_rng)
            window.append(ctx.train_step(batch))
            if ctx.step % cfg.log_every == 0 or ctx.step == last_step:
                if ctx.device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                mean = {k: sum(w[k] for w in window) / len(window) for k in LOSS_TERMS}
                row = {
                    "step": ctx.step,
                    "epoch": ctx.step / sch.steps_per_epoch,
                    "lr": float(ctx.scheduler.get_last_lr()[0]),
                    **mean,
                    "step_ms": 1000 * (now - window_start) / len(window),
                    "peak_vram_mb": peak_vram_mb(),
                }
                train_log.write(row)
                if tb is not None:
                    for k in LOSS_TERMS:
                        tb.add_scalar(f"train/{k}", mean[k], ctx.step)
                window, window_start = [], time.perf_counter()
            if cfg.val.every and ctx.step % cfg.val.every == 0 and ctx.step != last_step:
                validate(tb)
            if ctx.step % cfg.checkpoint_every == 0 and ctx.step != last_step:
                save_checkpoint(ctx, out / "latest.pt")
        if not history or history[-1]["step"] != ctx.step:
            validate(tb)
        save_checkpoint(ctx, out / "latest.pt")
    summary = {
        "steps": ctx.step,
        "parameters": count_parameters(ctx.net),
        "precision": cfg.precision,
        "first_val": history[0] if history else None,
        "last_val": history[-1],
        "peak_vram_mb": peak_vram_mb(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- profiler


def profile_step(cfg: TrainConfig, steps: int = 30, warmup: int = 5) -> dict[str, float]:
    """Mean milliseconds per phase of a training step.

    Phases: ``data`` (synthesis), ``forward`` (network), ``loss`` (render + matching + terms),
    ``backward``, ``optimizer`` and ``step`` (their sum). ``matching`` is the time of the cost
    matrix and the assignment alone, measured on the same predictions (it is part of ``loss``).
    """
    ctx = _Context(cfg)
    cuda = ctx.device.type == "cuda"

    def clock() -> float:
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    totals = dict.fromkeys(("data", "forward", "loss", "matching", "backward", "optimizer"), 0.0)
    for i in range(warmup + steps):
        t0 = clock()
        batch = ctx.generator.sample(cfg.batch_size, ctx.data_rng)
        t1 = clock()
        ctx.net.train()
        shape, colour, logits = ctx.forward(batch)
        t2 = clock()
        out = ctx.loss(shape, colour, logits, batch)
        t3 = clock()
        with torch.no_grad():
            valid = batch.gt_decision.bool()
            pred = shape if cfg.model.colour_mode == "sampled" else torch.cat([shape, colour], -1)
            gt = batch.gt_param if cfg.model.colour_mode == "sampled" else None
            if gt is None:
                gt = torch.cat([batch.gt_param, batch.gt_color], -1)
            cost = ctx.loss.cost_matrix(pred, logits, gt)
            ctx.loss.match(cost, valid)
        t4 = clock()
        ctx.optimizer.zero_grad(set_to_none=True)
        out.total.backward()  # type: ignore[no-untyped-call]
        t5 = clock()
        ctx.optimizer.step()
        t6 = clock()
        if i >= warmup:
            for k, v in zip(
                totals, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5), strict=True
            ):
                totals[k] += v
    ms = {k: 1000 * v / steps for k, v in totals.items()}
    # `matching` was measured separately (after `loss`), so it is not part of the step sum
    ms["step"] = sum(v for k, v in ms.items() if k != "matching")
    return ms
