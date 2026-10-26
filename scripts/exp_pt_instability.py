"""Late-training instability of convention option A on the stretched recipe (T-023).

    python -m scripts.exp_pt_instability trace  --name A_s0 --seed 0 [--convention positive]
        # option A (or raw) on the stretched recipe, every step logged with gradient/Adam/activation
        # diagnostics, rotating checkpoints (the last 12,000 steps, frozen by an early alert), stops 600
        # steps after the first loss spike
    python -m scripts.exp_pt_instability branch --name ctl1 --from-trace A_s0 --steps 6000 --data-seed 1 [factor flags]
        # resume from the checkpoint --back steps before the early alert, change ONE factor, record alerts and spikes
    python -m scripts.exp_pt_instability report
        # tables and figure -> docs/experiments/pt_training.md (block T023)

An *alert* is a rise of the 100-step mean total loss by more than 4 standard deviations of the previous 2000 steps
(it fires at the onset of an excursion); a *spike* is a jump of the 100-step mean total loss above 1.25 x the median of the previous 1000 steps
(the scan used on the stored T-019 logs). The training step is the one of ``brushstroke.stroke_model.train``
plus diagnostics (tested to produce the identical weights). Data: ``docs/experiments/data/pt_instability/``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import statistics
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from brushstroke.stroke_model import model as model_module
from brushstroke.stroke_model.inference import _NUMPY_STATE_GLOBALS
from brushstroke.stroke_model.train import (
    LOSS_TERMS,
    TrainConfig,
    _Context,
    evaluate,
    load_checkpoint,
    make_validation_set,
    save_checkpoint,
)
from brushstroke.utils.config import load_config

ROOT = Path("outputs/pt_instab")
CONFIG = Path("configs/stroke_model/pt.yaml")
STRETCHED = ["schedule.n_epochs=400", "schedule.n_epochs_decay=40"]  # the T-019 stretched recipe
SPIKE_FACTOR, SPIKE_WINDOW, SPIKE_BASE, SPIKE_WARMUP = 1.25, 100, 1000, 5000
AFTER_SPIKE, CKPT_EVERY, CKPT_KEEP = 600, 500, 24  # 24 x 500: the 12,000 steps before an alert
# early alert: the 100-step mean of the total loss is more than ALERT_Z standard deviations above the
# median of the previous 2000 steps. It fires at the onset of an excursion, thousands of steps before
# the 1.25x rule of the spike (seed 1 of T-019: 31,175 against 35,625), and freezes the checkpoint rotation.
ALERT_Z, ALERT_BASE, ALERT_HOLD = 4.0, 2000, 4000
GROUPS = ("cnn", "tr_enc", "tr_dec", "head", "embed")


def group_of(name: str) -> str:
    """Parameter group of a ``PaintTransformer`` parameter name."""
    if name.startswith(("enc_img", "enc_canvas", "conv")):
        return "cnn"
    if name.startswith("transformer.encoder"):
        return "tr_enc"
    if name.startswith("transformer.decoder"):
        return "tr_dec"
    if name.startswith(("linear_param", "linear_decider")):
        return "head"
    return "embed"  # query_pos, row_embed, col_embed


class SpikeDetector:
    """Online version of the scan: the 100-step mean of the total loss against 1.25 x the median of the
    previous 1000 steps (sampled every 25 steps). ``update`` returns True once, at the first spike."""

    def __init__(self) -> None:
        self.totals: deque[float] = deque(maxlen=SPIKE_WINDOW)
        self.base: deque[float] = deque(maxlen=SPIKE_BASE // 25)
        self.spike: int | None = None

    def update(self, step: int, total: float) -> bool:
        self.totals.append(total)
        if len(self.totals) < SPIKE_WINDOW:
            return False
        mean = sum(self.totals) / SPIKE_WINDOW
        hit = (
            self.spike is None
            and step > SPIKE_WARMUP
            and len(self.base) == self.base.maxlen
            and mean > SPIKE_FACTOR * statistics.median(self.base)
        )
        if step % 25 == 0:
            self.base.append(mean)
        if hit:
            self.spike = step
        return hit


class AlertDetector:
    """Online early alert: z-score of the 100-step mean total against the previous ``ALERT_BASE`` steps
    (100-step means sampled every 25 steps). ``update`` returns True when the score exceeds ``ALERT_Z``."""

    def __init__(self) -> None:
        self.totals: deque[float] = deque(maxlen=SPIKE_WINDOW)
        self.base: deque[float] = deque(maxlen=ALERT_BASE // 25)

    def update(self, step: int, total: float) -> bool:
        self.totals.append(total)
        if len(self.totals) < SPIKE_WINDOW:
            return False
        mean = sum(self.totals) / SPIKE_WINDOW
        hit = False
        if step > SPIKE_WARMUP and len(self.base) == self.base.maxlen:
            spread = statistics.pstdev(self.base)
            hit = spread > 0 and (mean - statistics.median(self.base)) / spread > ALERT_Z
        if step % 25 == 0:
            self.base.append(mean)
        return hit


class Tap:
    """Forward hooks on the pieces the diagnostics read: the transformer encoder output (a collapsed
    network has the same output for every token and sample) and the raw stroke parameters."""

    def __init__(self, net: torch.nn.Module) -> None:
        self.enc_std = float("nan")
        self.theta_abs_max = float("nan")
        self.wh_raw_min = float("nan")
        self.near_wrap = float("nan")
        net.transformer.encoder.register_forward_hook(self._encoder)  # type: ignore[union-attr]
        net.linear_param.register_forward_hook(self._head)  # type: ignore[union-attr]

    def _encoder(self, _m: torch.nn.Module, _i: Any, out: torch.Tensor) -> None:
        self.enc_std = float(out.detach().reshape(-1, out.shape[-1]).std(0).mean())

    def _head(self, _m: torch.nn.Module, _i: Any, out: torch.Tensor) -> None:
        raw = out.detach()
        frac = raw[..., 4] - torch.floor(raw[..., 4])
        self.theta_abs_max = float(raw[..., 4].abs().max())
        self.wh_raw_min = float(raw[..., 2:4].min())
        self.near_wrap = float(((frac < 0.02) | (frac > 0.98)).float().mean())


def grad_stats(ctx: _Context) -> dict[str, float]:
    """Gradient norms (total, per group) and the ratio ``||g|| / ||sqrt(v_hat)||`` of Adam per group,
    taken after backward and before the optimizer step (``v`` from the previous steps)."""
    o = ctx.cfg.optim
    gsq = dict.fromkeys(GROUPS, 0.0)
    vsq = dict.fromkeys(GROUPS, 0.0)
    gtensors: dict[str, list[torch.Tensor]] = {g: [] for g in GROUPS}
    vtensors: dict[str, list[torch.Tensor]] = {g: [] for g in GROUPS}
    for name, p in ctx.net.named_parameters():
        if p.grad is None:
            continue
        g = group_of(name)
        gtensors[g].append(p.grad.detach().reshape(-1))
        state = ctx.optimizer.state.get(p)
        if state:
            t = state["step"].item() if torch.is_tensor(state["step"]) else state["step"]
            vhat = state["exp_avg_sq"] / (1 - o.beta2**t)
            vtensors[g].append(vhat.sqrt().reshape(-1))
    for g in GROUPS:
        gsq[g] = float(torch.cat(gtensors[g]).norm()) if gtensors[g] else 0.0
        vsq[g] = float(torch.cat(vtensors[g]).norm()) if vtensors[g] else float("nan")
    total = sum(v * v for v in gsq.values()) ** 0.5
    out = {"gnorm": total}
    for g in GROUPS:
        out[f"g_{g}"] = gsq[g]
        out[f"r_{g}"] = gsq[g] / vsq[g] if vsq[g] == vsq[g] and vsq[g] > 0 else float("nan")
    return out


def traced_step(ctx: _Context, batch: Any, tap: Tap) -> dict[str, float]:
    """``_Context.train_step`` plus the diagnostics (same operations in the same order)."""
    ctx.net.train()
    out = ctx.loss(*ctx.forward(batch), batch)
    ctx.optimizer.zero_grad(set_to_none=True)
    out.total.backward()  # type: ignore[no-untyped-call]
    row = {k: float(getattr(out, k).detach()) for k in LOSS_TERMS}
    row.update(grad_stats(ctx))
    if ctx.cfg.optim.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(ctx.net.parameters(), ctx.cfg.optim.grad_clip)
    ctx.optimizer.step()
    ctx.scheduler.step()
    ctx.step += 1
    row.update(
        enc_std=tap.enc_std,
        theta_abs_max=tap.theta_abs_max,
        wh_raw_min=tap.wh_raw_min,
        near_wrap=tap.near_wrap,
    )
    return row


FIELDS = [
    "step",
    "lr",
    *LOSS_TERMS,
    "gnorm",
    *(f"g_{g}" for g in GROUPS),
    *(f"r_{g}" for g in GROUPS),
    "enc_std",
    "theta_abs_max",
    "wh_raw_min",
    "near_wrap",
]


def run_loop(
    ctx: _Context,
    out: Path,
    max_steps: int,
    *,
    keep_checkpoints: bool = True,
    prime: Iterable[tuple[int, float]] = (),
    val_every: int = 0,
) -> dict[str, Any]:
    """Train ``ctx`` for ``max_steps`` (total step count) or until ``AFTER_SPIKE`` steps after the first
    spike, logging every step; writes ``trace.csv``, rotating ``ckpt_<step>.pt`` and ``summary.json``."""
    out.mkdir(parents=True, exist_ok=True)
    tap = Tap(ctx.net)
    detector, alarm = SpikeDetector(), AlertDetector()
    for step, total in prime:  # the losses before the start of a branch, for the reference level
        detector.update(step, total)
        alarm.update(step, total)
    detector.spike = None
    alerts: list[int] = []
    alert_at: int | None = None  # the alert in force: the checkpoint rotation is frozen
    stop_at = max_steps
    ckpts: deque[Path] = deque()
    val_set = make_validation_set(ctx) if val_every else []
    with (out / "trace.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        while ctx.step < stop_at:
            batch = ctx.generator.sample(ctx.cfg.batch_size, ctx.data_rng)
            lr = float(ctx.scheduler.get_last_lr()[0])
            row = traced_step(ctx, batch, tap)
            writer.writerow({"step": ctx.step, "lr": lr, **row})
            if alarm.update(ctx.step, row["total"]) and alert_at is None:
                alert_at = ctx.step
                alerts.append(ctx.step)
                print(f"alert at step {ctx.step}", flush=True)
            if alert_at is not None and detector.spike is None and ctx.step - alert_at > ALERT_HOLD:
                alert_at = None  # no spike followed: resume the rotation
            if detector.update(ctx.step, row["total"]):
                stop_at = min(stop_at, ctx.step + AFTER_SPIKE)
                fh.flush()
                print(f"spike at step {ctx.step}", flush=True)
            frozen = alert_at is not None or detector.spike is not None
            if keep_checkpoints and ctx.step % CKPT_EVERY == 0 and not frozen:
                path = out / f"ckpt_{ctx.step}.pt"
                save_checkpoint(ctx, path)
                ckpts.append(path)
                while len(ckpts) > CKPT_KEEP:
                    ckpts.popleft().unlink(missing_ok=True)
            if val_every and ctx.step % val_every == 0:
                evaluate(ctx, val_set, mode="running")  # the validation of the T-019 runs
            if ctx.step % 1000 == 0:
                fh.flush()
                print(ctx.step, round(row["total"], 3), flush=True)
    summary = {
        "steps": ctx.step,
        "spike": detector.spike,
        "alert": alert_at if detector.spike is not None else None,
        "alerts": alerts,
        "convention": ctx.cfg.model.convention,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- factors of the branches


def _positive(wh: str, theta: str) -> Callable[[torch.Tensor], torch.Tensor]:
    def convention(param: torch.Tensor) -> torch.Tensor:
        out = param.clone()
        if wh == "softplus":
            out[..., 2:4] = F.softplus(param[..., 2:4])
        elif wh == "exp":
            out[..., 2:4] = torch.exp(param[..., 2:4])
        else:
            raise ValueError(wh)
        if theta == "wrap":
            out[..., 4] = param[..., 4] - torch.floor(param[..., 4])
        elif theta != "none":
            raise ValueError(theta)
        return out

    return convention


@contextlib.contextmanager
def patched_convention(wh: str, theta: str) -> Iterator[None]:
    """Replace the option-A parameterization (``softplus`` + wrapped ``theta`` by default) for a branch."""
    original = model_module.positive_convention
    model_module.positive_convention = _positive(wh, theta)
    try:
        yield
    finally:
        model_module.positive_convention = original


def make_cfg(
    seed: int, convention: str, output: Path, extra: list[str] | None = None
) -> TrainConfig:
    return load_config(
        TrainConfig,
        CONFIG,
        [
            f"seed={seed}",
            f"model.convention={convention}",
            *STRETCHED,
            f"output_dir={output}",
            "val.every=0",
            *(extra or []),
        ],
    )


# ----------------------------------------------------------------------------- stages


def cmd_trace(args: argparse.Namespace) -> None:
    out = ROOT / args.name
    cfg = make_cfg(args.seed, args.convention, out)
    ctx = _Context(cfg)
    run_loop(ctx, out, args.max_steps, val_every=args.val_every)


def last_good_checkpoint(trace: Path, back: int) -> Path:
    """The checkpoint at least ``back`` steps before the alert that preceded the spike of ``trace`` (the
    spike itself for a summary without alert): the rotation keeps the 12,000 steps before the alert."""
    info = json.loads((trace / "summary.json").read_text(encoding="utf-8"))
    spike = info.get("alert") or info["spike"]
    if spike is None:
        raise SystemExit(f"{trace} has no spike")
    steps = sorted(int(p.stem.split("_")[1]) for p in trace.glob("ckpt_*.pt"))
    ok = [s for s in steps if s <= spike - back]
    if not ok:
        raise SystemExit(f"no checkpoint {back} steps before the alert/spike {spike} in {trace}")
    return trace / f"ckpt_{ok[-1]}.pt"


def scale_lr(ctx: _Context, factor: float) -> None:
    """Multiply the learning rate of the whole remaining schedule (the scheduler multiplies its
    ``base_lrs`` by the schedule factor at every step) and the current one."""
    ctx.scheduler.base_lrs = [lr * factor for lr in ctx.scheduler.base_lrs]  # type: ignore[attr-defined]
    for g in ctx.optimizer.param_groups:
        g["lr"] *= factor


def keep_batchnorm_in_eval(net: torch.nn.Module) -> None:
    """``net.train()`` keeps the BatchNorm layers in eval mode (running statistics, frozen)."""
    bn = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    original_train = net.train

    def train_keep_bn(mode: bool = True) -> torch.nn.Module:
        original_train(mode)
        for m in bn:
            m.eval()
        return net

    net.train = train_keep_bn  # type: ignore[method-assign]


def parent_totals(trace: Path, upto: int) -> list[tuple[int, float]]:
    """(step, total loss) of the parent trace up to ``upto``: the reference level of a branch's detector."""
    with (trace / "trace.csv").open(newline="", encoding="utf-8") as fh:
        rows = [(int(r["step"]), float(r["total"])) for r in csv.DictReader(fh)]
    return [(s, t) for s, t in rows if s <= upto]


def cmd_branch(args: argparse.Namespace) -> None:
    trace = ROOT / args.from_trace
    ckpt = Path(args.ckpt) if args.ckpt else last_good_checkpoint(trace, args.back)
    out = ROOT / "branches" / args.name
    with torch.serialization.safe_globals(
        _NUMPY_STATE_GLOBALS
    ):  # our own checkpoint, never a full unpickle
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    convention = state["config"]["model"]["convention"]
    if convention != "positive" and (args.wh, args.theta) != ("softplus", "wrap"):
        raise SystemExit("--wh and --theta change the positive parameterization: the parent is raw")
    cfg = make_cfg(state["config"]["seed"], convention, out)
    if args.grad_clip:
        cfg.optim.grad_clip = args.grad_clip
    ctx = _Context(cfg)
    start = load_checkpoint(ctx, ckpt)
    ctx.data_rng.manual_seed(1000 + args.data_seed)  # replicate = another data stream
    if args.lr_scale != 1.0:
        scale_lr(ctx, args.lr_scale)
    if args.beta1 is not None:
        for g in ctx.optimizer.param_groups:
            g["betas"] = (args.beta1, g["betas"][1])
    if args.bn_eval:
        keep_batchnorm_in_eval(ctx.net)
    with patched_convention(args.wh, args.theta):
        summary = run_loop(
            ctx, out, start + args.steps, keep_checkpoints=False, prime=parent_totals(trace, start)
        )
    summary.update(
        {
            "from": str(ckpt),
            "start": start,
            "factors": factor_dict(args),
            "data_seed": args.data_seed,
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


def factor_dict(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "lr_scale": args.lr_scale,
        "grad_clip": args.grad_clip,
        "beta1": args.beta1,
        "bn_eval": args.bn_eval,
        "wh": args.wh,
        "theta": args.theta,
    }


# ----------------------------------------------------------------------------- report

DATA = Path("docs/experiments/data/pt_instability")
FIGURE = Path("docs/experiments/pt_instability.png")
T19 = Path("outputs/pt_conv_long/optA_positive")
REPLICATES = {  # run -> (convention, validation)
    "A_s0": ("positive", "none"),
    "A_s1": ("positive", "none"),
    "A_s2": ("positive", "none"),
    "A_s0v": ("positive", "T-019"),
    "A_s1v": ("positive", "T-019"),
    "A_s5": ("positive", "T-019"),
    "A_s6": ("positive", "T-019"),
    "R_s0": ("raw", "none"),
}
EXACT = {
    "T19_s0": 0,
    "T19_s1": 1,
}  # outputs/pt_instab/T19_s*: scripts.train_pt with the T-019 command


def _column(path: Path, name: str) -> tuple[Any, Any]:
    import numpy as np

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return (
        np.array([float(r["step"]) for r in rows]),
        np.array([float(r[name]) for r in rows]),
    )


def scan_spikes(step: Any, total: Any, every: int) -> list[int]:
    """The spike rule on a log sampled every ``every`` steps: 100-step mean above ``SPIKE_FACTOR`` x the
    median of the previous 1000 steps, at least 3000 steps apart, after ``SPIKE_WARMUP``."""
    import numpy as np

    k, base = SPIKE_WINDOW // every, SPIKE_BASE // every
    sm = np.convolve(total, np.ones(k) / k, mode="valid")
    st = step[k - 1 :]
    hits: list[int] = []
    for i in range(base, len(st)):
        if st[i] <= SPIKE_WARMUP or (hits and st[i] - hits[-1] <= 3000):
            continue
        if sm[i] > SPIKE_FACTOR * np.median(sm[i - base : i]):
            hits.append(int(st[i]))
    return hits


TERMS = ("total", "pixel", "gt", "w", "decision")


def z_scores(step: Any, x: Any, every: int) -> Any:
    """Score of each 100-step mean of ``x`` against the median and standard deviation of the previous
    ``ALERT_BASE`` steps (NaN before). ``x`` is logged every ``every`` steps."""
    import numpy as np

    k, base = SPIKE_WINDOW // every, ALERT_BASE // every
    sm = np.convolve(x, np.ones(k) / k, mode="valid")
    z = np.full(len(sm), np.nan)
    for i in range(base, len(sm)):
        ref = sm[i - base : i]
        if ref.std() > 0:
            z[i] = (sm[i] - np.median(ref)) / ref.std()
    return step[k - 1 :], z


def scan_alerts(step: Any, total: Any, every: int, upto: int = 10**9) -> list[int]:
    """The early alerts (``z > ALERT_Z``, at least 1500 steps apart) of a log, before ``upto``."""
    st, z = z_scores(step, total, every)
    hits: list[int] = []
    for s, v in zip(st, z, strict=True):
        if SPIKE_WARMUP < s <= upto and v > ALERT_Z and (not hits or s - hits[-1] > 1500):
            hits.append(int(s))
    return hits


def departure(a: Any, b: Any, every: int = 25, threshold: float = 0.1) -> int | None:
    """First step (after 10,000) where the 100-step mean total of run ``a`` exceeds the one of the
    same-seed rerun ``b`` by ``threshold`` (the largest difference of the first 20,000 steps is 0.05)."""
    import numpy as np

    n, k = min(len(a["step"]), len(b["step"])), SPIKE_WINDOW // every
    d = np.convolve(a["total"][:n], np.ones(k) / k, "valid") - np.convolve(
        b["total"][:n], np.ones(k) / k, "valid"
    )
    st = a["step"][k - 1 : n]
    hit = np.nonzero((d > threshold) & (st > 10_000))[0]
    return int(st[hit[0]]) if len(hit) else None


def first_crossings(log: dict[str, Any], event: int) -> dict[str, int | None]:
    """For each loss term, the first step within [event - 100, event + 1000] where its score exceeds ``ALERT_Z``."""
    out: dict[str, int | None] = {}
    for term in TERMS:
        st, z = z_scores(log["step"], log[term], 25)
        window = [
            int(s)
            for s, v in zip(st, z, strict=True)
            if event - 100 <= s <= event + 1000 and v > ALERT_Z
        ]
        out[term] = window[0] if window else None
    return out


def _log(path: Path) -> dict[str, Any]:
    import numpy as np

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("step", *TERMS)}


def cmd_report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from scripts.pt_repro_common import write_block

    lines = [
        "| run | source | convention | validation | steps | spikes (first) | `enc_std` (min / median) | raw theta max (median / max) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=110)

    def smooth(step: Any, total: Any, width: int) -> tuple[Any, Any]:
        return step[width - 1 :], np.convolve(total, np.ones(width) / width, mode="valid")

    for s in range(3):
        step, total = _column(T19 / f"seed{s}" / "train.csv", "total")
        hits = scan_spikes(step, total, 25)
        out[f"T19_seed{s}"] = {"spikes": hits, "steps": int(step[-1])}
        lines.append(
            f"| T-019 seed {s} | stored log | positive | T-019 | {int(step[-1])} | {len(hits)} ({hits[0] if hits else '-'}) | - | - |"
        )
        x, y = smooth(step, total, 40)
        ax.plot(
            x,
            y,
            lw=1.2,
            color="#c0392b" if hits else "#2c7fb8",
            label=f"T-019 seed {s}" + (" (spikes)" if hits else ""),
            zorder=3,
        )
    first = True
    for name, (conv, val) in REPLICATES.items():
        d = ROOT / name
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        step, total = _column(d / "trace.csv", "total")
        hits = scan_spikes(step, total, 1)
        _, enc = _column(d / "trace.csv", "enc_std")
        _, th = _column(d / "trace.csv", "theta_abs_max")
        late = step > 30_000
        out[name] = {
            "spike": summary["spike"],
            "scan": hits,
            "steps": summary["steps"],
            "enc_std_min": float(enc[late].min()),
            "theta_abs_max": float(th[late].max()),
        }
        lines.append(
            f"| {name} | trace (this study) | {conv} | {val} | {summary['steps']} | {len(hits)} ({hits[0] if hits else '-'}) | {enc[late].min():.3f} / {np.median(enc[late]):.3f} | {np.median(th[late]):.2f} / {th[late].max():.2f} |"
        )
        x, y = smooth(step, total, 1000)
        ax.plot(
            x, y, lw=0.8, color="#888888", alpha=0.7, label="replicates (traces)" if first else None
        )
        first = False
    for name, seed in EXACT.items():
        step, total = _column(ROOT / name / "train.csv", "total")
        hits = scan_spikes(step, total, 25)
        out[name] = {"scan": hits, "steps": int(step[-1])}
        lines.append(
            f"| {name} | `scripts.train_pt`, T-019 command | positive | T-019 | {int(step[-1])} | {len(hits)} ({hits[0] if hits else '-'}) | - | - |"
        )
        x, y = smooth(step, total, 40)
        ax.plot(
            x,
            y,
            lw=0.8,
            color="#4daf4a",
            alpha=0.8,
            label="exact T-019 command" if seed == 0 else None,
        )
    ax.set(
        xlabel="step",
        ylabel="total loss (1000-step mean for traces)",
        ylim=(3.5, 9),
        title="Option A, stretched recipe: T-019 vs replicates",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE)
    onset_lines = [
        "| T-019 seed | departs from its rerun | first alerts (before 48,000) | first step with score > 4, per term at the first alert (total / pixel / gt / w / decision) | first 1.25x spike |",
        "|---|---|---|---|---|",
    ]
    for seed in (0, 1):
        a = _log(T19 / f"seed{seed}" / "train.csv")
        b = _log(ROOT / f"T19_s{seed}" / "train.csv")
        alerts = scan_alerts(a["step"], a["total"], 25, upto=48_000)
        cross = first_crossings(a, alerts[0]) if alerts else {}
        dep = departure(a, b)
        spikes = out[f"T19_seed{seed}"]["spikes"]
        out[f"onset_seed{seed}"] = {"departure": dep, "alerts": alerts, "first_crossings": cross}
        onset_lines.append(
            f"| {seed} | {dep} | {', '.join(map(str, alerts))} | "
            + " / ".join(str(cross.get(t)) for t in TERMS)
            + f" | {spikes[0]} |"
        )
    healthy = _log(T19 / "seed2" / "train.csv")
    quiet = {"T19_seed2": scan_alerts(healthy["step"], healthy["total"], 25, upto=48_000)}
    for name in REPLICATES:
        step, total = _column(ROOT / name / "trace.csv", "total")
        quiet[name] = scan_alerts(step, total, 1)
    for name in EXACT:
        step, total = _column(ROOT / name / "train.csv", "total")
        quiet[name] = scan_alerts(step, total, 25)
    out["alerts_healthy"] = quiet
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "runs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    write_block(
        "T023", chr(10).join([*lines, "", *onset_lines]), Path("docs/experiments/pt_training.md")
    )
    print(FIGURE, FIGURE.stat().st_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--name", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--convention", default="positive", choices=["positive", "raw"])
    t.add_argument("--max-steps", type=int, default=60_000)
    t.add_argument("--val-every", type=int, default=0, help="run the T-019 validation (running BN)")
    b = sub.add_parser("branch")
    b.add_argument("--name", required=True)
    b.add_argument("--from-trace", required=True)
    b.add_argument("--ckpt", default="")
    b.add_argument(
        "--back",
        type=int,
        default=2000,
        help="checkpoint at least this many steps before the early alert (up to 11000 are kept)",
    )
    b.add_argument("--steps", type=int, default=6000)
    b.add_argument("--data-seed", type=int, default=1)
    b.add_argument("--lr-scale", type=float, default=1.0)
    b.add_argument("--grad-clip", type=float, default=0.0)
    b.add_argument("--beta1", type=float, default=None)
    b.add_argument("--bn-eval", action="store_true")
    b.add_argument("--wh", default="softplus", choices=["softplus", "exp"])
    b.add_argument("--theta", default="wrap", choices=["wrap", "none"])
    sub.add_parser("report")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.stage == "trace":
        cmd_trace(args)
    elif args.stage == "branch":
        cmd_branch(args)
    else:
        cmd_report()


if __name__ == "__main__":
    main(sys.argv[1:])
