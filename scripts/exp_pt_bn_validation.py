"""BatchNorm in the validation of the PT training: the 8000-step diagnostic of T-020.

    python -m scripts.exp_pt_bn_validation run        # 8000 steps, seed 0; every 100 steps the validation set is
                                                      # scored with running / recalibrated (K = 5, 20, 50) / batch statistics
    python -m scripts.exp_pt_bn_validation bitwise    # train() with val.bn = running vs recalibrate vs batch: same weights?
    python -m scripts.exp_pt_bn_validation report     # tables -> docs/experiments/pt_training.md (block BN)

All the modes score the same network at the same step in one run (the validation never changes the
training state), so the comparison is paired. Data: ``docs/experiments/data/pt_bn/``. Design note:
``docs/design/pt_bn_validation.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch

from brushstroke.stroke_model.train import (
    TrainConfig,
    _Context,
    evaluate,
    make_calibration_set,
    make_validation_set,
    train,
)
from scripts.pt_repro_common import write_block

DATA = Path("docs/experiments/data/pt_bn")
DOC = Path("docs/experiments/pt_training.md")
STEPS, EVERY, SEED = 8000, 100, 0  # the T-014 diagnostic
CALIBRATIONS = (5, 20, 50)  # K of the recalibration
SERIES = ("running", *(f"recal{k}" for k in CALIBRATIONS), "batch")
SPIKE, WINDOW, SETTLED = 0.3, 7, 1000  # spike height over the local median, window, steps skipped
BITWISE_STEPS, BITWISE_BATCH = (
    40,
    16,
)  # CPU: the GPU training is not bitwise reproducible (see report)


def run() -> None:
    cfg = TrainConfig(seed=SEED, output_dir="outputs/pt_bn")
    cfg = replace(cfg, val=replace(cfg.val, bn_batches=max(CALIBRATIONS)))
    ctx = _Context(cfg)
    val_set, calibration = make_validation_set(ctx), make_calibration_set(ctx)
    DATA.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    def score() -> None:
        row: dict[str, float] = {"step": float(ctx.step)}
        for name in SERIES:
            if name.startswith("recal"):
                k = int(name.removeprefix("recal"))
                metrics = evaluate(ctx, val_set, "recalibrate", calibration[:k])
            else:
                metrics = evaluate(ctx, val_set, name)
            row.update({f"{name}_{t}": v for t, v in metrics.items() if t in ("total", "decision")})
        rows.append(row)
        print({k: round(v, 3) for k, v in row.items() if k.endswith("total")}, flush=True)

    score()
    while ctx.step < STEPS:
        ctx.train_step(ctx.generator.sample(cfg.batch_size, ctx.data_rng))
        if ctx.step % EVERY == 0:
            score()
    with (DATA / "curves.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_hash(path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name in sorted(state["model"]):
        digest.update(state["model"][name].numpy().tobytes())
    return digest.hexdigest()[:16]


def bitwise() -> None:
    """The same training on the CPU with each validation mode: identical weights (SHA-256).

    On the GPU two runs of the identical configuration (even with ``deterministic: true``) end with
    different weights (control measured for T-020), so only the CPU can show bitwise equality.
    """
    hashes: dict[str, str] = {}
    for mode in ("running", "recalibrate", "batch"):
        cfg = TrainConfig(
            seed=SEED,
            device="cpu",
            batch_size=BITWISE_BATCH,
            output_dir=f"outputs/pt_bn_bitwise/{mode}",
        )
        cfg = replace(
            cfg,
            schedule=replace(cfg.schedule, max_steps=BITWISE_STEPS),
            val=replace(cfg.val, every=10, bn=mode),
        )
        train(cfg)
        hashes[mode] = state_hash(Path(cfg.output_dir) / "latest.pt")
        print(mode, hashes[mode], flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {"steps": BITWISE_STEPS, "hash": hashes}
    (DATA / "bitwise.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8", newline="\n"
    )


def rolling_median(values: list[float], window: int) -> list[float]:
    half = window // 2
    return [statistics.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]


def curve_stats(steps: list[float], values: list[float]) -> dict[str, float]:
    """Spread of a validation curve after the first ``SETTLED`` steps: std of the values, std of
    the residual to the local median (what a smooth curve would not have), spikes = residuals
    above ``SPIKE``."""
    keep = [i for i, s in enumerate(steps) if s >= SETTLED]
    residual = [v - m for v, m in zip(values, rolling_median(values, WINDOW), strict=True)]
    res = [residual[i] for i in keep]
    return {
        "std": statistics.stdev(values[i] for i in keep),
        "residual_std": statistics.pstdev(res),
        "spikes": float(sum(r > SPIKE for r in res)),
        "max_spike": max(res),
    }


def report() -> None:
    with (DATA / "curves.csv").open(newline="", encoding="utf-8") as fh:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    steps = [r["step"] for r in rows]
    col = {n: [r[f"{n}_total"] for r in rows] for n in SERIES}
    stats = {n: curve_stats(steps, col[n]) for n in SERIES}
    names = {
        "running": "running statistics (before)",
        **{f"recal{k}": f"recalibrated, K = {k} batches" for k in CALIBRATIONS},
        "batch": "batch statistics",
    }
    n_kept = sum(x >= SETTLED for x in steps)
    lines = [
        "| validation statistics | std of the curve | std of the residual to the local median | spikes > 0.3 | largest spike |",
        "|---|---|---|---|---|",
        *(
            f"| {names[n]} | {s['std']:.3f} | {s['residual_std']:.3f} | {s['spikes']:.0f} / {n_kept} | {s['max_spike']:.3f} |"
            for n, s in stats.items()
        ),
    ]
    diff = [abs(a - b) for a, b in zip(col["running"], col["recal20"], strict=True)]
    big = sum(d > SPIKE for d in diff)
    last = rows[-1]
    gap_batch = max(abs(a - b) for a, b in zip(col["recal20"], col["batch"], strict=True))
    k_gap = max(abs(a - b) for a, b in zip(col["recal20"], col["recal50"], strict=True))
    text = [
        f"Diagnostic run of {int(steps[-1])} steps (seed 0, official recipe, validation every {EVERY} steps = {len(rows)} "
        "evaluations of the same network with every mode; `python -m scripts.exp_pt_bn_validation run`). Statistics over the "
        f"evaluations from step {SETTLED} on (the first thousand steps are the initial descent); residual = value minus the "
        f"median of {WINDOW} neighbouring evaluations.",
        "",
        *lines,
        "",
        f"- Running statistics differ from the recalibrated ones (K = 20) by more than {SPIKE} in {big} of {len(rows)} evaluations "
        f"(largest difference {max(diff):.3f}); at the last step the two totals are {last['running_total']:.3f} and {last['recal20_total']:.3f}.",
        f"- Recalibration with K = 20 and with K = 50 batches differ by at most {k_gap:.3f}; recalibrated and batch statistics differ by at most {gap_batch:.3f}.",
    ]
    bw = DATA / "bitwise.json"
    if bw.exists():
        data = json.loads(bw.read_text(encoding="utf-8"))
        same = len(set(data["hash"].values())) == 1
        hashes = ", ".join(f"`{m}` {h}" for m, h in data["hash"].items())
        verdict = "identical" if same else "DIFFERENT"
        text.append(
            f"- Training weights ({data['steps']} steps on the CPU, batch {BITWISE_BATCH}, SHA-256 of the state dict): {hashes} -> **{verdict}**."
        )
    write_block("BN", "\n".join(text), DOC)
    print("\n".join(text))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["run", "bitwise", "report"])
    args = parser.parse_args(argv)
    {"run": run, "bitwise": bitwise, "report": report}[args.stage]()


if __name__ == "__main__":
    main(sys.argv[1:])
