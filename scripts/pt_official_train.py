"""Train the *official* Paint Transformer code once (reference run of T-014), outside git.

Runs ``PainterModel`` of ``Huage001/PaintTransformer`` (downloaded by ``scripts.pt_reference_study
fetch``) with the recipe of ``train.sh`` and its own on-the-fly data (``set_input``): batch 64,
Adam lr 1e-4 betas (0.5, 0.999), 56,320 steps, lr x0.3 every 5 epochs after epoch 200 (epochs of
256 steps). Third-party code is executed. Writes ``train.csv`` (mean of the four official loss
terms over ``--log-every`` steps) and ``model.pth`` (the ``state_dict`` of the network, loadable by
:class:`brushstroke.stroke_model.PaintTransformer`) into ``--out`` (default
``outputs/pt_official_train``, git-ignored).

    python -m scripts.pt_official_train [--seed 0] [--steps 56320] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from scripts.pt_reference_study import _official_model

STEPS_PER_EPOCH, N_EPOCHS = 256, 200
TERMS = ("pixel", "gt", "w", "decision")


def lr_factor(step: int) -> float:
    """Official ``linear`` policy at the (per-step) step index."""
    return float(0.3 ** max(0, (step // STEPS_PER_EPOCH + 1 - N_EPOCHS) // 5))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=56320)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("outputs/pt_official_train"))
    args = parser.parse_args(argv)
    out = args.out.resolve()  # the official model changes the working directory
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = _official_model(gpu=True)
    scheduler = LambdaLR(model.optimizer, lr_factor)  # type: ignore[attr-defined]
    window = dict.fromkeys(TERMS, 0.0)
    start = time.perf_counter()
    with (out / "train.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "lr", "total", *TERMS, "elapsed_s"])
        for step in range(1, args.steps + 1):
            model.set_input({"A_paths": ["x"] * 64})  # type: ignore[attr-defined]
            model.optimize_parameters()  # type: ignore[attr-defined]
            scheduler.step()
            for k in TERMS:
                window[k] += float(getattr(model, f"loss_{k}"))
            if step % args.log_every == 0:
                mean = [window[k] / args.log_every for k in TERMS]
                writer.writerow(
                    [step, model.optimizer.param_groups[0]["lr"], sum(mean), *mean]  # type: ignore[attr-defined]
                    + [time.perf_counter() - start]
                )
                fh.flush()
                window = dict.fromkeys(TERMS, 0.0)
    state = model.net_g.state_dict()  # type: ignore[attr-defined]
    # DataParallel wrapper: drop the "module." prefix
    state = {k.removeprefix("module."): v for k, v in state.items()}
    torch.save(state, out / "model.pth")
    print(f"done in {(time.perf_counter() - start) / 3600:.2f} h -> {out}")


if __name__ == "__main__":
    main()
