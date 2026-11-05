"""Error report of the Learning to Paint neural renderer (T-028): ours vs the analytic rasterizer, and vs the official weights.

Usage (repo root)::

    python -m scripts.l2p_renderer_report --weights outputs/l2p_renderer/renderer.pt \\
        [--official data/raw/l2p_official/renderer.pkl] [--actor data/raw/l2p_official/actor.pkl --image path.png] \\
        --out docs/experiments/l2p_renderer.md

On ``--strokes`` (default 1000) fresh uniform random strokes (a seed no training batch uses) it reports, for our renderer and for the
official weights, the error against the analytic rasterizer :func:`brushstroke.baselines.l2p.bezier.stroke_coverage` (itself equal
to the official OpenCV ``draw``): mean l2 (MSE), mean absolute error, largest pixel error, worst stroke. It also reports our renderer
against the official one. With ``--actor`` and ``--image`` it runs the released actor on one image with each renderer (a smoke run of the
rollout and of the producer skeleton, not a result): l2 loss after 5 / 25 / 100 / 500 / 1700 / 2000 / 4250 strokes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brushstroke.baselines.l2p.actor import load_actor
from brushstroke.baselines.l2p.bezier import stroke_coverage
from brushstroke.baselines.l2p.renderer import load_renderer, random_strokes, renderer_error
from brushstroke.baselines.l2p.rollout import RolloutConfig, rollout


@torch.no_grad()
def between(
    a: torch.nn.Module, b: torch.nn.Module, strokes: torch.Tensor, block: int = 250
) -> dict[str, float]:
    """Mean l2 and largest pixel difference between two renderers."""
    mse, mx = [], 0.0
    for i in range(0, strokes.shape[0], block):
        d = a(strokes[i : i + block]) - b(strokes[i : i + block])
        mse.append(d.square().flatten(1).mean(1))
        mx = max(mx, float(d.abs().max()))
    return {"mse": float(torch.cat(mse).mean()), "max_abs": mx}


def main(argv: list[str] | None = None) -> dict[str, object]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--official", type=Path, default=None)
    ap.add_argument("--actor", type=Path, default=None)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--strokes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ours = load_renderer(args.weights, dev)
    strokes = random_strokes(args.strokes, torch.Generator().manual_seed(args.seed), dev)
    truth = stroke_coverage(strokes)
    report: dict[str, object] = {
        "strokes": args.strokes,
        "seed": args.seed,
        "ours_vs_analytic": renderer_error(ours, strokes, truth),
    }
    official = (
        load_renderer(args.official, dev) if args.official and args.official.exists() else None
    )
    if official is not None:
        report["official_vs_analytic"] = renderer_error(official, strokes, truth)
        with torch.no_grad():
            report["ours_vs_official"] = between(ours, official, strokes)
    if official is not None and args.actor and args.image:
        import cv2

        img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)  # BGR, as the official code
        target = (
            torch.from_numpy(cv2.resize(img, (512, 512))).permute(2, 0, 1).float().to(dev) / 255.0
        )
        actor = load_actor(args.actor, dev)
        smoke: dict[str, object] = {}
        for name, net in (("official_renderer", official), ("our_renderer", ours)):
            painting = rollout(actor, net, target, RolloutConfig())
            smoke[name] = {
                f"l2_after_{k}_strokes": float(
                    (painting.canvas(k).to(dev) - target).square().mean()
                )
                for k in (0, 5, 25, 100, 500, 1700, 2000, 4250)
            }
            smoke[name]["n_total"] = painting.n_total  # type: ignore[index]
        report["released_actor_smoke"] = smoke
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
