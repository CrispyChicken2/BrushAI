"""The evaluation harness command line (T-026): run producers on the evaluation set, write the tables and the figure.

    python -m scripts.eval_run run    --split dev --methods optim pt --out outputs/eval/dev_smoke
    python -m scripts.eval_run report --out outputs/eval/dev_smoke --md docs/experiments/eval_harness_smoke.md
    python -m scripts.eval_run tau    --out outputs/eval/dev_smoke [--write]

``run`` needs the prepared targets (``scripts.fetch_eval_set``), the saliency maps (``scripts.compute_saliency``; their SHA-256
are checked against the record before anything is scored) and the checkpoints named in ``configs/eval/protocol.yaml``. It is
resumable (existing records are kept, ``--force`` redoes them), processes images in sorted id order and records one JSON per
(method, image, seed). **Only the dev split runs here**: the test split is one run after a code freeze (T-032, ``--allow-test``).

``tau`` computes the strokes-to-target thresholds of section 5 from DEV records: the strongest baseline is the one with the lowest
dev mean LPIPS-Alex at 1000 strokes, and ``tau_B`` is the median over the dev images of the LPIPS-Alex it reaches at ``B``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from brushstroke.eval import report
from brushstroke.eval.dataset import load_manifest, pixel_sha256
from brushstroke.eval.producers import OptimProducer, Producer, PTProducer
from brushstroke.eval.protocol import PROTOCOL, Protocol, load_protocol
from brushstroke.eval.runner import ImageContext, run_harness
from brushstroke.eval.saliency import map_sha256

SALIENCY_RECORD = Path("docs/experiments/data/saliency/saliency_maps.json")


def load_images(
    proto: Protocol, split: str, ids: list[str] | None, device: torch.device
) -> list[ImageContext]:
    """Targets and saliency maps of a split; every map is checked against the SHA-256 recorded by T-025."""
    _, images = load_manifest(Path(proto.eval_set))
    record = json.loads(SALIENCY_RECORD.read_text(encoding="utf-8"))["images"]
    out = []
    for im in images:
        if (split == "dev") != im.dev or (ids and im.id not in ids):
            continue
        pixels = np.asarray(
            Image.open(Path(proto.eval_root) / "prepared" / f"{im.id}.png").convert("RGB"),
            dtype=np.float32,
        )
        if pixel_sha256(pixels.astype(np.uint8)) != im.prepared_sha256:
            raise SystemExit(
                f"target {im.id} differs from the manifest prepared_sha256 (stale data/raw/eval: python -m scripts.fetch_eval_set)"
            )
        target = torch.from_numpy(pixels / 255.0).permute(2, 0, 1).contiguous().to(device)
        maps = {}
        for name in proto.saliency_methods:
            m = np.load(Path(proto.saliency_dir) / name / f"{im.id}.npy")
            if map_sha256(m) != record[im.id][name]["sha256"]:
                raise SystemExit(
                    f"saliency map {name}/{im.id} differs from the recorded hash (python -m scripts.compute_saliency)"
                )
            maps[name] = torch.from_numpy(m).to(device)
        out.append(ImageContext(im.id, split, im.category, im.stratum, target, maps))
    return out


def build_producers(proto: Protocol, names: list[str], device: torch.device) -> list[Producer]:
    producers: list[Producer] = []
    for name in names:
        spec = proto.producers[name]
        if name == "optim":
            producers.append(OptimProducer(spec["config"], device))
        elif name == "pt":
            producers.append(PTProducer(spec["checkpoints"], spec["config"], device))
        else:
            raise SystemExit(f"unknown method {name!r}")
    return producers


def cmd_run(args: argparse.Namespace) -> None:
    proto = load_protocol()
    if args.split != "dev" and not args.allow_test:
        raise SystemExit(
            "only the dev split runs here; the test split is T-032 (one run after the code freeze)"
        )
    device = torch.device(args.device)
    images = load_images(proto, args.split, args.images, device)
    if args.limit:
        images = sorted(images, key=lambda c: c.id)[: args.limit]
    seeds = {
        "optim": args.optim_seeds or list(proto.seeds),
        "pt": args.pt_seeds or list(proto.seeds),
    }
    counts = run_harness(
        build_producers(proto, args.methods, device),
        images,
        seeds,
        proto,
        args.out,
        args.force,
        timed=not args.untimed,
    )
    print(json.dumps(counts))


def cmd_report(args: argparse.Namespace) -> None:
    proto = load_protocol()
    res = report.Results(report.load_records(args.out), report.load_controls(args.out), proto)
    env = json.loads((args.out / "environment.json").read_text(encoding="utf-8"))
    splits = sorted({r["split"] for r in res.records})
    parts = [
        f"# Evaluation harness run: `{args.out}` ({', '.join(splits)} split)\n",
        "Generated by `python -m scripts.eval_run report`. A dagger † marks a metric that is in the method's own loss (docs/design/evaluation.md 2.1). "
        "Cells are `mean ± std over images [95 % bootstrap interval of the mean] (mean over images of the replicate std)`.\n",
        "## Run\n\n" + report.summary_lines(res, env),
        "## B1 final metrics at the budgets\n\n" + report.table_b1(res),
        "## Controls (section 2.2)\n\n" + report.table_controls(res),
        "## B2 anytime AUC\n\n" + report.table_b2(res),
        "## B3 saliency-weighted LPIPS\n\n" + report.table_b3(res),
        "## B4 strokes-to-target\n\n" + report.table_b4(res),
        "## B5 runtime and VRAM\n\n" + report.table_b5(res),
        "## Paired statistics\n\n" + report.stats_tables(res),
    ]
    if args.figure:
        fig = report.figure(res, args.figure)
        parts.append(f"## Figure\n\n![anytime curves]({Path(fig).name})")
    text = "\n\n".join(parts) + "\n"
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]  # λ, ±, † in the tables
    print(text)


def compute_tau(res: report.Results, proto: Protocol) -> dict[str, object]:
    """Section 5: the strongest baseline (lowest dev mean LPIPS-Alex at 1000), tau_B = median over dev images of its LPIPS-Alex at B."""
    if any(r["split"] != "dev" for r in res.records):
        raise SystemExit("tau is computed from dev records only")
    means = {}
    for method in res.methods:
        if res.reachable(method, 1000):
            values = res.per_image(method, lambda r: res.at_budget(r, "lpips_alex", 1000))
            means[method] = statistics.fmean(m for m, _ in values.values())
    best = min(means, key=lambda m: means[m])
    values: dict[int, float] = {}
    for budget in (500, 1000, 2000):
        per = res.per_image(best, lambda r, b=budget: res.at_budget(r, "lpips_alex", b))
        values[budget] = round(float(np.median([m for m, _ in per.values()])), 4)
    return {
        "strongest_baseline": best,
        "dev_mean_lpips_alex_at_1000": {m: round(v, 4) for m, v in means.items()},
        "values": values,
    }


def cmd_tau(args: argparse.Namespace) -> None:
    proto = load_protocol()
    res = report.Results(report.load_records(args.out), report.load_controls(args.out), proto)
    tau = compute_tau(res, proto)
    print(json.dumps(tau, indent=1))
    if not args.write:
        return
    text = PROTOCOL.read_text(encoding="utf-8")
    block = (
        "tau:\n"
        f"  status: provisional      # from the baselines available on the dev split ({', '.join(res.methods)}); final values are written when Learning to Paint (T-030) is in, before the test run\n"
        f"  strongest_baseline: {tau['strongest_baseline']}\n"
        f"  values: {{500: {tau['values'][500]}, 1000: {tau['values'][1000]}, 2000: {tau['values'][2000]}}}\n"  # type: ignore[index]
    )
    new, n = re.subn(r"tau:\n(?:  .*\n)+", block, text, count=1)
    if n != 1:
        raise SystemExit("could not find the tau block of protocol.yaml")
    PROTOCOL.write_text(new, encoding="utf-8", newline="\n")
    print(f"wrote {PROTOCOL}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    r = sub.add_parser("run")
    r.add_argument("--split", default="dev", choices=["dev", "test"])
    r.add_argument("--allow-test", action="store_true", help="T-032 only")
    r.add_argument("--methods", nargs="+", default=["optim", "pt"])
    r.add_argument("--optim-seeds", type=int, nargs="*")
    r.add_argument("--pt-seeds", type=int, nargs="*")
    r.add_argument("--images", nargs="*", help="restrict to these image ids")
    r.add_argument("--limit", type=int, default=0, help="the first N images (sorted ids)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--force", action="store_true")
    r.add_argument(
        "--untimed",
        action="store_true",
        help="skip the idle-GPU check (runtimes are then indicative)",
    )
    rep = sub.add_parser("report")
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--md", type=Path)
    rep.add_argument("--figure", type=Path)
    t = sub.add_parser("tau")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--write", action="store_true", help="write the values into configs/eval/protocol.yaml"
    )
    args = parser.parse_args(argv)
    {"run": cmd_run, "report": cmd_report, "tau": cmd_tau}[args.stage](args)


if __name__ == "__main__":
    main(sys.argv[1:])
