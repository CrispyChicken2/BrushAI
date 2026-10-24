"""M2 demos: Paint Transformer painting -> StrokeSequence -> animation, next to the M1 baseline (T-018).

One command regenerates everything::

    python -m scripts.make_m2_demos [--insecure] [--only NAME] [--checkpoint PATH] [--reuse]

For every ``configs/demos/m1_*.yaml`` input (the three M1 demo inputs): fetch/verify the input, paint it
with the Paint Transformer pipeline (``paint_pt``, config ``configs/demos/m2.yaml``), take the M1
optimization baseline of the same input from ``outputs/m1/<name>`` (painted with the M1 demo
configuration when missing: about 9.4 min each, bitwise reproducible), animate both sequences with the
animation module and write, committed:

* ``docs/media/m2_<name>.gif``: the PT painting stroke by stroke (480 px);
* ``docs/media/m2_<name>_vs_m1.gif``: side by side, M1 baseline (left) and PT (right), same timeline;
* ``docs/media/m2_<name>_compare.jpg``: input | M1 baseline | PT, final images;
* ``docs/media/m2_results.md``: metrics of both, stroke counts, runtimes, sequence hash.

The default checkpoint is the T-014 selected model (``outputs/pt_long/seed2/latest.pt``, not committed);
``--checkpoint data/raw/pt_official/model.pth`` uses the released weights
(``python -m scripts.pt_reference_study fetch``) and needs no training.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brushstroke.animation import (
    AnimationConfig,
    export_frames,
    make_frame_renderer,
)
from brushstroke.animation.frames import output_size
from brushstroke.stroke_model.inference import PaintPTConfig
from brushstroke.strokes import StrokeSequence
from brushstroke.utils.config import dump_config, load_config
from scripts import paint_pt
from scripts.fetch_demo_inputs import DEMO_DIR, fetch_demo_inputs
from scripts.make_m1_demos import MAX_GIF_BYTES, DemoConfig, paint_demo

MEDIA = Path("docs/media")
CONFIG = DEMO_DIR / "m2.yaml"


@dataclass
class M2Config:
    """``configs/demos/m2.yaml``: PT painting + animation settings of the M2 demos."""

    paint: PaintPTConfig = field(default_factory=PaintPTConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)
    m1_dir: str = "outputs/m1"  # M1 baseline runs (painted here when missing)
    panel: int = 384  # side of one panel of the side-by-side GIF and of the compare image


def sequence_sha256(seq: StrokeSequence) -> str:
    """Hash of what defines the painting (parameters and tags), not of timings."""
    h = hashlib.sha256()
    for t in (seq.params, seq.scale_level, seq.patch_id):
        if t is not None:
            h.update(t.cpu().numpy().tobytes())
    h.update(json.dumps(seq.canvas_size).encode())
    return h.hexdigest()


def m1_baseline(name: str, m1_cfg: DemoConfig, image: Path, m1_dir: Path) -> tuple[Path, dict]:
    """``(run directory, metrics)`` of the M1 baseline of a demo, painted when not stored."""
    run = m1_dir / name
    if not (run / "sequence.npz").exists() or not (run / "metrics.json").exists():
        print(f"== {name}: painting the M1 baseline (about 9 min)", flush=True)
        run.mkdir(parents=True, exist_ok=True)
        seq, _, metrics = paint_demo(m1_cfg, image, run)
        (run / "metrics.json").write_text(
            json.dumps({"name": name, "metrics": metrics}, indent=2), encoding="utf-8"
        )
    stored = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return run, stored["metrics"]


def labelled(image: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 8 * len(text) + 8, 15], fill=(255, 255, 255))
    draw.text((4, 2), text, fill=(0, 0, 0))
    return image


def compare_image(paths: dict[str, Path], dest: Path, side: int) -> None:
    """Side-by-side JPEG of the labelled images ``label -> png``."""
    tiles = [
        labelled(Image.open(p).convert("RGB").resize((side, side), Image.LANCZOS), label)
        for label, p in paths.items()
    ]
    sheet = Image.new("RGB", (len(tiles) * side + 8 * (len(tiles) - 1), side), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (i * (side + 8), 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=88)


def side_by_side_frames(left, right):  # type: ignore[no-untyped-def]
    """Concatenate two frame iterators horizontally (same number of frames)."""
    for a, b in zip(left, right, strict=True):
        yield np.concatenate([a, b], axis=1)


def write_gif(files: dict[str, Path], dest: Path) -> float:
    size = files["gif"].stat().st_size
    if size > MAX_GIF_BYTES:
        raise RuntimeError(f"{dest.name}: GIF is {size / 1e6:.2f} MB > 4 MB; lower gif_width/fps")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(files["gif"], dest)
    return size / 1e6


def results_markdown(rows: list[dict]) -> str:
    lines = [
        "# M2 demos: results",
        "",
        "Generated by `python -m scripts.make_m2_demos` (do not edit by hand): Paint Transformer "
        "(this repository's model, T-014 checkpoint) against the M1 optimization baseline of the "
        "same input (`docs/media/m1_results.md`), 512 px.",
        "",
        "| demo | input | method | strokes | L1 | PSNR (dB) | SSIM | LPIPS-VGG | LPIPS-Alex | runtime (s) | peak VRAM (GB) | sequence sha256 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for method, m in (("M1 optimization", r["m1"]), ("Paint Transformer", r["pt"])):
            lines.append(
                f"| {r['name']} | {r['image_id']} | {method} | {m['num_strokes']:.0f} | {m['l1']:.4f} | "
                f"{m['psnr']:.2f} | {m['ssim']:.3f} | {m['lpips_vgg']:.3f} | {m['lpips_alex']:.3f} | "
                f"{m['runtime_seconds']:.1f} | {m['peak_vram_mb'] / 1024:.1f} | "
                f"{'`' + r['sequence_sha256'][:12] + '`' if method.startswith('Paint') else '(see m1_results.md)'} |"
            )
    lines += [
        "",
        "Notes: the PT painting is the `official` final-layer variant (every stroke of the shifted last layer painted) "
        "from the checkpoint named in `configs/demos/m2.yaml`; its runtime is one painting after the model is loaded "
        "(first-call CUDA warm-up included). The M1 baseline runs 2,037 strokes with the default configuration. "
        "The GIF of the PT painting is rendered with the PT brush images at the final resolution (generic render, "
        "`docs/design/paint_transformer.md` section 14), so its last frame differs slightly from the pipeline canvas "
        "(`last_frame_mean_abs` in `outputs/m2/<name>/metrics.json`).",
        "",
        "GIFs: "
        + ", ".join(
            f"[{r['name']}](m2_{r['name']}.gif) / [vs M1](m2_{r['name']}_vs_m1.gif) / "
            f"[compare](m2_{r['name']}_compare.jpg) ({r['gif_mb']:.2f} + {r['sbs_gif_mb']:.2f} MB)"
            for r in rows
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out", type=Path, default=Path("outputs/m2"))
    parser.add_argument("--media", type=Path, default=MEDIA)
    parser.add_argument("--checkpoint", type=Path, default=None, help="overrides the config")
    parser.add_argument("--only", action="append", help="demo name without prefix (repeatable)")
    parser.add_argument("--reuse", action="store_true", help="reuse an existing PT sequence")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification (download)")
    args = parser.parse_args(argv)

    cfg = load_config(M2Config, args.config)
    if args.checkpoint is not None:
        cfg.paint.checkpoint = str(args.checkpoint)
    paths = fetch_demo_inputs(args.insecure)
    rows: list[dict] = []
    for m1_path in sorted(DEMO_DIR.glob("m1_*.yaml")):
        m1_cfg = load_config(DemoConfig, m1_path)
        name = m1_cfg.name.removeprefix("m1_")
        if args.only and name not in args.only:
            continue
        image = paths[m1_cfg.image_id]
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        pt_dir = out / "pt"
        if args.reuse and (pt_dir / "sequence.npz").exists():
            metrics = json.loads((pt_dir / "metrics.json").read_text(encoding="utf-8"))
        else:
            print(f"== {name}: painting {m1_cfg.image_id} with the Paint Transformer", flush=True)
            dump_config(cfg, out)
            metrics = paint_pt.main(
                [
                    "--input",
                    str(image),
                    "--out",
                    str(pt_dir),
                    "--checkpoint",
                    cfg.paint.checkpoint,
                    f"seed={cfg.paint.seed}",
                    f"height={cfg.paint.height}",
                    f"width={cfg.paint.width}",
                    f"inference.final_layer={cfg.paint.inference.final_layer}",
                ]
            )
        seq = StrokeSequence.load(pt_dir / "sequence.npz")
        variant = cfg.paint.inference.final_layer
        pt_metrics = dict(metrics[variant]) | {
            "num_strokes": float(len(seq)),
            "runtime_seconds": float(metrics["runtime_seconds"]),
            "peak_vram_mb": float(metrics["peak_vram_mb"]),
        }
        m1_run, m1_metrics = m1_baseline(m1_cfg.name, m1_cfg, image, Path(cfg.m1_dir))
        m1_seq = StrokeSequence.load(m1_run / "sequence.npz")

        # PT painting alone
        print(f"== {name}: animating {len(seq)} PT strokes", flush=True)
        renderer = make_frame_renderer(seq, cfg.animation)
        h, w = output_size(seq, cfg.animation)
        files = export_frames(renderer.frames(), out / f"m2_{name}", (w, h), cfg.animation)
        gif_mb = write_gif(files, args.media / f"m2_{name}.gif")
        renderer.reset()
        last = renderer.render_frame(renderer.timeline.paint_frames - 1).cpu().numpy()
        canvas = np.asarray(Image.open(pt_dir / "final.png").convert("RGB"), dtype=np.float32) / 255
        last_mean_abs = float(np.abs(last.transpose(1, 2, 0) - canvas).mean())

        # side by side with the M1 baseline, same timeline
        panel = cfg.panel
        pt_panel = dataclasses.replace(cfg.animation, height=panel, width=panel)
        m1_panel = dataclasses.replace(
            m1_cfg.animation,
            height=panel,
            width=panel,
            fps=cfg.animation.fps,
            duration=cfg.animation.duration,
            hold=cfg.animation.hold,
        )
        left = make_frame_renderer(m1_seq, m1_panel)
        right = make_frame_renderer(seq, pt_panel)
        sbs_cfg = dataclasses.replace(cfg.animation, formats=["gif"], gif_width=2 * panel)
        sbs = export_frames(
            side_by_side_frames(left.frames(), right.frames()),
            out / f"m2_{name}_vs_m1",
            (2 * panel, panel),
            sbs_cfg,
        )
        sbs_mb = write_gif(sbs, args.media / f"m2_{name}_vs_m1.gif")

        compare_image(
            {
                "input": pt_dir / "target.png",
                "M1 optimization": m1_run / "final.png",
                "Paint Transformer": pt_dir / "final.png",
            },
            args.media / f"m2_{name}_compare.jpg",
            panel,
        )
        row = {
            "name": name,
            "image_id": m1_cfg.image_id,
            "pt": pt_metrics,
            "m1": m1_metrics,
            "sequence_sha256": sequence_sha256(seq),
            "gif_mb": gif_mb,
            "sbs_gif_mb": sbs_mb,
            "last_frame_mean_abs": last_mean_abs,
        }
        (out / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(
            json.dumps({k: row[k] for k in ("name", "pt", "m1", "gif_mb", "sbs_gif_mb")}, indent=1)
        )
    if rows and not args.only:
        args.media.mkdir(parents=True, exist_ok=True)
        (args.media / "m2_results.md").write_text(results_markdown(rows), encoding="utf-8")
        print(f"wrote {args.media / 'm2_results.md'}")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
