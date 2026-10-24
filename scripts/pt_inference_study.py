"""Measurements of the Paint Transformer inference (T-015): parity with the official code, final-layer
variants, runtime, refinement. Results are written as JSON under ``docs/experiments/data/pt_inference``
and the tables of ``docs/experiments/pt_inference.md`` are regenerated from them.

Usage (repo root; the first three stages need a GPU and the official code from
``python -m scripts.pt_reference_study fetch``, ``report`` needs neither)::

    python -m scripts.pt_inference_study parity     # ours (official weights) vs official inference.py
    python -m scripts.pt_inference_study variants   # official/gated final layer, counts, runtime, VRAM
    python -m scripts.pt_inference_study refine     # per-patch refinement on 2 images (+ figure)
    python -m scripts.pt_inference_study layers     # figure: the canvas after each layer
    python -m scripts.pt_inference_study report     # tables of docs/experiments/pt_inference.md

The five evaluation images are those of the T-010 study (``data/raw``, centre-cropped to 512 x 512).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

from brushstroke.eval.metrics import image_metrics
from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.inference import PTInferenceConfig, load_pt_model, paint
from brushstroke.stroke_model.pt_sequence import replay_sequence
from brushstroke.stroke_model.refine import RefineConfig
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image, save_image

IMAGES = ["mona_lisa", "starry_night", "great_wave", "pearl_earring", "earth_apollo17"]
REFINE_IMAGES = ["mona_lisa", "great_wave"]
DATA = Path("docs/experiments/data/pt_inference")
DOC = Path("docs/experiments/pt_inference.md")
FIGURES = Path("docs/media")
WORK = Path("outputs/pt_inference")
OFFICIAL_WEIGHTS = Path("data/raw/pt_official/model.pth")
T014_CHECKPOINT = Path("outputs/pt_long/seed2/latest.pt")
BEGIN, END = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
MODELS = {"official weights": OFFICIAL_WEIGHTS, "T-014 seed 2": T014_CHECKPOINT}
TIMING_RUNS, WARMUP_RUNS = 5, 2
REFINE_SETTINGS: dict[str, dict[str, Any]] = {
    "off": {"enabled": False},
    "default (100 it, lr 0.05)": {"enabled": True},
    "colour fixed": {"enabled": True, "optimize_colour": False},
    "short (30 it, lr 0.02)": {"enabled": True, "iters": 30, "lr": 0.02},
    "long (300 it, lr 0.05)": {"enabled": True, "iters": 300},
}


def load_inputs(size: int = 512) -> dict[str, Path]:
    """Centre-crop the evaluation images to ``size`` and store them as PNG (what both pipelines
    read); returns name -> PNG path."""
    WORK.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in IMAGES:
        path = WORK / "in" / f"{name}_{size}.png"
        if not path.exists():
            image = center_crop_resize(load_image(Path("data/raw") / f"{name}.jpg"), (size, size))
            save_image(image, path)
        paths[name] = path
    return paths


def _dump(name: str, payload: object) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / f"{name}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _load(name: str) -> Any:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- parity


def parity() -> list[dict[str, object]]:
    """Our pipeline with the official weights against the float output of the official script."""
    from scripts.pt_reference_study import official_paint

    device = get_device("auto")
    net, _ = load_pt_model(OFFICIAL_WEIGHTS, device)
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    rows: list[dict[str, object]] = []
    jobs = [(name, 512, path) for name, path in load_inputs(512).items()]
    big = center_crop_resize(load_image(Path("data/raw/mona_lisa.jpg")), (1024, 1024))
    save_image(big, WORK / "in" / "mona_lisa_1024.png")
    jobs.append(("mona_lisa", 1024, WORK / "in" / "mona_lisa_1024.png"))
    for name, size, path in jobs:
        official, counts = official_paint(path)
        image = load_image(path).to(device)
        result = paint(net, image, brushes)
        ours = result.canvas.cpu()
        diff = (ours - official).abs()
        official_png = (official * 255).to(torch.uint8)  # what the official script writes
        ours_png = (ours * 255).to(torch.uint8)
        rows.append(
            {
                "image": name,
                "size": size,
                "max_abs_diff": float(diff.max()),
                "mean_abs_diff": float(diff.mean()),
                "pixels_over_1_255": int((diff.amax(0) > 1 / 255).sum()),
                "png_max_abs_diff": int((official_png.int() - ours_png.int()).abs().max()),
                "strokes_official_code": sum(c[1] for c in counts),
                "strokes_ours": result.stats["strokes_official"],
                "layers_official_code": [c[1] for c in counts],
                "layers_ours": result.stats["painted_per_layer"],
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    _dump("parity", rows)
    return rows


# ----------------------------------------------------------------------------- variants


def variants() -> list[dict[str, object]]:
    """Both final-layer variants for both models: counts, metrics, runtime, VRAM, sequence check.

    Timing and VRAM are measured first (median of ``TIMING_RUNS`` runs after ``WARMUP_RUNS``; the
    peak is ``torch.cuda.max_memory_allocated`` since the start of the run, so it includes the
    network, the brushes and the image), before the LPIPS networks of the metrics are loaded.
    """
    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    inputs = load_inputs(512)
    timing: dict[tuple[str, str], dict[str, float]] = {}
    for model_name, path in MODELS.items():
        net, _ = load_pt_model(path, device)
        for name, png in inputs.items():
            image = load_image(png).to(device)
            for _ in range(WARMUP_RUNS):
                paint(net, image, brushes)
            seconds, vram = [], []
            for _ in range(TIMING_RUNS):
                stats = paint(net, image, brushes).stats
                seconds.append(float(stats["runtime_seconds"]))  # type: ignore[arg-type]
                vram.append(float(stats["peak_vram_mb"]))  # type: ignore[arg-type]
            timing[(model_name, name)] = {
                "runtime_seconds_median": statistics.median(seconds),
                "peak_vram_mb": max(vram),
            }
    rows: list[dict[str, object]] = []
    for model_name, path in MODELS.items():
        net, _ = load_pt_model(path, device)
        for name, png in inputs.items():
            image = load_image(png).to(device)
            result = paint(net, image, brushes)
            raw = paint(net, image, brushes, PTInferenceConfig(abs_h=False))
            seq = result.sequence.to(device)
            replay = replay_sequence(seq, brushes)
            generic = seq.render(cfg=_pt_render_config())
            row = {
                "model": model_name,
                "image": name,
                "strokes_official": result.stats["strokes_official"],
                "strokes_gated": result.stats["strokes_gated"],
                "painted_per_layer": result.stats["painted_per_layer"],
                **timing[(model_name, name)],
                "negative_h_fraction": raw.stats["negative_h_fraction"],
                "metrics_official": image_metrics(result.canvas, image),
                "metrics_gated": image_metrics(result.canvas_gated, image),
                "replay_max_abs": float((replay - result.canvas).abs().max()),
                "replay_mean_abs": float((replay - result.canvas).abs().mean()),
                "replay_pixels_over_1_255": int(
                    ((replay - result.canvas).abs().amax(0) > 1 / 255).sum()
                ),
                "generic_mean_abs": float((generic - result.canvas).abs().mean()),
                "generic_pixels_over_1_255": int(
                    ((generic - result.canvas).abs().amax(0) > 1 / 255).sum()
                ),
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if not k.startswith("metrics")}),
                flush=True,
            )
    _dump("variants", rows)
    return rows


def _pt_render_config() -> Any:
    from brushstroke.renderer.pipeline import RenderConfig

    return RenderConfig(mode="pt", pt_alpha="pt_infer")


# ----------------------------------------------------------------------------- refinement


def refine() -> list[dict[str, object]]:
    """Per-patch refinement on two images with the T-014 checkpoint (and the official weights)."""
    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    inputs = load_inputs(512)
    rows: list[dict[str, object]] = []
    panels: dict[str, list[torch.Tensor]] = {}
    for model_name, path in MODELS.items():
        net, _ = load_pt_model(path, device)
        for name in REFINE_IMAGES:
            image = load_image(inputs[name]).to(device)
            for setting, kwargs in REFINE_SETTINGS.items():
                if model_name != "T-014 seed 2" and setting not in (
                    "off",
                    "default (100 it, lr 0.05)",
                ):
                    continue
                result = paint(net, image, brushes, refine=RefineConfig(**kwargs))
                rows.append(
                    {
                        "model": model_name,
                        "image": name,
                        "setting": setting,
                        "runtime_seconds": result.stats["runtime_seconds"],
                        "peak_vram_mb": result.stats["peak_vram_mb"],
                        "strokes": result.stats["strokes_official"],
                        "metrics": image_metrics(result.canvas, image),
                        "refine_layers": result.stats["refine_layers"],
                    }
                )
                print(
                    json.dumps({k: v for k, v in rows[-1].items() if k != "refine_layers"}),
                    flush=True,
                )
                if model_name == "T-014 seed 2" and setting in ("off", "default (100 it, lr 0.05)"):
                    panels.setdefault(name, [image.cpu()]).append(result.canvas.cpu())
    _dump("refine", rows)
    _figure(panels, FIGURES / "pt_inference_refine.jpg")
    return rows


def _figure(panels: dict[str, list[torch.Tensor]], path: Path) -> None:
    """Rows = images; columns = target, PT, PT + refinement; and a 128 px detail crop of each."""
    import numpy as np
    from PIL import Image

    tiles = []
    for tensors in panels.values():
        row = [t.clamp(0, 1) for t in tensors]
        crops = [t[:, 150:278, 180:308] for t in row]  # 128 x 128 detail
        crops = [
            torch.nn.functional.interpolate(c[None], scale_factor=2, mode="nearest")[0]
            for c in crops
        ]
        rows = [torch.cat(row, dim=2), torch.cat(crops, dim=2)]
        tiles.append(
            torch.cat(
                [
                    rows[0],
                    torch.nn.functional.pad(
                        rows[1], [0, rows[0].shape[2] - rows[1].shape[2]], value=1.0
                    ),
                ],
                dim=1,
            )
        )
    sheet = torch.cat(tiles, dim=1)
    arr = (sheet.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((arr.shape[1] // 2, arr.shape[0] // 2), Image.LANCZOS).save(
        path, quality=85, optimize=True
    )


def layers() -> None:
    """Figure: the canvas after each layer (T-014 checkpoint, Mona Lisa) and the two variants."""
    import numpy as np
    from PIL import Image

    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    net, _ = load_pt_model(T014_CHECKPOINT, device)
    image = load_image(load_inputs(512)["mona_lisa"]).to(device)
    result = paint(net, image, brushes)
    panels = [image, *result.layer_canvases[:-1], result.canvas_gated, result.canvas]
    panels = [
        torch.nn.functional.interpolate(p[None].cpu(), size=(256, 256), mode="area")[0]
        for p in panels
    ]
    sheet = torch.cat([torch.cat(panels[:4], dim=2), torch.cat(panels[4:], dim=2)], dim=1)
    arr = (sheet.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(FIGURES / "pt_inference_layers.jpg", quality=85, optimize=True)


# ----------------------------------------------------------------------------- report


def _fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def build_parity_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| image | size | strokes (official code / ours) | max abs diff | mean abs diff | pixels > 1/255 | max diff of the 8-bit PNGs |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['image']} | {r['size']} | {r['strokes_official_code']} / {r['strokes_ours']} | "
            f"{r['max_abs_diff']:.1e} | {r['mean_abs_diff']:.1e} | {r['pixels_over_1_255']} | {r['png_max_abs_diff']} |"
        )
    return "\n".join(lines)


def build_variants_tables(rows: list[dict[str, Any]]) -> str:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        sub = [r for r in rows if r["model"] == model]
        lines = [
            f"**{model}** (512 x 512; strokes per layer 0..K and the shifted final layer):",
            "",
            "| image | strokes official / gated | per layer | L1 off. / gated | PSNR off. / gated | SSIM off. / gated | LPIPS-VGG off. / gated | LPIPS-Alex off. / gated | time (s) | VRAM (MiB) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in sub:
            mo, mg = r["metrics_official"], r["metrics_gated"]
            lines.append(
                f"| {r['image']} | {r['strokes_official']} / {r['strokes_gated']} | {r['painted_per_layer']} | "
                f"{_fmt(mo['l1'])} / {_fmt(mg['l1'])} | {_fmt(mo['psnr'], 2)} / {_fmt(mg['psnr'], 2)} | "
                f"{_fmt(mo['ssim'])} / {_fmt(mg['ssim'])} | {_fmt(mo['lpips_vgg'])} / {_fmt(mg['lpips_vgg'])} | "
                f"{_fmt(mo['lpips_alex'])} / {_fmt(mg['lpips_alex'])} | {r['runtime_seconds_median']:.2f} | {r['peak_vram_mb']:.0f} |"
            )
        mean = lambda key, sub=sub: statistics.mean(r["metrics_official"][key] for r in sub)  # noqa: E731
        meang = lambda key, sub=sub: statistics.mean(r["metrics_gated"][key] for r in sub)  # noqa: E731
        lines.append(
            f"| **mean** | {statistics.mean(r['strokes_official'] for r in sub):.0f} / "
            f"{statistics.mean(r['strokes_gated'] for r in sub):.0f} | | {_fmt(mean('l1'))} / {_fmt(meang('l1'))} | "
            f"{_fmt(mean('psnr'), 2)} / {_fmt(meang('psnr'), 2)} | {_fmt(mean('ssim'))} / {_fmt(meang('ssim'))} | "
            f"{_fmt(mean('lpips_vgg'))} / {_fmt(meang('lpips_vgg'))} | {_fmt(mean('lpips_alex'))} / {_fmt(meang('lpips_alex'))} | "
            f"{statistics.mean(r['runtime_seconds_median'] for r in sub):.2f} | {max(r['peak_vram_mb'] for r in sub):.0f} |"
        )
        out.append("\n".join(lines))
    return "\n\n".join(out)


def build_sequence_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | image | strokes | replay: max abs / mean abs / pixels > 1/255 | generic `StrokeSequence.render`: mean abs / pixels > 1/255 | raw `h < 0` |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['image']} | {r['strokes_official']} | {r['replay_max_abs']:.1e} / {r['replay_mean_abs']:.1e} / "
            f"{r['replay_pixels_over_1_255']} | {r['generic_mean_abs']:.4f} / {r['generic_pixels_over_1_255']} | "
            f"{100 * r['negative_h_fraction']:.1f} % |"
        )
    return "\n".join(lines)


def build_refine_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| model | image | refinement | L1 | PSNR (dB) | SSIM | LPIPS-VGG | LPIPS-Alex | time (s) | patch loss before -> after (layers 0..K+1) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["metrics"]
        layers = " ".join(
            f"{x['loss_before']:.3f}->{x['loss_after']:.3f}" for x in r["refine_layers"]
        )
        lines.append(
            f"| {r['model']} | {r['image']} | {r['setting']} | {_fmt(m['l1'])} | {_fmt(m['psnr'], 2)} | {_fmt(m['ssim'])} | "
            f"{_fmt(m['lpips_vgg'])} | {_fmt(m['lpips_alex'])} | {r['runtime_seconds']:.1f} | {layers or '-'} |"
        )
    return "\n".join(lines)


def report() -> None:
    parts = []
    if (DATA / "parity.json").exists():
        parts.append(
            "### Parity with the official `inference.py` (official weights)\n\n"
            + build_parity_table(_load("parity"))
        )
    if (DATA / "variants.json").exists():
        rows = _load("variants")
        parts.append(
            "### Final-layer variants, stroke counts, runtime and VRAM\n\n"
            + build_variants_tables(rows)
        )
        parts.append(
            "### The exported `StrokeSequence` against the pipeline canvas\n\n"
            + build_sequence_table(rows)
        )
    if (DATA / "refine.json").exists():
        parts.append(
            "### Per-patch refinement\n\n"
            + build_refine_table(_load("refine"))
            + "\n\n![refinement](../media/pt_inference_refine.jpg)"
        )
    block = f"{BEGIN}\n\n" + "\n\n".join(parts) + f"\n\n{END}"
    text = DOC.read_text(encoding="utf-8")
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        text = head + block + rest.split(END, 1)[1]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    DOC.write_text(text, encoding="utf-8")
    print(block)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("stage", choices=["parity", "variants", "refine", "layers", "report"])
    args = parser.parse_args(argv)
    {"parity": parity, "variants": variants, "refine": refine, "layers": layers, "report": report}[
        args.stage
    ]()


if __name__ == "__main__":
    main(sys.argv[1:])
