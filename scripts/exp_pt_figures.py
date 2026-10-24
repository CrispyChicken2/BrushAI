"""Figures 5, 6 and 7 of the Paint Transformer paper on our models (T-016).

Fig. 5: effect of the number of layers K (the official procedure on the image resized to ``32 * 2^K``,
shown at 512 px, nearest). Fig. 6: the full model and the four loss ablations on one image. Fig. 7: the
same trained network with the brush images replaced by a rectangle and a disc. Protocol:
``docs/experiments/pt_reproduction.md``. Written to ``docs/media/pt_repro_fig{5,6,7}.jpg`` (<= 500 KB).

    python -m scripts.exp_pt_figures [--image mona_lisa] [--checkpoint outputs/pt_long/seed2/latest.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from brushstroke.renderer.pt import load_pt_brushes
from brushstroke.stroke_model.inference import PTInferenceConfig, load_pt_model, paint
from brushstroke.utils.device import get_device
from brushstroke.utils.io import center_crop_resize, load_image
from scripts.exp_pt_table1 import ABLATIONS
from scripts.pt_repro_common import DATA, FIGURES

PANEL = 256


def _tile(img: torch.Tensor, label: str) -> Image.Image:
    small = F.interpolate(img[None].cpu(), size=(PANEL, PANEL), mode="area")[0]
    arr = (small.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    tile = Image.fromarray(arr)
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, 8 * len(label) + 6, 14], fill=(255, 255, 255))
    draw.text((3, 1), label, fill=(0, 0, 0))
    return tile


def _sheet(tiles: list[Image.Image], columns: int, path: Path) -> None:
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * PANEL, rows * PANEL), (255, 255, 255))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * PANEL, (i // columns) * PANEL))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85, optimize=True)
    print(path, path.stat().st_size // 1024, "KB")


def _l1(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"{float((a - b).abs().mean()):.3f}"


def figures(image_name: str, checkpoint: Path) -> dict[str, object]:
    device = get_device("auto")
    brushes = load_pt_brushes("inference", "data/raw/pt_brush", device)
    image = center_crop_resize(load_image(Path("data/raw") / f"{image_name}.jpg"), (512, 512)).to(
        device
    )
    net, _ = load_pt_model(checkpoint, device)
    info: dict[str, object] = {}

    # Fig. 5: number of layers
    tiles = [_tile(image, "target")]
    for k in (1, 2, 3, 4):
        small = center_crop_resize(image.cpu(), (32 * 2**k, 32 * 2**k)).to(device)
        result = paint(net, small, brushes)
        up = F.interpolate(result.canvas[None], size=(512, 512), mode="nearest")[0]
        tiles.append(
            _tile(
                up, f"K={k} ({int(result.stats['strokes_official'])} strokes) L1 {_l1(up, image)}"
            )
        )  # type: ignore[call-overload]
    _sheet(tiles, 5, FIGURES / "pt_repro_fig5.jpg")

    # Fig. 6: loss ablations (seed 1 of each setting)
    tiles = [_tile(image, "target")]
    models = {"full": Path("outputs/pt_full/seed1/latest.pt")} | {
        label: Path(f"outputs/pt_abl/{name}/seed1/latest.pt") for name, label in ABLATIONS.items()
    }
    for label, path in models.items():
        if not path.exists():
            continue
        model, _ = load_pt_model(path, device)
        result = paint(model, image, brushes)
        tiles.append(_tile(result.canvas, f"{label} L1 {_l1(result.canvas, image)}"))
    _sheet(tiles, 3, FIGURES / "pt_repro_fig6.jpg")

    # Fig. 7: other brushes (binary masks of the size of the official images, same on both orientations)
    size = brushes.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    rectangle = torch.ones(size, size)
    disc = (
        ((xx - size / 2 + 0.5) / (size / 2)) ** 2 + ((yy - size / 2 + 0.5) / (size / 2)) ** 2 <= 1
    ).float()
    tiles = [_tile(image, "target")]
    l1: dict[str, float] = {}
    for label, mask in (("PT brush", None), ("rectangle", rectangle), ("disc", disc)):
        alt = brushes if mask is None else mask[None, None].repeat(2, 1, 1, 1).to(device)
        result = paint(net, image, alt, PTInferenceConfig())
        l1[label] = float((result.canvas - image).abs().mean())
        tiles.append(_tile(result.canvas, f"{label} L1 {l1[label]:.3f}"))
    _sheet(tiles, 4, FIGURES / "pt_repro_fig7.jpg")
    info["fig7_l1"] = l1
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--image", default="mona_lisa")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/pt_long/seed2/latest.pt"))
    args = parser.parse_args(argv)
    info = figures(args.image, args.checkpoint)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "figures.json").write_text(json.dumps(info, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
