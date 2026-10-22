"""Write a contact sheet of the brush banks and the canvas texture to ``outputs/`` (not committed).

Usage (repo root): ``python -m scripts.preview_brushes [key.sub=value ...]``, e.g.
``python -m scripts.preview_brushes brush.bank_size=6 output_dir=outputs/brush_preview``.
Rows = brush kinds (flat, oil, round), columns = bank variants; the last row shows the canvas
texture and a mid-gray image shaded with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from brushstroke.config import RunConfig
from brushstroke.renderer.brushes import BRUSH_KINDS, BrushBank
from brushstroke.renderer.canvas import apply_canvas, make_canvas_texture
from brushstroke.utils.config import dump_config, load_config
from brushstroke.utils.io import save_image

PAD = 4
BACKGROUND = 0.15
SHADE_PREVIEW_STRENGTH = 0.25  # deliberately exaggerated (config canvas.strength is 0.06)


def _grid(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    """Paste ``(1, h, w)`` tiles into a ``(3, H, W)`` sheet (white paint on dark background)."""
    th = max(t.shape[1] for r in rows for t in r)
    tw = max(t.shape[2] for r in rows for t in r)
    ncols = max(len(r) for r in rows)
    sheet = torch.full((3, PAD + len(rows) * (th + PAD), PAD + ncols * (tw + PAD)), BACKGROUND)
    for i, row in enumerate(rows):
        for j, tile in enumerate(row):
            y, x = PAD + i * (th + PAD), PAD + j * (tw + PAD)
            sheet[:, y : y + tile.shape[1], x : x + tile.shape[2]] = tile.expand(3, -1, -1)
    return sheet


def main(argv: list[str]) -> Path:
    cfg = load_config(RunConfig, Path("configs/default.yaml"), argv)
    out_dir = Path(cfg.output_dir)
    dump_config(cfg, out_dir)
    size = (cfg.brush.mask_height, cfg.brush.mask_width)
    rows = []
    for kind in BRUSH_KINDS:
        bank = BrushBank.build(kind, cfg.brush.bank_size, size, cfg.brush.seed)
        rows.append([bank[i] for i in range(len(bank))])
    tex = make_canvas_texture(
        size[0] * 2,
        size[1],
        cfg.canvas.seed,
        weave_period=cfg.canvas.weave_period,
        low_freq_amount=cfg.canvas.low_freq_amount,
        low_freq_scale=cfg.canvas.low_freq_scale,
    )
    shaded = apply_canvas(
        torch.full((3, *tex.shape[1:]), 0.6), tex, strength=SHADE_PREVIEW_STRENGTH
    )
    sheet = _grid(rows + [[tex]])
    path = out_dir / "brush_contact_sheet.png"
    save_image(sheet, path)
    save_image(shaded, out_dir / "canvas_shaded.png")
    print(f"wrote {path}")
    return path


if __name__ == "__main__":
    main(sys.argv[1:])
