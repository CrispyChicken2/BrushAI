"""Bounding-box rasterization path of the renderer (T-027): equivalence with the full-canvas path."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
import torch

from brushstroke.baselines.optim.config import PaintOptimConfig
from brushstroke.baselines.optim.optimize import config_hash
from brushstroke.renderer.brushes import get_brush_bank
from brushstroke.renderer.params import canonicalize
from brushstroke.renderer.pipeline import RenderConfig, render, render_progressive
from brushstroke.renderer.rasterize import (
    brush_masks,
    rasterize_flat,
    rasterize_window,
    render_alphas,
    stroke_boxes,
)

# The window grid is computed by a matrix product instead of F.affine_grid: the sampling coordinates differ by
# float32 rounding (~1e-7), which the bilinear sampler turns into alpha differences of at most ~1e-4 on the
# steepest brush texels; a different chunking adds only accumulation-order rounding. Image values are in [0, 1].
ATOL_SAME_CHUNKS = 2e-4
ATOL_OTHER_CHUNKS = 2e-4
GRAD_RTOL = 2e-3  # relative to the largest gradient entry


@pytest.fixture(scope="module")
def bank():  # type: ignore[no-untyped-def]
    return get_brush_bank("oil", 4, (32, 64), 0)


def random_strokes(n: int, seed: int = 0, size: tuple[float, float] = (0.03, 0.25)) -> torch.Tensor:
    """Strokes with centres up to 10 % outside the canvas (border crossing) and any rotation."""
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(1, n, 9, generator=g)
    p[..., 0:2] = p[..., 0:2] * 1.2 - 0.1
    p[..., 2:4] = size[0] + (size[1] - size[0]) * p[..., 2:4]
    return p


def cfg(**kw: object) -> RenderConfig:
    return RenderConfig(**{"chunk_size": 16, "bbox_max_fraction": 1.01, **kw})  # type: ignore[arg-type]


@pytest.mark.parametrize("hw", [(64, 64), (48, 80)])
def test_window_alpha_is_the_slice_of_the_full_alpha(bank, hw) -> None:  # type: ignore[no-untyped-def]
    h, w = hw
    p = random_strokes(40)
    masks = brush_masks(bank)
    idx = torch.arange(40) % masks.shape[0]
    flat = canonicalize(p, h, w)[0]
    full = rasterize_flat(flat, masks[idx], h, w)
    boxes = stroke_boxes(p, (masks.shape[-2], masks.shape[-1]), h, w)[0]
    for i in range(40):
        y0, y1, x0, x1 = (int(v) for v in boxes[i])
        if y1 <= y0 or x1 <= x0:
            assert float(full[i].abs().max()) == 0.0
            continue
        win = rasterize_window(flat[i : i + 1], masks[idx[i : i + 1]], h, w, (y0, y1, x0, x1))
        torch.testing.assert_close(win, full[i : i + 1, :, y0:y1, x0:x1], atol=1e-4, rtol=0)


def test_the_box_holds_every_nonzero_alpha_pixel_for_any_rotation_and_border(bank) -> None:  # type: ignore[no-untyped-def]
    h, w = 64, 96
    p = random_strokes(300, seed=3, size=(0.01, 0.6))
    p[:, :60, 4] = torch.linspace(0, 1, 60)  # every rotation
    p[:, 60:90, 0] = 1.1  # beyond the right edge: some wholly outside
    idx = (torch.arange(300) % 4)[None]
    alphas = render_alphas(p, bank, h, w, brush_index=idx)[0, :, 0]
    boxes = stroke_boxes(p, (32, 64), h, w)[0]
    for i in range(300):
        y0, y1, x0, x1 = (int(v) for v in boxes[i])
        outside = alphas[i].clone()
        outside[y0:y1, x0:x1] = 0
        assert float(outside.max()) == 0.0, f"stroke {i} has alpha outside its box"


def test_forward_matches_the_full_path_with_the_same_chunks(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(120)
    full = render(p, bank, 64, 96, cfg=cfg())
    fast = render(p, bank, 64, 96, cfg=cfg(bbox=True))
    assert float((full - fast).abs().max()) <= ATOL_SAME_CHUNKS


def test_forward_matches_with_smaller_bbox_chunks(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(120, seed=1)
    full = render(p, bank, 64, 96, cfg=cfg())
    fast = render(p, bank, 64, 96, cfg=cfg(bbox=True, bbox_chunk=4))
    assert float((full - fast).abs().max()) <= ATOL_OTHER_CHUNKS


def test_progressive_checkpoints_match(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(50, seed=2)
    cps = [0, 7, 33, 50]
    full = render_progressive(p, bank, 48, 48, cps, cfg=cfg())
    fast = render_progressive(p, bank, 48, 48, cps, cfg=cfg(bbox=True, bbox_chunk=5))
    assert float((full - fast).abs().max()) <= ATOL_OTHER_CHUNKS


@pytest.mark.parametrize(
    ("sampler", "ckpt", "bbox_chunk"),
    [("grid_sample", False, 0), ("grid_sample", True, 4), ("gather", False, 0)],
)
def test_gradients_match_the_full_path(bank, sampler, ckpt, bbox_chunk) -> None:  # type: ignore[no-untyped-def]
    p0 = random_strokes(60, seed=4)
    target = torch.rand(1, 3, 48, 80, generator=torch.Generator().manual_seed(9))
    grads = []
    for bbox in (False, True):
        p = p0.clone().requires_grad_()
        img = render(
            p,
            bank,
            48,
            80,
            cfg=cfg(bbox=bbox, sampler=sampler, use_checkpoint=ckpt, bbox_chunk=bbox_chunk),
        )
        ((img - target) ** 2).mean().backward()  # type: ignore[no-untyped-call]
        assert p.grad is not None
        grads.append(p.grad.clone())
    scale = float(grads[0].abs().max())
    assert scale > 0
    assert float((grads[0] - grads[1]).abs().max()) <= GRAD_RTOL * scale


def test_a_stroke_wholly_outside_the_canvas_changes_nothing(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(8)
    p[:, 3:5, 0] = 3.0
    ref = render(p, bank, 40, 40, cfg=cfg(bbox=False))
    out = render(p, bank, 40, 40, cfg=cfg(bbox=True))
    assert float((ref - out).abs().max()) <= ATOL_SAME_CHUNKS
    only_out = p[:, 3:5].clone()
    white = render(only_out, bank, 40, 40, cfg=cfg(bbox=True))
    assert float((white - 1.0).abs().max()) == 0.0


def test_a_dense_chunk_falls_back_to_the_full_path(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(16, size=(0.8, 1.0))
    a = render(p, bank, 32, 32, cfg=cfg(bbox=False, bbox_max_fraction=0.75))
    b = render(p, bank, 32, 32, cfg=cfg(bbox=True, bbox_max_fraction=0.75))
    assert torch.equal(a, b)  # same code path, bitwise equal


def test_rendering_is_deterministic(bank) -> None:  # type: ignore[no-untyped-def]
    p = random_strokes(60, seed=5)
    a = render(p, bank, 48, 48, cfg=cfg(bbox=True, bbox_chunk=8))
    b = render(p, bank, 48, 48, cfg=cfg(bbox=True, bbox_chunk=8))
    assert torch.equal(a, b)


def test_default_is_off_and_the_config_hash_ignores_default_bbox_fields() -> None:
    assert RenderConfig().bbox is False
    c = PaintOptimConfig()
    legacy = json.dumps(
        [
            {
                k: v
                for k, v in dataclasses.asdict(x).items()
                if k not in ("bbox", "bbox_chunk", "bbox_max_fraction")
            }
            for x in (c.optim, c.loss, c.brush, c.render)
        ],
        sort_keys=True,
    )
    legacy_hash = hashlib.sha256(legacy.encode()).hexdigest()[
        :12
    ]  # what main computed before T-027
    h_off = config_hash(c.optim, c.loss, c.brush, c.render)
    h_on = config_hash(c.optim, c.loss, c.brush, dataclasses.replace(c.render, bbox=True))
    assert h_off == legacy_hash
    assert h_off != h_on
