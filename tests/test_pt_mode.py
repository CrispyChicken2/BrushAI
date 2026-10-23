"""Tests of the Paint Transformer compatible rendering mode (T-011), against the official code when
available."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brushstroke.renderer import ours_to_pt, patch_to_canvas, pt_to_ours, render
from brushstroke.renderer.pipeline import RenderConfig
from brushstroke.renderer.pt import (
    dilation,
    erosion,
    load_pt_brushes,
    pt_composite,
    pt_param2stroke,
    render_pt,
    render_pt_progressive,
)
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
needs_official = pytest.mark.skipif(
    not (po.have_brushes() and po.have_official()),
    reason="official code/assets missing: run fetch_pt_brushes + pt_reference_study fetch",
)
TOLERANCE = 1.0 / 255.0
DUMMY_BRUSH = torch.ones(1, 1, 8, 8)  # the default-mode brush argument, ignored in PT mode


def random_params(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, 8, generator=g)
    p[:, 0:2] = 0.15 + 0.7 * p[:, 0:2]
    p[:, 2:4] = 0.08 + 0.55 * p[:, 2:4]
    return p


# ------------------------------------------------------------------ assets and basic behaviour


def test_missing_assets_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fetch_pt_brushes"):
        load_pt_brushes("inference", tmp_path)
    with pytest.raises(ValueError):
        load_pt_brushes("other")
    cfg = RenderConfig(mode="pt", pt_brush_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="fetch_pt_brushes"):
        render(torch.rand(1, 2, 9), DUMMY_BRUSH, 32, 32, cfg=cfg)


def test_morphology_matches_unfold_reference() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, 20, 17, generator=g)

    def unfold_op(t: torch.Tensor, pad_value: float, reduce: str) -> torch.Tensor:
        padded = torch.nn.functional.pad(t, [1, 1, 1, 1], value=pad_value)
        cols = torch.nn.functional.unfold(padded, 3).view(t.shape[0], t.shape[1], -1, *t.shape[2:])
        return cols.max(dim=2)[0] if reduce == "max" else cols.min(dim=2)[0]

    assert torch.equal(dilation(x), unfold_op(x, -1e9, "max"))
    assert torch.equal(erosion(x), unfold_op(x, 1e9, "min"))


def test_composite_formula_and_decisions() -> None:
    canvas = torch.rand(2, 3, 8, 8)
    fg, al = torch.rand(2, 3, 3, 8, 8), torch.rand(2, 3, 3, 8, 8)
    dec = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    out = pt_composite(canvas, fg, al, dec)
    ref = canvas.clone()
    for k in range(3):
        a = al[:, k] * dec[:, k].view(-1, 1, 1, 1)
        ref = fg[:, k] * a + ref * (1 - a)
    assert torch.allclose(out, ref, atol=1e-7)
    assert torch.equal(pt_composite(canvas, fg, al, torch.zeros(2, 3)), canvas)  # all decisions 0


# ------------------------------------------------------------------ adapters


def test_adapters_roundtrip_and_mapping() -> None:
    p8 = random_params(10)
    p9 = pt_to_ours(p8)
    assert p9.shape == (10, 9) and torch.equal(p9[:, :8], p8) and (p9[:, 8] == 1).all()
    back, dec = ours_to_pt(p9)
    assert torch.equal(back, p8) and (dec == 1).all()
    p9[3, 8] = 0.0
    _, dec = ours_to_pt(p9)
    assert dec[3] == 0 and dec.sum() == 9
    assert pt_to_ours(p8, torch.arange(10) % 2 == 0)[:, 8].sum() == 5
    with pytest.raises(ValueError):
        ours_to_pt(p8)
    with pytest.raises(ValueError):
        pt_to_ours(p9)


def test_patch_to_canvas_formulas() -> None:
    patch = torch.tensor([[0.5, 0.25, 0.4, 0.2, 0.3, 1.0, 0.5, 0.0]])
    out = patch_to_canvas(patch, 1, 2, canvas_size=128)  # layer 2: 4 x 4 patches of 32 px
    expected = torch.tensor([(64 + 16) / 128, (32 + 8) / 128, 12.8 / 128, 6.4 / 128, 0.3])
    assert torch.allclose(out[0, :5], expected)
    assert torch.equal(out[0, 5:], patch[0, 5:])  # colour copied
    shifted = patch_to_canvas(patch, 0, 0, canvas_size=512, shifted=True)
    assert torch.allclose(shifted[0, :2], torch.tensor([(-16 + 16) / 512, (-16 + 8) / 512]))


# ---------------------------------------------------------- parity with the official renderer


@needs_official
@pytest.mark.parametrize("size", [(64, 64), (32, 32), (48, 80)])
def test_pt_infer_equals_official_param2stroke(size: tuple[int, int]) -> None:
    official = po.official_inference()
    meta = po.read_official_brushes("inference")
    ours_brushes = load_pt_brushes("inference", po.BRUSHES)
    assert torch.equal(meta, ours_brushes)  # same files, same conversion
    param = random_params(60, seed=3)
    param[::5, 2:4] = param[::5, 2:4].flip(1)  # make sure both brushes (h > w and h <= w) are used
    h, w = size
    fg_ref, al_ref = official.param2stroke(param.clone(), h, w, meta)
    fg, al = pt_param2stroke(param, h, w, ours_brushes, "pt_infer")
    diff = max(float((fg - fg_ref).abs().max()), float((al - al_ref).abs().max()))
    print(f"pt_infer {size}: max abs diff {diff}")
    assert diff <= TOLERANCE
    assert (param[:, 3] > param[:, 2]).any() and (param[:, 3] <= param[:, 2]).any()


@needs_official
@pytest.mark.parametrize("size", [32, 64])
def test_pt_train_equals_official_param2stroke(size: int) -> None:
    model = po.official_train_model()
    morph = po.official_train_morphology()
    param = random_params(60, seed=4)
    param12 = torch.rand(60, 12, generator=torch.Generator().manual_seed(5))
    param12[:, :8] = param
    param12[:, 8:11] = param[:, 5:8]  # R2, G2, B2 = R0, G0, B0 as the official generator does
    fg_ref, al_ref = model.param2stroke(param12, size, size)
    fg_ref, al_ref = morph.Dilation2d(m=1)(fg_ref), morph.Erosion2d(m=1)(al_ref)
    brushes = load_pt_brushes("train", po.BRUSHES)
    fg, al = pt_param2stroke(param, size, size, brushes, "pt_train")
    diff = max(float((fg - fg_ref).abs().max()), float((al - al_ref).abs().max()))
    print(f"pt_train {size}: max abs diff {diff}")
    assert diff <= TOLERANCE


@needs_official
def test_official_morphology_modules_equal_ours() -> None:
    morph = po.official_train_morphology()
    x = torch.rand(3, 3, 24, 24, generator=torch.Generator().manual_seed(6))
    assert torch.equal(
        morph.Dilation2d(m=1)(x.clone()), dilation(x)
    )  # the official module works in place
    assert torch.equal(morph.Erosion2d(m=1)(x.clone()), erosion(x))


@needs_official
def test_patch_adapter_matches_the_official_window_renderer_geometry() -> None:
    """One stroke in one patch: official 4-phase window rendering vs full-canvas rendering with
    patch_to_canvas.

    The official code renders in 64 px windows with a nearest-resized brush, ours on the whole
    canvas, so the
    textures differ slightly; the painted footprints must agree (same centroid, same extent, high
    overlap).
    """
    official = po.official_inference()
    meta = po.read_official_brushes("inference")
    brushes = load_pt_brushes("inference", po.BRUSHES)
    n, patch, canvas_px = 4, 32, 128  # layer 2: 4 x 4 patches
    patch_params = torch.zeros(1, n, n, 1, 8)
    patch_params[0, 1, 2, 0] = torch.tensor([0.45, 0.55, 0.5, 0.25, 0.2, 1.0, 1.0, 1.0])
    decision = torch.zeros(1, n, n, 1, dtype=torch.bool)
    decision[0, 1, 2, 0] = True
    remapped = patch_params.clone()
    remapped[..., :2] = remapped[..., :2] / 2 + 0.25  # what inference.main does before rendering
    remapped[..., 2:4] = remapped[..., 2:4] / 2
    canvas = torch.zeros(1, 3, canvas_px, canvas_px)
    ref = official.param2img_parallel(remapped, decision, meta, canvas)[0, 0]
    stroke = patch_to_canvas(patch_params[0, 1, 2], 1, 2, canvas_px, patch)  # (1, 8)
    mine = render_pt(stroke[None], canvas_px, canvas_px, brushes, "pt_infer")[0, 0]
    ys_r, xs_r = torch.nonzero(ref > 0.05, as_tuple=True)
    ys_m, xs_m = torch.nonzero(mine > 0.05, as_tuple=True)
    assert abs(float(xs_r.float().mean() - xs_m.float().mean())) < 1.0
    assert abs(float(ys_r.float().mean() - ys_m.float().mean())) < 1.0
    inter = ((ref > 0.05) & (mine > 0.05)).sum().item()
    union = ((ref > 0.05) | (mine > 0.05)).sum().item()
    assert inter / union > 0.9, inter / union


# ------------------------------------------------------------------ pipeline integration


@needs_brushes
def test_render_pt_mode_dispatch_black_canvas_and_default_unchanged() -> None:
    p9 = pt_to_ours(random_params(6, seed=8))[None]  # (1, 6, 9)
    cfg = RenderConfig(mode="pt", pt_brush_dir=str(po.BRUSHES))
    out = render(p9, DUMMY_BRUSH, 48, 48, cfg=cfg)
    brushes = load_pt_brushes("inference", po.BRUSHES)
    ref = render_pt(p9[..., :8], 48, 48, brushes, "pt_infer")
    assert torch.equal(out, ref) and out.shape == (1, 3, 48, 48)
    assert float(out[:, :, 0, 0].max()) == 0.0  # black canvas stays black far from strokes
    default = render(p9, DUMMY_BRUSH, 48, 48)  # default mode: white canvas, soft brush, opacity
    assert float(default[:, :, 0, 0].min()) > 0.5 and not torch.equal(default, out)
    p9[0, 2, 8] = 0.0  # decision 0: this stroke is not painted
    out2 = render(p9, DUMMY_BRUSH, 48, 48, cfg=cfg)
    keep = [0, 1, 3, 4, 5]
    ref2 = render_pt(p9[:, keep, :8], 48, 48, brushes, "pt_infer")
    assert torch.equal(out2, ref2)
    cfg_train = RenderConfig(mode="pt", pt_alpha="pt_train", pt_brush_dir=str(po.BRUSHES))
    assert not torch.equal(render(p9, DUMMY_BRUSH, 48, 48, cfg=cfg_train), out2)
    with pytest.raises(ValueError):
        render(p9, DUMMY_BRUSH, 8, 8, cfg=RenderConfig(mode="nope"))
    with pytest.raises(ValueError):
        render(p9, DUMMY_BRUSH, 8, 8, cfg=RenderConfig(mode="pt", pt_alpha="x"))


@needs_brushes
def test_pt_progressive_matches_prefix_renders_and_chunks() -> None:
    brushes = load_pt_brushes("inference", po.BRUSHES)
    p8 = random_params(20, seed=9).view(2, 10, 8)
    full = render_pt_progressive(p8, 40, 40, brushes, [0, 3, 10], chunk_size=4)
    assert torch.equal(full[:, 0], torch.zeros(2, 3, 40, 40))
    for i, k in enumerate([0, 3, 10]):
        ref = (
            render_pt(p8[:, :k], 40, 40, brushes, chunk_size=7) if k else torch.zeros(2, 3, 40, 40)
        )
        assert torch.allclose(full[:, i], ref, atol=1e-6)
    with pytest.raises(ValueError):
        render_pt_progressive(p8, 40, 40, brushes, [11])
