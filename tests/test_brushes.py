"""Tests for procedural brush masks, the brush bank and the canvas texture."""

from __future__ import annotations

import math

import pytest
import torch

from brushstroke.renderer.brushes import BRUSH_KINDS, BrushBank, BrushConfig, make_brush
from brushstroke.renderer.canvas import apply_canvas, make_canvas_texture

SIZE = (64, 128)
SEEDS = range(6)


@pytest.mark.parametrize("kind", BRUSH_KINDS)
def test_shape_dtype_range(kind: str) -> None:
    m = make_brush(kind, SIZE, 0)
    assert m.shape == (1, *SIZE)
    assert m.dtype == torch.float32
    assert torch.isfinite(m).all()
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert m.max() > 0.5  # not an empty mask


@pytest.mark.parametrize("size", [(8, 8), (33, 71), (64, 256)])
@pytest.mark.parametrize("kind", BRUSH_KINDS)
def test_odd_sizes(kind: str, size: tuple[int, int]) -> None:
    m = make_brush(kind, size, 1)
    assert m.shape == (1, *size)
    assert torch.isfinite(m).all()


def test_invalid_args() -> None:
    with pytest.raises(ValueError):
        make_brush("watercolor", SIZE, 0)
    with pytest.raises(ValueError):
        make_brush("oil", (4, 4), 0)


def test_flat_interior_and_exterior() -> None:
    m = make_brush("flat", SIZE, 0)[0]
    h, w = SIZE
    assert torch.all(m[3 : h - 3, 3 : w - 3] == 1.0)
    for border in (m[0], m[-1], m[:, 0], m[:, -1]):
        assert torch.all(border == 0.0)
    assert torch.all((m == 0.0) | (m == 1.0) | ((m > 0) & (m < 1)))  # sanity
    assert ((m > 0) & (m < 1)).any()  # antialiased border exists


@pytest.mark.parametrize("kind", BRUSH_KINDS)
def test_determinism_and_seed_sensitivity(kind: str) -> None:
    a, b = make_brush(kind, SIZE, 7), make_brush(kind, SIZE, 7)
    assert torch.equal(a, b)
    c = make_brush(kind, SIZE, 8)
    if kind == "flat":
        assert torch.equal(a, c)  # seed-independent by design
    else:
        assert not torch.equal(a, c)


def _principal_angle_deg(mask: torch.Tensor) -> float:
    """Angle in degrees (0..90) between x and the principal axis of the alpha covariance."""
    m = mask[0].double()
    h, w = m.shape
    ys = (torch.arange(h, dtype=torch.float64) + 0.5)[:, None].expand(h, w)
    xs = (torch.arange(w, dtype=torch.float64) + 0.5)[None, :].expand(h, w)
    total = m.sum()
    mx, my = (m * xs).sum() / total, (m * ys).sum() / total
    cxx = (m * (xs - mx) ** 2).sum() / total
    cyy = (m * (ys - my) ** 2).sum() / total
    cxy = (m * (xs - mx) * (ys - my)).sum() / total
    theta = 0.5 * math.atan2(2 * float(cxy), float(cxx - cyy))
    return abs(math.degrees(theta))


@pytest.mark.parametrize("seed", SEEDS)
def test_oil_principal_axis_is_x(seed: int) -> None:
    m = make_brush("oil", SIZE, seed)
    assert _principal_angle_deg(m) < 5.0
    # and the mask is elongated along x (variance ratio), not isotropic
    mm = m[0]
    h, w = mm.shape
    xs = (torch.arange(w) + 0.5)[None, :].expand(h, w)
    ys = (torch.arange(h) + 0.5)[:, None].expand(h, w)
    tot = mm.sum()
    vx = (mm * (xs - (mm * xs).sum() / tot) ** 2).sum() / tot
    vy = (mm * (ys - (mm * ys).sum() / tot) ** 2).sum() / tot
    assert vx > 2.0 * vy


@pytest.mark.parametrize("seed", SEEDS)
def test_oil_is_textured_and_tapered(seed: int) -> None:
    m = make_brush("oil", SIZE, seed)[0]
    h, w = m.shape
    support = m > 0.05
    inner = m[support]
    assert inner.std() > 0.1  # bristle streaks, not a flat blob
    assert 0.35 < inner.mean() < 0.95
    # Streaks run along x: neighbouring rows differ more than neighbouring columns.
    core = m[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    assert (core[1:] - core[:-1]).abs().mean() > 2.0 * (core[:, 1:] - core[:, :-1]).abs().mean()
    # Ends taper: outer 10% of the length is lighter than the central 50%.
    k = w // 10
    ends = torch.cat([m[:, :k], m[:, w - k :]], dim=1).mean()
    center = m[:, w // 4 : 3 * w // 4].mean()
    assert ends < 0.5 * center


@pytest.mark.parametrize("seed", SEEDS)
def test_round_is_centered_and_soft(seed: int) -> None:
    m = make_brush("round", SIZE, seed)[0]
    h, w = m.shape
    assert m[h // 2 - 4 : h // 2 + 4, w // 2 - 8 : w // 2 + 8].mean() > 0.7
    assert max(m[0].max(), m[-1].max(), m[:, 0].max(), m[:, -1].max()) < 0.02
    assert _principal_angle_deg(m.unsqueeze(0)) < 5.0


def test_bank_shape_cache_and_device() -> None:
    cfg = BrushConfig(kind="oil", bank_size=4, mask_height=32, mask_width=64, seed=3)
    bank = BrushBank.from_config(cfg)
    assert len(bank) == 4
    assert bank.masks.shape == (4, 1, 32, 64)
    assert bank.size == (32, 64)
    assert torch.equal(bank[2], make_brush("oil", (32, 64), 5))  # variant i uses seed + i
    assert not torch.equal(bank[0], bank[1])
    again = BrushBank.from_config(cfg)
    assert torch.equal(bank.masks, again.masks)
    bank.masks.zero_()  # mutating a returned bank must not poison the cache
    assert BrushBank.from_config(cfg).masks.max() > 0
    moved = again.to("cpu", torch.float64)
    assert moved.masks.dtype == torch.float64


def test_bank_default_generation_is_fast() -> None:
    import time

    t0 = time.perf_counter()
    for kind in BRUSH_KINDS:
        BrushBank.build(kind, BrushConfig().bank_size, (64, 128), 0)
    assert time.perf_counter() - t0 < 1.0


def test_bank_validation() -> None:
    with pytest.raises(ValueError):
        BrushBank.build("oil", 0, SIZE, 0)
    with pytest.raises(ValueError):
        BrushBank(torch.zeros(3, 8, 8), "oil", 0)


@pytest.mark.gpu
def test_bank_to_cuda() -> None:
    bank = BrushBank.build("oil", 2, SIZE, 0).to("cuda")
    assert bank.masks.is_cuda


# ---------------------------------------------------------------- canvas


def test_canvas_shape_range_determinism() -> None:
    t = make_canvas_texture(96, 160, seed=1)
    assert t.shape == (1, 96, 160)
    assert t.dtype == torch.float32
    assert torch.isfinite(t).all()
    assert t.min() >= 0.0 and t.max() <= 1.0
    assert t.std() > 0.05
    assert torch.equal(t, make_canvas_texture(96, 160, seed=1))
    assert not torch.equal(t, make_canvas_texture(96, 160, seed=2))


def test_canvas_strength_zero_is_flat() -> None:
    assert torch.allclose(make_canvas_texture(32, 32, 0, strength=0.0), torch.tensor(0.5))
    weak = make_canvas_texture(64, 64, 0, strength=0.2)
    full = make_canvas_texture(64, 64, 0, strength=1.0)
    assert weak.std() < full.std()


def _radial_power(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Power spectrum of ``t`` (DC removed) and the radial frequency (cycles/pixel) of each bin."""
    spec = torch.fft.rfft2(t - t.mean()).abs() ** 2
    fy = torch.fft.fftfreq(t.shape[0])[:, None].abs()
    fx = torch.fft.rfftfreq(t.shape[1])[None, :]
    return spec, torch.sqrt(fx**2 + fy**2)


@pytest.mark.parametrize("low_freq_amount", [0.0, 0.2])
def test_canvas_has_weave_frequency_peak(low_freq_amount: float) -> None:
    period = 6
    t = make_canvas_texture(
        192, 192, 0, weave_period=float(period), low_freq_amount=low_freq_amount
    )
    spec, radius = _radial_power(t[0])
    # Plain weave: energy at the cell alternation (1 / (2 period)) and thread period (1 / period).
    weave = ((radius > 0.4 / period) & (radius < 0.6 / period)) | (
        (radius > 0.9 / period) & (radius < 1.1 / period)
    )
    background = (radius > 0.2 / period) & (radius < 0.35 / period)  # between low-freq and weave
    assert spec[weave].mean() > 10.0 * spec[background].mean()
    if low_freq_amount == 0.0:
        spec[0, 0] = 0
        py, px = divmod(int(spec.argmax()), spec.shape[1])
        pr = float(radius[py, px])
        assert min(abs(pr - 1 / period), abs(pr - 0.5 / period)) < 0.1 / period


def test_canvas_validation() -> None:
    with pytest.raises(ValueError):
        make_canvas_texture(4, 4, 0)
    with pytest.raises(ValueError):
        make_canvas_texture(32, 32, 0, strength=1.5)
    with pytest.raises(ValueError):
        make_canvas_texture(32, 32, 0, weave_period=1.0)


def test_apply_canvas() -> None:
    tex = make_canvas_texture(32, 32, 0)
    img = torch.full((3, 32, 32), 0.5)
    out = apply_canvas(img, tex, 0.1)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.equal(apply_canvas(img, tex, 0.0), img)
    assert torch.allclose(apply_canvas(img, torch.full((1, 32, 32), 0.5), 0.3), img)
    batch = apply_canvas(img[None].repeat(2, 1, 1, 1), tex, 0.1)
    assert torch.equal(batch[1], out)
    with pytest.raises(ValueError):
        apply_canvas(img, tex[:, :16], 0.1)
