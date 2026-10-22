"""Tests for the differentiable batched stroke renderer."""

from __future__ import annotations

import math

import pytest
import torch

from brushstroke.renderer import (
    NUM_PARAMS,
    BrushBank,
    RenderConfig,
    canonicalize,
    clamp_params,
    composite_parallel,
    composite_sequential,
    make_brush,
    params_from_unconstrained,
    render,
    render_alphas,
    render_progressive,
    unconstrained_from_params,
)
from brushstroke.renderer.rasterize import default_brush_index
from brushstroke.utils.seed import seed_everything

ONES = torch.ones(1, 1, 256, 256)  # opaque box; edge pixels are ~0.5 (bilinear + zero padding)
FLAT = BrushBank.build("flat", 1, (64, 128), 0)
OIL = BrushBank.build("oil", 4, (64, 128), 0)


def random_params(
    b: int, n: int, seed: int = 0, size: tuple[float, float] = (0.05, 0.4)
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(b, n, NUM_PARAMS, generator=g)
    p[..., 2:4] = size[0] + (size[1] - size[0]) * p[..., 2:4]
    return p


def stroke(x: float, y: float, w: float, h: float, theta: float, a: float = 1.0) -> torch.Tensor:
    return torch.tensor([[[x, y, w, h, theta, 1.0, 1.0, 1.0, a]]])


def coverage(p: torch.Tensor, brush: torch.Tensor | BrushBank, size: int = 256) -> torch.Tensor:
    """Alpha of a white stroke over black: ``(H, W)``."""
    return render(p, brush, size, size, canvas=[0.0, 0.0, 0.0], cfg=RenderConfig())[0, 0]


def measure(img: torch.Tensor, thr: float = 0.5) -> tuple[float, float, float, float]:
    """Centroid (x, y) as fractions and extents (width, height) in pixels, from alpha > thr."""
    h, w = img.shape
    mask = img > thr
    ys, xs = torch.nonzero(mask, as_tuple=True)
    weights = img
    cx = float((weights * (torch.arange(w) + 0.5)[None, :]).sum() / weights.sum()) / w
    cy = float((weights * (torch.arange(h) + 0.5)[:, None]).sum() / weights.sum()) / h
    return cx, cy, float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)


# ------------------------------------------------------------------ shapes / dtypes


@pytest.mark.parametrize(
    ("b", "n", "chunk"), [(1, 1, 4), (2, 7, 3), (3, 8, 8), (1, 10, 64), (2, 5, 1)]
)
def test_shapes_and_dtypes(b: int, n: int, chunk: int) -> None:
    p = random_params(b, n)
    cfg = RenderConfig(chunk_size=chunk)
    out = render(p, OIL, 24, 40, cfg=cfg)
    assert out.shape == (b, 3, 24, 40)
    assert out.dtype == torch.float32 and out.device == p.device
    assert torch.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0 + 1e-6
    al = render_alphas(p, OIL, 24, 40)
    assert al.shape == (b, n, 1, 24, 40)
    out64 = render(p.double(), OIL, 24, 40, cfg=cfg)
    assert out64.dtype == torch.float64


def test_zero_strokes_returns_canvas() -> None:
    p = torch.zeros(2, 0, NUM_PARAMS)
    out = render(p, OIL, 16, 16, canvas=[0.2, 0.4, 0.6])
    assert out.shape == (2, 3, 16, 16)
    assert torch.allclose(out[:, 1], torch.tensor(0.4))


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        render(torch.zeros(1, 3, 8), OIL, 16, 16)
    with pytest.raises(ValueError):
        render(random_params(1, 3), OIL, 16, 16, cfg=RenderConfig(chunk_size=0))
    with pytest.raises(ValueError):
        render_progressive(random_params(1, 3), OIL, 16, 16, [4])
    with pytest.raises(ValueError):
        render(random_params(1, 3), OIL, 16, 16, canvas=torch.zeros(3, 8, 8))


def test_canvas_forms() -> None:
    p = random_params(2, 3)
    img = torch.rand(3, 16, 16)
    a = render(p, OIL, 16, 16, canvas=img)
    b = render(p, OIL, 16, 16, canvas=img[None].repeat(2, 1, 1, 1))
    assert torch.equal(a, b)


# ------------------------------------------------------------------ geometry


def test_flat_stroke_location_and_extent_with_exact_box() -> None:
    x, y, w, h = 0.3, 0.6, 0.4, 0.2
    cx, cy, ew, eh = measure(coverage(stroke(x, y, w, h, 0.0), ONES))
    assert abs(cx - x) * 256 < 1 and abs(cy - y) * 256 < 1
    assert abs(ew - w * 256) < 2 and abs(eh - h * 256) < 2


def test_flat_brush_kind_location_and_extent() -> None:
    # The `flat` mask is inset by 1.5 px of 128 x 64: visible box = (1 - 3/128) w, (1 - 3/64) h.
    x, y, w, h = 0.3, 0.6, 0.4, 0.2
    cx, cy, ew, eh = measure(coverage(stroke(x, y, w, h, 0.0), FLAT))
    assert abs(cx - x) * 256 < 1 and abs(cy - y) * 256 < 1
    assert abs(ew - w * 256 * (1 - 3 / 128)) < 2
    assert abs(eh - h * 256 * (1 - 3 / 64)) < 2


def test_rotation_swaps_extents_and_is_periodic() -> None:
    w, h = 0.4, 0.2
    _, _, ew0, eh0 = measure(coverage(stroke(0.5, 0.5, w, h, 0.0), ONES))
    cx, cy, ew1, eh1 = measure(coverage(stroke(0.5, 0.5, w, h, 0.5), ONES))
    assert abs(ew1 - eh0) < 2 and abs(eh1 - ew0) < 2
    assert abs(cx - 0.5) * 256 < 1 and abs(cy - 0.5) * 256 < 1
    a = coverage(stroke(0.4, 0.5, w, h, 0.0), ONES)
    b = coverage(stroke(0.4, 0.5, w, h, 1.0), ONES)
    assert torch.allclose(a, b, atol=1e-4)
    c = coverage(stroke(0.4, 0.5, w, h, 0.5), ONES)
    d = coverage(stroke(0.4, 0.5, w, h, 1.5), ONES)
    assert torch.allclose(c, d, atol=1e-4)


def test_rotation_45_degrees_bounding_box() -> None:
    w, h = 0.5, 0.2
    _, _, ew, eh = measure(
        coverage(stroke(0.5, 0.5, w, h, 0.25), ONES), thr=0.02
    )  # pointy corners: low threshold
    half = math.sqrt(0.5)
    expect_w = (w * 256 * half) + (h * 256 * half)  # bounding box of a 45-degree rectangle
    assert abs(ew - expect_w) < 3 and abs(eh - expect_w) < 3


def test_non_square_canvas_isotropic_rotation() -> None:
    p = stroke(0.5, 0.5, 0.25, 0.5, 0.0)  # 0.25*W = 128 px long, 0.5*H = 64 px thick on 512 x 128
    img = render(p, ONES, 128, 512, canvas=[0.0, 0.0, 0.0])[0, 0]
    _, _, ew, eh = measure(img)
    assert abs(ew - 128) < 2 and abs(eh - 64) < 2
    img90 = render(stroke(0.5, 0.5, 0.25, 0.5, 0.5), ONES, 128, 512, canvas=[0.0, 0.0, 0.0])[0, 0]
    _, _, ew90, eh90 = measure(img90)
    assert abs(ew90 - 64) < 2 and abs(eh90 - 128) < 2  # rotated 90 deg: 64 px wide, 128 tall (= H)


def test_canonicalization_is_footprint_preserving() -> None:
    a = coverage(stroke(0.5, 0.5, 0.1, 0.4, 0.13), FLAT)
    b = coverage(stroke(0.5, 0.5, 0.4, 0.1, 0.63), FLAT)
    assert torch.allclose(a, b, atol=1e-5)
    p = random_params(2, 20, seed=3)
    c = canonicalize(p)
    assert (c[..., 2] >= c[..., 3]).all()
    assert torch.equal(canonicalize(c), c)
    assert torch.allclose(c[..., 2] * c[..., 3], p[..., 2] * p[..., 3])


def test_canonicalization_on_non_square_canvas() -> None:
    # 0.1 * 512 = 51 px long x 0.8 * 128 = 102 px thick: swapped in *pixels*, footprint unchanged.
    p = stroke(0.5, 0.5, 0.1, 0.8, 0.2)
    c = canonicalize(p, 128, 512)
    assert (
        abs(float(c[0, 0, 2]) * 512 - 102.4) < 1e-3 and abs(float(c[0, 0, 3]) * 128 - 51.2) < 1e-3
    )
    img = render(p, ONES, 128, 512, canvas=[0.0, 0.0, 0.0])
    explicit = render(c, ONES, 128, 512, canvas=[0.0, 0.0, 0.0])
    assert torch.allclose(img, explicit, atol=1e-5)
    # Bounding box of the ORIGINAL footprint: 51.2 px along angle 0.2 pi, 102.4 px across.
    t0 = 0.2 * math.pi
    exp_w = 51.2 * abs(math.cos(t0)) + 102.4 * abs(math.sin(t0))
    exp_h = 51.2 * abs(math.sin(t0)) + 102.4 * abs(math.cos(t0))
    cx, cy, ew, eh = measure(img[0, 0], thr=0.02)
    assert abs(cx - 0.5) * 512 < 1 and abs(cy - 0.5) * 128 < 1
    assert abs(ew - exp_w) < 3 and abs(eh - min(exp_h, 128)) < 3


def test_texture_follows_long_side_for_tall_strokes() -> None:
    """Oil bristles run along the stroke's long side, also when h > w (canonicalization)."""
    wide = coverage(stroke(0.5, 0.5, 0.6, 0.25, 0.0), OIL[:1])
    tall = coverage(stroke(0.5, 0.5, 0.25, 0.6, 0.0), OIL[:1])

    def anisotropy(img: torch.Tensor) -> tuple[float, float]:
        core = img[96:160, 96:160]
        dy = (core[1:] - core[:-1]).abs().mean()
        dx = (core[:, 1:] - core[:, :-1]).abs().mean()
        return float(dx), float(dy)

    wdx, wdy = anisotropy(wide)
    tdx, tdy = anisotropy(tall)
    assert wdy > 1.5 * wdx  # horizontal bristles: values change quickly across rows
    assert tdx > 1.5 * tdy  # vertical bristles: values change quickly across columns


# ------------------------------------------------------------------ colours / opacity


def test_zero_opacity_leaves_canvas_exactly() -> None:
    p = random_params(2, 9)
    p[..., 8] = 0.0
    canvas = torch.rand(2, 3, 32, 32)
    out = render(p, OIL, 32, 32, canvas=canvas, cfg=RenderConfig(chunk_size=4))
    assert torch.equal(out, canvas)


def test_opaque_flat_interior_equals_colour() -> None:
    p = torch.tensor([[[0.5, 0.5, 0.5, 0.5, 0.1, 0.2, 0.5, 0.9, 1.0]]])
    out = render(p, FLAT, 64, 64, canvas=[0.7, 0.7, 0.7])[0]
    center = out[:, 28:36, 28:36]
    expect = torch.tensor([0.2, 0.5, 0.9]).view(3, 1, 1).expand_as(center)
    assert torch.allclose(center, expect, atol=1e-5)


def test_later_strokes_cover_earlier_ones() -> None:
    red = torch.tensor([0.5, 0.5, 0.6, 0.6, 0.0, 1.0, 0.0, 0.0, 1.0])
    blue = torch.tensor([0.5, 0.5, 0.6, 0.6, 0.0, 0.0, 0.0, 1.0, 1.0])
    out = render(torch.stack([red, blue])[None], ONES, 32, 32)[0]
    assert torch.allclose(out[:, 16, 16], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)
    out = render(torch.stack([blue, red])[None], ONES, 32, 32)[0]
    assert torch.allclose(out[:, 16, 16], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


# ------------------------------------------------------------------ compositing


@pytest.mark.parametrize("chunk", [1, 3, 8, 17, 64])
def test_parallel_chunked_equals_sequential(chunk: int) -> None:
    p = random_params(2, 17, seed=5)
    canvas = torch.rand(2, 3, 40, 48)
    idx = default_brush_index(2, 17, len(OIL), 0)
    alphas = render_alphas(p, OIL, 40, 48, brush_index=idx)
    ref = composite_sequential(canvas, alphas, p[..., 5:8])
    out = render(p, OIL, 40, 48, canvas=canvas, cfg=RenderConfig(chunk_size=chunk))
    assert torch.allclose(out, ref, atol=1e-5)


def test_composite_parallel_with_saturated_alphas() -> None:
    g = torch.Generator().manual_seed(1)
    alphas = torch.rand(2, 6, 1, 8, 8, generator=g)
    alphas[:, 2] = 1.0  # exact zeros in (1 - alpha): cumprod backward must stay correct
    alphas[:, 4, :, :4] = 0.0
    colors = torch.rand(2, 6, 3, generator=g)
    canvas = torch.rand(2, 3, 8, 8, generator=g)
    ref = composite_sequential(canvas, alphas, colors)
    assert torch.allclose(composite_parallel(canvas, alphas, colors), ref, atol=1e-6)
    a1 = alphas.clone().requires_grad_()
    a2 = alphas.clone().requires_grad_()
    composite_sequential(canvas, a1, colors).square().sum().backward()
    composite_parallel(canvas, a2, colors).square().sum().backward()
    assert a1.grad is not None and a2.grad is not None
    assert torch.allclose(a1.grad, a2.grad, atol=1e-5)


def test_progressive_equals_rendering_prefixes() -> None:
    p = random_params(2, 23, seed=7)
    canvas = torch.rand(3, 32, 32)
    cps = [7, 0, 23, 1, 12, 12]
    cfg = RenderConfig(chunk_size=5)
    prog = render_progressive(p, OIL, 32, 32, cps, canvas=canvas, cfg=cfg)
    assert prog.shape == (2, len(cps), 3, 32, 32)
    for i, k in enumerate(cps):
        ref = render(p[:, :k], OIL, 32, 32, canvas=canvas, cfg=cfg)
        assert torch.allclose(prog[:, i], ref, atol=1e-6), k


def test_default_brush_index_prefix_stable() -> None:
    a = default_brush_index(3, 10, 4, 0)
    b = default_brush_index(3, 25, 4, 0)
    assert torch.equal(a, b[:, :10])
    assert not torch.equal(default_brush_index(3, 10, 4, 1), a)


def test_checkpointing_matches_plain() -> None:
    p = random_params(1, 20, seed=2).requires_grad_()
    p2 = p.detach().clone().requires_grad_()
    o1 = render(p, OIL, 32, 32, cfg=RenderConfig(chunk_size=6))
    o2 = render(p2, OIL, 32, 32, cfg=RenderConfig(chunk_size=6, use_checkpoint=True))
    assert torch.allclose(o1, o2, atol=1e-6)
    o1.square().sum().backward()
    o2.square().sum().backward()
    assert p.grad is not None and p2.grad is not None
    assert torch.allclose(p.grad, p2.grad, atol=1e-4, rtol=1e-4)


def test_bfloat16_compositing_close_to_fp32() -> None:
    p = random_params(1, 24, seed=4)
    ref = render(p, OIL, 48, 48)
    out = render(p, OIL, 48, 48, cfg=RenderConfig(dtype="bfloat16"))
    assert out.dtype == torch.float32
    assert (out - ref).abs().max() < 0.05


# ------------------------------------------------------------------ gradients


def _smooth_brush() -> torch.Tensor:
    ax = torch.linspace(-1, 1, 12, dtype=torch.float64)
    return torch.exp(-3.0 * (ax[None, :] ** 2 + ax[:, None] ** 2))[None, None]


def test_gradcheck_float64_all_params() -> None:
    brush = _smooth_brush()
    canvas = torch.rand(
        1, 3, 16, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    p = torch.tensor(
        [
            [0.45, 0.52, 0.62, 0.41, 0.17, 0.2, 0.5, 0.8, 0.9],
            [0.60, 0.35, 0.33, 0.55, 0.62, 0.9, 0.1, 0.3, 0.7],
            [0.30, 0.66, 0.58, 0.37, 0.83, 0.4, 0.6, 0.2, 0.8],
        ],
        dtype=torch.float64,
    )[None].requires_grad_()
    cfg = RenderConfig(chunk_size=2)
    assert torch.autograd.gradcheck(
        lambda q: render(q, brush, 16, 16, canvas=canvas, cfg=cfg),
        (p,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-4,
    )


def test_gradients_finite_and_nonzero_cpu() -> None:
    p = random_params(2, 30, seed=11).requires_grad_()
    render(p, OIL, 48, 48, cfg=RenderConfig(chunk_size=8)).square().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert (p.grad.abs().sum(dim=(0, 1)) > 0).all()


@pytest.mark.gpu
def test_gradients_finite_and_nonzero_every_param_gpu() -> None:
    p = random_params(2, 64, seed=12).cuda().requires_grad_()
    out = render(p, OIL.to("cuda"), 96, 96, cfg=RenderConfig(chunk_size=16))
    out.square().sum().backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    per_param = p.grad.abs().sum(dim=(0, 1))
    assert (per_param > 0).all(), per_param


# ------------------------------------------------------------------ determinism / devices


def _run(
    p: torch.Tensor, brush: BrushBank, cfg: RenderConfig | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    q = p.clone().requires_grad_()
    out = render(q, brush, 48, 48, cfg=cfg or RenderConfig(chunk_size=7))
    out.square().sum().backward()
    assert q.grad is not None
    return out.detach(), q.grad.detach()


def test_determinism_cpu_bitwise() -> None:
    p = random_params(2, 25, seed=9)
    o1, g1 = _run(p, OIL)
    o2, g2 = _run(p, OIL)
    assert torch.equal(o1, o2) and torch.equal(g1, g2)


def test_gather_sampler_matches_grid_sample() -> None:
    p = random_params(
        2, 20, seed=21, size=(0.1, 1.2)
    )  # includes strokes sticking out of the canvas
    p[..., :2] = p[..., :2] * 1.4 - 0.2
    cfg_g = RenderConfig(chunk_size=6, sampler="gather")
    cfg_s = RenderConfig(chunk_size=6)
    q1 = p.clone().requires_grad_()
    q2 = p.clone().requires_grad_()
    o1 = render(q1, OIL, 40, 56, cfg=cfg_g)
    o2 = render(q2, OIL, 40, 56, cfg=cfg_s)
    assert torch.allclose(o1, o2, atol=1e-5)
    o1.square().sum().backward()
    o2.square().sum().backward()
    assert q1.grad is not None and q2.grad is not None
    assert torch.allclose(q1.grad, q2.grad, atol=1e-3, rtol=1e-4)
    with pytest.raises(ValueError):
        render(p, OIL, 8, 8, cfg=RenderConfig(sampler="nearest"))


@pytest.mark.gpu
def test_determinism_gpu_deterministic_flags_with_gather_sampler() -> None:
    seed_everything(0, deterministic=True)
    try:
        p = random_params(2, 40, seed=9).cuda()
        brush = OIL.to("cuda")
        runs = [_run(p, brush, RenderConfig(chunk_size=7, sampler="gather")) for _ in range(2)]
        with pytest.raises(RuntimeError, match="deterministic"):
            _run(p, brush, RenderConfig(chunk_size=7))  # grid_sample backward refuses the flag
    finally:
        seed_everything(0, deterministic=False)
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])


@pytest.mark.gpu
def test_grid_sample_gpu_reproducible_in_practice() -> None:
    """Without the flag, grid_sample was bitwise reproducible here (documented, not guaranteed)."""
    p = random_params(2, 40, seed=9).cuda()
    brush = OIL.to("cuda")
    (o1, g1), (o2, g2) = _run(p, brush), _run(p, brush)
    assert torch.equal(o1, o2) and torch.equal(g1, g2)
    (o3, g3) = _run(p, brush, RenderConfig(chunk_size=7, sampler="gather"))
    assert torch.allclose(o1, o3, atol=1e-5)


@pytest.mark.gpu
def test_cpu_gpu_agree() -> None:
    p = random_params(2, 30, seed=13)
    cpu = render(p, OIL, 64, 64)
    gpu = render(p.cuda(), OIL.to("cuda"), 64, 64).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-4)
    q1 = p.clone().requires_grad_()
    q2 = p.clone().cuda().requires_grad_()
    render(q1, OIL, 64, 64).sum().backward()
    render(q2, OIL.to("cuda"), 64, 64).sum().backward()
    assert q1.grad is not None and q2.grad is not None
    assert torch.allclose(q1.grad, q2.grad.cpu(), atol=1e-2, rtol=1e-3)


# ------------------------------------------------------------------ parameter helpers


def test_clamp_and_reparameterization() -> None:
    p = torch.randn(2, 5, NUM_PARAMS) * 3
    c = clamp_params(p, min_size=0.01)
    assert c.min() >= 0.0 and c.max() <= 1.0
    assert c[..., 2:4].min() >= 0.01
    raw = torch.randn(2, 5, NUM_PARAMS, dtype=torch.float64, requires_grad=True)
    q = params_from_unconstrained(raw, min_size=0.01)
    assert q.min() >= 0.0 and q.max() <= 1.0 and q[..., 2:4].min() >= 0.01
    q.sum().backward()
    assert raw.grad is not None and (raw.grad > 0).all()
    back = unconstrained_from_params(q.detach(), min_size=0.01)
    assert torch.allclose(back, raw.detach(), atol=1e-3)
    with pytest.raises(ValueError):
        clamp_params(torch.zeros(3, 9))


def test_render_alphas_uses_brush_index() -> None:
    p = random_params(1, 4, seed=1)
    idx0 = torch.zeros(1, 4, dtype=torch.long)
    idx1 = torch.ones(1, 4, dtype=torch.long)
    a0 = render_alphas(p, OIL, 32, 32, brush_index=idx0)
    a1 = render_alphas(p, OIL, 32, 32, brush_index=idx1)
    assert not torch.equal(a0, a1)
    single = render_alphas(p, make_brush("oil", (64, 128), 0)[None], 32, 32)
    assert torch.equal(single, a0)
