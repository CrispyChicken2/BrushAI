"""Tests for the shared image metrics."""

from __future__ import annotations

import pytest
import torch

from brushstroke.eval.metrics import image_metrics, ssim


def naive_ssim(x: torch.Tensor, y: torch.Tensor, win: int = 11, sigma: float = 1.5) -> float:
    """Direct per-window implementation (loops) to validate the convolutional one."""
    ax = torch.arange(win, dtype=torch.float64) - (win - 1) / 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    w = (g[:, None] * g[None, :]) / (g.sum() ** 2)
    x, y = x.double(), y.double()
    c, h, wd = x.shape
    vals = []
    for ch in range(c):
        for i in range(h - win + 1):
            for j in range(wd - win + 1):
                px, py = x[ch, i : i + win, j : j + win], y[ch, i : i + win, j : j + win]
                mx, my = (w * px).sum(), (w * py).sum()
                vx, vy = (w * px * px).sum() - mx**2, (w * py * py).sum() - my**2
                vxy = (w * px * py).sum() - mx * my
                vals.append(
                    ((2 * mx * my + 1e-4) * (2 * vxy + 9e-4))
                    / ((mx**2 + my**2 + 1e-4) * (vx + vy + 9e-4))
                )
    return float(torch.stack(vals).mean())


def test_ssim_identity_symmetry_and_monotonicity() -> None:
    g = torch.Generator().manual_seed(0)
    a = torch.rand(3, 32, 32, generator=g)
    assert float(ssim(a, a)) == pytest.approx(1.0, abs=1e-6)
    noise = torch.randn(3, 32, 32, generator=g)
    values = [float(ssim(a, (a + s * noise).clamp(0, 1))) for s in (0.02, 0.1, 0.4)]
    assert values[0] > values[1] > values[2] and values[0] < 1.0
    b = torch.rand(3, 32, 32, generator=g)
    assert float(ssim(a, b)) == pytest.approx(float(ssim(b, a)), abs=1e-6)
    with pytest.raises(ValueError):
        ssim(a, b[:, :16])


def test_ssim_matches_naive_reference() -> None:
    g = torch.Generator().manual_seed(1)
    a = torch.rand(3, 20, 24, generator=g)
    b = (a * 0.8 + 0.1 * torch.rand(3, 20, 24, generator=g)).clamp(0, 1)
    assert float(ssim(a, b)) == pytest.approx(naive_ssim(a, b), abs=1e-5)


@pytest.mark.slow
def test_image_metrics_keys_and_identity() -> None:
    g = torch.Generator().manual_seed(2)
    a = torch.rand(3, 64, 64, generator=g)
    m = image_metrics(a, a)
    assert set(m) == {"l1", "psnr", "ssim", "lpips_vgg", "lpips_alex"}
    assert (
        m["l1"] == 0.0
        and m["ssim"] == pytest.approx(1.0, abs=1e-6)
        and m["lpips_vgg"] == pytest.approx(0.0, abs=1e-6)
    )
    n = image_metrics(a, torch.rand(3, 64, 64, generator=g))
    assert n["psnr"] < 20 and n["ssim"] < 0.5 and n["lpips_alex"] > 0.05
