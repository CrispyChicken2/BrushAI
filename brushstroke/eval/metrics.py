"""Final-image metrics shared by the demos and experiments: L1, PSNR, SSIM, LPIPS-VGG / LPIPS-Alex.

All images are ``(3, H, W)`` float tensors in [0, 1] on the same device. LPIPS-Alex is the held-out
metric (never used in a training loss, see ``docs/experiments/optim_sweep.md``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from brushstroke.baselines.optim.losses import get_lpips, lpips_distance


def _gaussian_window(
    size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(ax**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean structural similarity (Wang et al., IEEE TIP 2004) over the colour channels.

    Gaussian window (11 x 11, sigma 1.5), no padding (valid region), ``C1 = 0.01^2``,
    ``C2 = 0.03^2`` (dynamic range 1). Returns a scalar in (-1, 1]; 1 for identical images.
    """
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(
            f"expected two images of shape (3, H, W), got {tuple(x.shape)} and {tuple(y.shape)}"
        )
    w = _gaussian_window(window, sigma, x.device, x.dtype)[None, None].expand(x.shape[0], 1, -1, -1)

    def blur(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t[None], w, groups=t.shape[0])[0]

    mx, my = blur(x), blur(y)
    vxx, vyy, vxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vxx + vyy + c2))
    return s.mean()


def image_metrics(final: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """L1, PSNR (dB), SSIM, LPIPS-VGG and LPIPS-Alex between ``final`` and ``target``."""
    with torch.no_grad():
        mse = float((final - target).square().mean())
        return {
            "l1": float((final - target).abs().mean()),
            "psnr": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "ssim": float(ssim(final, target)),
            "lpips_vgg": float(lpips_distance(final[None], target[None], "vgg")),
            "lpips_alex": float(lpips_distance(final[None], target[None], "alex")),
        }


# ----------------------------------------------------------------------------- the evaluation protocol (T-026)


def get_lpips_spatial(device: torch.device | str) -> torch.nn.Module:
    """LPIPS-Alex with ``spatial=True``: a per-pixel distance map ``(B, 1, H, W)`` (docs/design/evaluation.md 4.1)."""
    key = f"alex-spatial:{device}"
    if key not in _SPATIAL_CACHE:
        import lpips

        model = lpips.LPIPS(net="alex", spatial=True, verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _SPATIAL_CACHE[key] = model
    return _SPATIAL_CACHE[key]


_SPATIAL_CACHE: dict[str, torch.nn.Module] = {}


@torch.no_grad()
def lpips_alex_map(canvases: torch.Tensor, target: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Spatial LPIPS-Alex map of every canvas ``(K, 3, H, W)`` against ``target`` ``(3, H, W)``: ``(K, H, W)``."""
    model = get_lpips_spatial(canvases.device)
    out = []
    for i in range(0, canvases.shape[0], chunk):
        part = canvases[i : i + chunk]
        out.append(model(part * 2.0 - 1.0, target[None].expand_as(part) * 2.0 - 1.0)[:, 0])
    return torch.cat(out)


def saliency_weights(saliency: torch.Tensor, lam: float) -> torch.Tensor:
    """``w(x) = (1 - lambda) s(x) / sum(s) + lambda / N`` (section 4.1), summing to 1. A map with no mass gives uniform weights."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lam}")
    n = saliency.numel()
    total = saliency.sum()
    if float(total) <= 0.0:
        return torch.full_like(saliency, 1.0 / n)
    return (1.0 - lam) * saliency / total + lam / n


def sw_lpips(maps: torch.Tensor, saliency: torch.Tensor, lam: float) -> torch.Tensor:
    """Saliency-weighted LPIPS: ``sum_x w(x) m(x)`` per canvas, for LPIPS maps ``(K, H, W)`` and a target map ``(H, W)``."""
    w = saliency_weights(saliency.to(maps), lam)
    return (maps * w).sum(dim=(-2, -1))


@torch.no_grad()
def lpips_vgg_chunked(canvases: torch.Tensor, target: torch.Tensor, chunk: int = 2) -> torch.Tensor:
    """LPIPS-VGG of every canvas, at most ``chunk`` canvases per call (the metric peaks near 10 GB for 15 canvases, 1.3 GB for 2)."""
    out = []
    for i in range(0, canvases.shape[0], chunk):
        part = canvases[i : i + chunk]
        model = get_lpips(part.device, "vgg")
        out.append(model(part * 2.0 - 1.0, target[None].expand_as(part) * 2.0 - 1.0).flatten())
    return torch.cat(out)


def area_downscale(x: torch.Tensor, size: int) -> torch.Tensor:
    """Area (box) downscale of ``(..., 3, H, W)`` to ``size`` x ``size`` (Learning to Paint's native domain, section 7.4)."""
    lead = x.shape[:-3]
    flat = x.reshape(-1, *x.shape[-3:])
    return F.interpolate(flat, size=(size, size), mode="area").reshape(*lead, 3, size, size)


@torch.no_grad()
def canvas_metrics(
    canvases: torch.Tensor,
    target: torch.Tensor,
    saliency: dict[str, torch.Tensor] | None = None,
    lambdas: tuple[float, ...] = (0.0, 0.1, 0.3),
    vgg_chunk: int = 2,
    low_res: int = 128,
) -> dict[str, list[float]]:
    """Every protocol metric of every canvas ``(K, 3, H, W)`` against ``target`` ``(3, H, W)``, in [0, 1] on one device.

    Keys: ``l1``, ``psnr``, ``ssim``, ``lpips_alex`` (the headline), ``lpips_vgg``, ``lpips_alex_<low_res>`` and ``l1_<low_res>``
    (area-downscaled canvas and target), and for every saliency map ``name`` and ``lambda``: ``sw_<name>_<lambda>``.
    """
    k = canvases.shape[0]
    mse = (canvases - target[None]).square().flatten(1).mean(1)
    maps = lpips_alex_map(canvases, target)
    small_c, small_t = area_downscale(canvases, low_res), area_downscale(target, low_res)
    alex_model = get_lpips(canvases.device, "alex")
    small_alex = torch.cat(
        [
            alex_model(
                small_c[i : i + 8] * 2.0 - 1.0,
                small_t[None].expand_as(small_c[i : i + 8]) * 2.0 - 1.0,
            ).flatten()
            for i in range(0, k, 8)
        ]
    )
    alex = torch.cat(
        [
            alex_model(
                canvases[i : i + 8] * 2.0 - 1.0,
                target[None].expand_as(canvases[i : i + 8]) * 2.0 - 1.0,
            ).flatten()
            for i in range(0, k, 8)
        ]
    )
    out: dict[str, list[float]] = {
        "l1": (canvases - target[None]).abs().flatten(1).mean(1).tolist(),
        "psnr": (10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))).tolist(),
        "ssim": [float(ssim(c, target)) for c in canvases],
        "lpips_alex": alex.tolist(),
        "lpips_vgg": lpips_vgg_chunked(canvases, target, vgg_chunk).tolist(),
        f"lpips_alex_{low_res}": small_alex.tolist(),
        f"l1_{low_res}": (small_c - small_t[None]).abs().flatten(1).mean(1).tolist(),
    }
    for name, s in (saliency or {}).items():
        for lam in lambdas:
            out[f"sw_{name}_{lam:g}"] = sw_lpips(maps, s, lam).tolist()
    return out
