"""Procedural brush alpha masks.

A brush mask is a ``(1, h, w)`` float32 tensor with values in ``[0, 1]`` (1 = fully opaque paint).
Convention (shared with the renderer, T-003): the **main stroke axis is the local x axis** (the
width direction ``w``); the local y axis (``h``) is the brush thickness. A stroke is rendered by
warping the mask with an affine transform (scale, rotation, translation), as in Paint Transformer
(Liu et al., ICCV 2021), which warps a single brush texture image. Stylized Neural Painting
(Zou et al., CVPR 2021) likewise renders textured brush images (oil-paint / marker-pen /
watercolor), whereas Learning to Paint (Huang et al., ICCV 2019) uses untextured quadratic
Bezier strokes.

Everything is generated procedurally on CPU from a seed (no third-party brush images, hence no
licensing issue) and is bitwise deterministic for a given ``(kind, size, seed)``.

Kinds:
    ``flat``  antialiased rectangle (ablation baseline; ignores the seed).
    ``oil``   bristle streaks along x, ragged soft lateral edges, tapered and dry (broken) ends.
    ``round`` soft elliptical dab with a mild isotropic texture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

BRUSH_KINDS = ("flat", "oil", "round")


@dataclass
class BrushConfig:
    """``brush:`` config section."""

    kind: str = "oil"  # one of BRUSH_KINDS
    bank_size: int = 8  # number of variants (seeds ``seed .. seed + bank_size - 1``)
    mask_height: int = 64  # pixels along local y (thickness)
    mask_width: int = 128  # pixels along local x (main axis)
    seed: int = 0


def gaussian_blur_1d(x: torch.Tensor, sigma: float, dim: int) -> torch.Tensor:
    """Blur ``x`` of shape ``(H, W)`` with a Gaussian (std ``sigma`` px) along ``dim`` (0 or 1).

    Reflect padding; a ``sigma`` <= 0 returns ``x`` unchanged.
    """
    if sigma <= 0:
        return x
    radius = max(1, math.ceil(3.0 * sigma))
    taps = torch.arange(-radius, radius + 1, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (taps / sigma) ** 2)
    kernel = kernel / kernel.sum()
    n = x.shape[dim]
    if radius >= n:  # reflect padding needs radius < size
        radius = n - 1
        kernel = kernel[(kernel.numel() // 2 - radius) : (kernel.numel() // 2 + radius + 1)]
        kernel = kernel / kernel.sum()
    xs = x.movedim(dim, -1).reshape(-1, 1, n)
    xs = F.pad(xs, (radius, radius), mode="reflect")
    out = F.conv1d(xs, kernel.view(1, 1, -1))
    return out.reshape(*x.movedim(dim, -1).shape).movedim(-1, dim)


def _normalized_noise(
    h: int, w: int, sigma_y: float, sigma_x: float, gen: torch.Generator
) -> torch.Tensor:
    """Anisotropically blurred white noise with zero mean, unit std. Shape ``(h, w)``."""
    noise = torch.randn(h, w, generator=gen)
    noise = gaussian_blur_1d(gaussian_blur_1d(noise, sigma_y, 0), sigma_x, 1)
    return (noise - noise.mean()) / (noise.std() + 1e-8)


def _smoothstep(lo: float, hi: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _coords(h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-center coordinates in [-1, 1]: ``u`` along x, ``v`` along y, both ``(h, w)``."""
    u = (torch.arange(w, dtype=torch.float32) + 0.5) / w * 2.0 - 1.0
    v = (torch.arange(h, dtype=torch.float32) + 0.5) / h * 2.0 - 1.0
    return u[None, :].expand(h, w), v[:, None].expand(h, w)


def _flat(h: int, w: int) -> torch.Tensor:
    """Rectangle inset by 1.5 px (1-px antialiased border, empty ring outside). Seed-independent."""
    xs = torch.arange(w, dtype=torch.float32) + 0.5
    ys = torch.arange(h, dtype=torch.float32) + 0.5
    inset = 1.5
    dx = torch.minimum(xs - inset, (w - inset) - xs)  # signed distance to the x edges (inside > 0)
    dy = torch.minimum(ys - inset, (h - inset) - ys)
    dist = torch.minimum(dx[None, :], dy[:, None])
    return (dist + 0.5).clamp(0.0, 1.0)[None]


def _oil(h: int, w: int, gen: torch.Generator) -> torch.Tensor:
    """Bristle streaks along x with ragged edges and tapered, dry ends."""
    u, v = _coords(h, w)
    # Streaks: noise that is smooth along x (long bristle marks) and ~1-2 px wide along y.
    streak = _normalized_noise(h, w, sigma_y=max(0.7, h / 90.0), sigma_x=w / 6.0, gen=gen)
    # Per-bristle length variation (one value per row) -> ragged ends.
    row = _normalized_noise(h, 1, sigma_y=max(0.7, h / 90.0), sigma_x=0.0, gen=gen)[:, :1]
    # Per-column lateral edge wobble (one value per column) -> ragged soft sides.
    col = _normalized_noise(1, w, sigma_y=0.0, sigma_x=w / 24.0, gen=gen)[:1, :]
    end = 1.0 - _smoothstep(0.55, 1.0, u.abs() * (1.0 + 0.12 * row))
    side = 1.0 - _smoothstep(0.65, 1.0, v.abs() + 0.07 * col)
    cover = end * side
    # Bristle contrast grows where coverage drops (dry brush at the ends and the sides).
    contrast = 0.3 + 0.6 * (1.0 - cover)
    return (cover * (0.9 + contrast * streak)).clamp(0.0, 1.0)[None]


def _round(h: int, w: int, gen: torch.Generator) -> torch.Tensor:
    """Soft elliptical dab, mildly textured."""
    u, v = _coords(h, w)
    r = torch.sqrt(u * u + v * v)
    cover = 1.0 - _smoothstep(0.55, 1.0, r)
    tex = _normalized_noise(h, w, sigma_y=max(1.0, h / 32.0), sigma_x=max(1.0, w / 32.0), gen=gen)
    return (cover * (0.95 + 0.15 * tex)).clamp(0.0, 1.0)[None]


def make_brush(kind: str, size: tuple[int, int], seed: int) -> torch.Tensor:
    """Generate one brush mask.

    Args:
        kind: one of ``BRUSH_KINDS``.
        size: ``(h, w)`` in pixels; ``w`` is the main stroke axis (local x). Both must be >= 8.
        seed: RNG seed; the result is bitwise reproducible for a given ``(kind, size, seed)``.

    Returns:
        float32 tensor ``(1, h, w)`` in [0, 1] on CPU.
    """
    h, w = size
    if kind not in BRUSH_KINDS:
        raise ValueError(f"unknown brush kind {kind!r}; expected one of {BRUSH_KINDS}")
    if h < 8 or w < 8:
        raise ValueError(f"brush size must be at least 8x8, got {size}")
    gen = torch.Generator().manual_seed(seed)
    if kind == "flat":
        return _flat(h, w)
    if kind == "oil":
        return _oil(h, w, gen)
    return _round(h, w, gen)


class BrushBank:
    """K variants of one brush kind stacked in a ``(K, 1, h, w)`` tensor (``masks``).

    Variant ``i`` uses seed ``seed + i``. Use :meth:`to` to move it to another device/dtype.
    """

    def __init__(self, masks: torch.Tensor, kind: str, seed: int) -> None:
        if masks.ndim != 4 or masks.shape[1] != 1:
            raise ValueError(f"expected masks of shape (K, 1, h, w), got {tuple(masks.shape)}")
        self.masks = masks
        self.kind = kind
        self.seed = seed

    @classmethod
    def build(cls, kind: str, num_variants: int, size: tuple[int, int], seed: int) -> BrushBank:
        """Generate the bank (CPU, deterministic). ``num_variants`` must be >= 1."""
        if num_variants < 1:
            raise ValueError(f"num_variants must be >= 1, got {num_variants}")
        masks = torch.stack([make_brush(kind, size, seed + i) for i in range(num_variants)])
        return cls(masks, kind, seed)

    @classmethod
    def from_config(cls, cfg: BrushConfig) -> BrushBank:
        """Build (or fetch from the in-memory cache) the bank described by ``cfg``."""
        return get_brush_bank(
            cfg.kind, cfg.bank_size, (cfg.mask_height, cfg.mask_width), cfg.seed
        ).clone()

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        """Mask of variant ``index``, shape ``(1, h, w)``."""
        return self.masks[index]

    @property
    def size(self) -> tuple[int, int]:
        """``(h, w)`` of the masks."""
        return int(self.masks.shape[2]), int(self.masks.shape[3])

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> BrushBank:
        """Return a bank whose masks are moved to ``device`` / ``dtype``."""
        return BrushBank(self.masks.to(device=device, dtype=dtype), self.kind, self.seed)

    def clone(self) -> BrushBank:
        """Independent copy (so callers cannot mutate the cached tensors)."""
        return BrushBank(self.masks.clone(), self.kind, self.seed)


@lru_cache(maxsize=16)
def get_brush_bank(kind: str, num_variants: int, size: tuple[int, int], seed: int) -> BrushBank:
    """Memoized :meth:`BrushBank.build`. The returned (shared) bank must not be mutated."""
    return BrushBank.build(kind, num_variants, size, seed)
