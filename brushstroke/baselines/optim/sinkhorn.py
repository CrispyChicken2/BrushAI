"""Log-domain Sinkhorn optimal transport on a small image grid.

Entropy-regularized optimal transport (Cuturi, *Sinkhorn Distances*, NIPS 2013), iterated in the
log domain for numerical stability (as in Peyre & Cuturi, *Computational Optimal Transport*, 2019).
Stylized Neural Painting (Zou et al., CVPR 2021) adds such a Sinkhorn term to the pixel L1 loss
because transport distances give useful gradients when strokes do not overlap their target yet.

:func:`sinkhorn_divergence` is the debiased value ``OT(a, b) - OT(a, a) / 2 - OT(b, b) / 2``:
0 for identical distributions, symmetric, non-negative, decreasing as distributions approach.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_cost(
    size: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Squared Euclidean cost between the cells of a ``size x size`` grid in the unit square.

    Returns ``(size**2, size**2)``; cell ``(i, j)`` is centred at ``((j+.5)/size, (i+.5)/size)``.
    """
    c = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
    ys, xs = torch.meshgrid(c, c, indexing="ij")
    pts = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    return torch.cdist(pts, pts).square()


def sinkhorn_ot(
    a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor, eps: float = 0.02, iters: int = 30
) -> torch.Tensor:
    """Entropic OT value (dual form ``<a, f> + <b, g>``) between batched distributions.

    Args:
        a: ``(B, n)`` non-negative weights summing to 1 per row.
        b: ``(B, m)`` likewise.
        cost: ``(n, m)`` ground cost.
        eps: entropic regularization (in cost units).
        iters: number of alternating dual updates (differentiable, unrolled).

    Returns:
        ``(B,)`` regularized transport cost.
    """
    log_a = torch.log(a.clamp_min(1e-30))
    log_b = torch.log(b.clamp_min(1e-30))
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(iters):
        f = -eps * torch.logsumexp((g[:, None, :] - cost[None]) / eps + log_b[:, None, :], dim=2)
        g = -eps * torch.logsumexp((f[:, :, None] - cost[None]) / eps + log_a[:, :, None], dim=1)
    return (a * f).sum(1) + (b * g).sum(1)


def sinkhorn_divergence(
    a: torch.Tensor, b: torch.Tensor, cost: torch.Tensor, eps: float = 0.02, iters: int = 30
) -> torch.Tensor:
    """Debiased Sinkhorn divergence ``OT(a,b) - OT(a,a)/2 - OT(b,b)/2`` per batch element, ``(B,)``.

    ``cost`` must be square and symmetric (same support for ``a`` and ``b``).
    """
    return (
        sinkhorn_ot(a, b, cost, eps, iters)
        - 0.5 * sinkhorn_ot(a, a, cost, eps, iters)
        - 0.5 * sinkhorn_ot(b, b, cost, eps, iters)
    )


def image_ot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    size: int = 24,
    eps: float = 0.02,
    iters: int = 30,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Sinkhorn divergence between the intensity distributions of two images (mean over B*3).

    Each colour channel of the images ``(B, 3, H, W)`` is average-pooled to ``size x size``, turned
    into a probability distribution over the grid cells (plus a small ``floor`` so that no cell has
    zero mass) and compared with the debiased Sinkhorn divergence under the squared grid distance.
    """

    def dist(x: torch.Tensor) -> torch.Tensor:
        small = F.adaptive_avg_pool2d(x, size).reshape(-1, size * size) + floor
        return small / small.sum(dim=1, keepdim=True)

    cost = grid_cost(size, pred.device, pred.dtype)
    return sinkhorn_divergence(dist(pred), dist(target), cost, eps, iters).mean()
