"""Alpha compositing of ordered strokes with the "over" operator.

``C_k = C_{k-1} (1 - a_k) + c_k a_k`` for strokes ``k = 0..n-1`` in index order (``C_{-1}`` =
initial canvas). Unrolling gives the closed form used by :func:`composite_parallel`::

    C_n = C_{-1} * S_0 + sum_k c_k a_k T_k
    S_k = prod_{j >= k} (1 - a_j),   T_k = S_{k+1},   T_{n-1} = 1

``S`` is one reverse cumulative product, so a whole chunk of strokes is composited without a Python
loop over strokes (the chunk's alphas must all be materialized: memory ~ ``n * H * W``; chunks are
chained by the caller to bound it).
"""

from __future__ import annotations

import torch


def composite_sequential(
    canvas: torch.Tensor, alphas: torch.Tensor, colors: torch.Tensor
) -> torch.Tensor:
    """Naive reference: loop over strokes.

    Args:
        canvas: ``(B, 3, H, W)`` initial canvas.
        alphas: ``(B, n, 1, H, W)`` stroke alphas in [0, 1].
        colors: ``(B, n, 3)`` stroke colours.

    Returns:
        ``(B, 3, H, W)`` canvas after the n strokes.
    """
    out = canvas
    for k in range(alphas.shape[1]):
        a = alphas[:, k]
        out = out * (1.0 - a) + colors[:, k, :, None, None] * a
    return out


def composite_parallel(
    canvas: torch.Tensor, alphas: torch.Tensor, colors: torch.Tensor
) -> torch.Tensor:
    """Closed-form composite of one chunk using reverse cumulative products (same args as above)."""
    if alphas.shape[1] == 0:
        return canvas
    keep = 1.0 - alphas  # (B, n, 1, H, W)
    suffix = torch.flip(torch.cumprod(torch.flip(keep, dims=[1]), dim=1), dims=[1])  # S_k
    behind = torch.cat([suffix[:, 1:], torch.ones_like(suffix[:, :1])], dim=1)  # T_k = S_{k+1}
    paint = (colors[..., None, None] * alphas * behind).sum(dim=1)
    return canvas * suffix[:, 0] + paint
