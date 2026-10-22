"""Loss terms of the optimization baseline: pixel L1, LPIPS-VGG and Sinkhorn OT.

All functions take images ``(B, 3, H, W)`` in [0, 1]. The combined loss is
``l1 * L1 + lpips * LPIPS + ot * Sinkhorn`` (:class:`CombinedLoss`).
"""

from __future__ import annotations

import torch

from brushstroke.baselines.optim.config import LossConfig
from brushstroke.baselines.optim.sinkhorn import image_ot_loss

_LPIPS_CACHE: dict[str, torch.nn.Module] = {}


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (scalar)."""
    return (pred - target).abs().mean()


def get_lpips(device: torch.device | str) -> torch.nn.Module:
    """LPIPS-VGG network (Zhang et al., CVPR 2018) on ``device``, frozen, eval mode, cached.

    The VGG16 backbone weights are downloaded by torchvision on first use.
    """
    key = str(device)
    if key not in _LPIPS_CACHE:
        import lpips

        net = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = net
    return _LPIPS_CACHE[key]


def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean LPIPS-VGG distance (scalar); inputs in [0, 1] are mapped to [-1, 1] for the net."""
    net = get_lpips(pred.device)
    return torch.as_tensor(net(pred * 2.0 - 1.0, target * 2.0 - 1.0)).mean()


class CombinedLoss:
    """Weighted sum of the enabled loss terms (weight > 0)."""

    def __init__(self, cfg: LossConfig) -> None:
        if cfg.l1 < 0 or cfg.lpips < 0 or cfg.ot < 0:
            raise ValueError("loss weights must be non-negative")
        if cfg.l1 == cfg.lpips == cfg.ot == 0:
            raise ValueError("at least one loss weight must be positive")
        self.cfg = cfg

    def terms(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """The weighted enabled terms by name."""
        c = self.cfg
        out: dict[str, torch.Tensor] = {}
        if c.l1 > 0:
            out["l1"] = c.l1 * l1_loss(pred, target)
        if c.lpips > 0:
            out["lpips"] = c.lpips * lpips_loss(pred, target)
        if c.ot > 0:
            out["ot"] = c.ot * image_ot_loss(pred, target, c.ot_size, c.ot_eps, c.ot_iters)
        return out

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum(self.terms(pred, target).values(), start=torch.zeros((), device=pred.device))
