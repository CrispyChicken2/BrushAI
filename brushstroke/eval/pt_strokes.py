"""Stroke-level metrics of the Paint Transformer paper (Table 1, right part) on synthesized strokes.

``L_pixel`` (canvas with the predicted strokes painted against the target patch), ``D_L1``
(parameter L1 over the matched stroke pairs) and ``D_W`` (Gaussian Wasserstein distance over the
matched pairs), all **unweighted** terms of
:class:`~brushstroke.stroke_model.losses.PaintTransformerLoss` with the official matching.
:func:`optim_strokes` is the "Optim" comparison of the paper: the same number of strokes per patch,
random initialization, gradient descent on the patch L1 (protocol in
``docs/experiments/pt_reproduction.md``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from brushstroke.stroke_model.losses import PaintTransformerLoss, PTLossConfig
from brushstroke.stroke_model.model import PaintTransformer
from brushstroke.stroke_model.refine import RefineConfig, refine_patch_strokes
from brushstroke.stroke_model.synthetic import PatchBatch

CONFIDENT = (
    10.0  # logit given to the strokes of an optimizer (painted), the opposite for the others
)


def unweighted_loss(brushes: torch.Tensor) -> PaintTransformerLoss:
    """The official loss with every weight 1: its terms are the raw metrics ``L_pixel``, ``D_L1``,
    ``D_W``. ``brushes``: the ``train`` brush images."""
    cfg = PTLossConfig(
        preset="custom",
        lambda_pixel=1.0,
        lambda_gt=1.0,
        lambda_w=1.0,
        lambda_decision=1.0,
        pos_weight=10.0,
        matching_cost="logit",
    )
    return PaintTransformerLoss(brushes, cfg)


@dataclass
class StrokeScores:
    """Sums over a set of patches (``pixel`` per patch, ``gt``/``w`` per matched pair)."""

    pixel: float = 0.0
    patches: int = 0
    l1: float = 0.0
    w: float = 0.0
    pairs: int = 0
    negative_h: int = 0
    negative_w: int = 0
    painted: int = 0

    def add(
        self,
        loss: PaintTransformerLoss,
        shape: torch.Tensor,
        colour: torch.Tensor,
        logits: torch.Tensor,
        batch: PatchBatch,
    ) -> None:
        out = loss(shape, colour, logits, batch)
        valid = batch.gt_decision.bool()
        rows, cols = torch.nonzero(valid, as_tuple=True)
        pairs = int(rows.numel())
        n = int(batch.target.shape[0])
        self.pixel += float(out.pixel) * n
        self.patches += n
        self.l1 += float(out.gt) * pairs
        self.w += float(out.w) * pairs
        self.pairs += pairs
        matched = shape[rows, out.assign[rows, cols]]
        self.negative_h += int((matched[:, 3] < 0).sum())
        self.negative_w += int((matched[:, 2] < 0).sum())
        self.painted += int((logits > 0).sum())

    def result(self) -> dict[str, float]:
        return {
            "l_pixel": self.pixel / self.patches,
            "d_l1": self.l1 / self.pairs,
            "d_w": self.w / self.pairs,
            "h_negative": self.negative_h / self.pairs,
            "w_negative": self.negative_w / self.pairs,
            "painted_per_patch": self.painted / self.patches,
        }


@torch.no_grad()
def network_scores(
    net: PaintTransformer,
    loss: PaintTransformerLoss,
    data: Iterable[PatchBatch],
    abs_h: bool,
    abs_w: bool = False,
) -> dict[str, float]:
    """Stroke metrics of a network on ``data`` (eval mode); ``abs_h`` / ``abs_w`` replace ``h`` /
    ``w`` by their absolute value (a diagnostic of the parameter convention: the fold changes the
    rendered strokes, it is not an exact symmetry of the renderer)."""
    net.eval()
    scores = StrokeScores()
    for batch in data:
        shape, colour, logits = net.strokes(batch.target, batch.canvas)
        if abs_h:
            shape = shape.clone()
            shape[..., 3] = shape[..., 3].abs()
        if abs_w:
            shape = shape.clone()
            shape[..., 2] = shape[..., 2].abs()
        scores.add(loss, shape, colour, logits, batch)
    return scores.result()


def optim_strokes(
    batch: PatchBatch,
    painted: torch.Tensor,
    brushes: torch.Tensor,
    cfg: RefineConfig,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Optimize ``painted.sum(1)`` strokes per patch from random initial strokes.

    Args:
        painted: ``(B, N)`` bool, the strokes that exist (the others stay unpainted).
        brushes: ``train`` brush images.
        cfg: optimizer settings (the M1 optimizer pieces of :mod:`brushstroke.stroke_model.refine`).

    Returns:
        ``(shape (B, N, 5), colour (B, N, 3), logits (B, N))`` ready for :class:`StrokeScores`.
    """
    g = torch.Generator(device=batch.target.device).manual_seed(seed)
    b, n = painted.shape
    shape = torch.rand(b, n, 5, generator=g, device=painted.device)
    shape[..., :2] = 0.2 + 0.6 * shape[..., :2]  # x, y
    shape[..., 2:4] = 0.2 + 0.5 * shape[..., 2:4]  # w, h
    colour = PaintTransformer.sample_colour(batch.target, shape)
    params = torch.cat([shape, colour], dim=-1)
    out, _ = refine_patch_strokes(batch.target, batch.canvas, params, painted, brushes, cfg)
    logits = torch.where(painted, CONFIDENT, -CONFIDENT).to(out.dtype)
    return out[..., :5], out[..., 5:], logits
