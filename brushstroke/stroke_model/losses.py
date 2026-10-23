"""Paint Transformer losses: pixel L1, Hungarian-matched stroke losses and confidence BCE.

Follows ``train/models/painter_model.py:optimize_parameters`` of the official code (commit 7960b8e)
and the paper
(Section 3.4, Eq. 2 and 7-13); the set loss with Hungarian matching is that of DETR (Carion et al.,
ECCV 2020), and the
stroke distance is the Wasserstein-2 distance between the 2-D Gaussians of the rotated rectangles
(the Gaussian
formulation for rotated boxes of Yang et al., "Rethinking rotated object detection with Gaussian
Wasserstein
distance loss", ICML 2021, cited by the paper as [33]). See ``docs/design/paint_transformer.md``
sections 4 and 9.

Per batch of patches with ``N = 8`` strokes: the 8 predictions are matched one to one with the
*valid* ground-truth
strokes (cost = parameter L1 + Wasserstein + confidence cost), then

* ``pixel``: ``L1(rec, target) * lambda_pixel`` where ``rec`` = the canvas with the predicted
  strokes painted;
* ``gt``: parameter L1 over the matched pairs ``* lambda_gt``;
* ``w``: Wasserstein distance over the matched pairs ``* lambda_w``;
* ``decision``: BCE-with-logits over all predictions (matched = positive, ``pos_weight``) ``*
  lambda_decision``.

Presets (``PTLossConfig.preset``): ``official`` = the code (``pos_weight`` 10, BCE 10, pixel 10,
stroke L1 1,
Wasserstein 10, matching cost ``1 - logit``); ``paper_weights`` = the paper (``lambda_r`` = 8 as
``pos_weight``,
BCE 1, pixel 1, stroke L1 1, Wasserstein 10, matching cost = ``lambda_r``-weighted BCE).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from functools import lru_cache

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from brushstroke.renderer.pt import pt_composite, pt_param2stroke
from brushstroke.stroke_model.model import sign_with_sigmoid_grad
from brushstroke.stroke_model.synthetic import PatchBatch

MATCHING_COSTS = ("logit", "bce")
MATCHERS = ("auto", "scipy", "gpu")
MAX_GPU_STROKES = 8  # exhaustive GPU matching enumerates N! permutations


@dataclass
class PTLossConfig:
    """``loss:`` config section. Set ``preset`` to ``official`` or ``paper_weights`` to use a
    documented preset."""

    preset: str = "official"  # official | paper_weights | custom (use the fields below as given)
    lambda_pixel: float = 10.0
    lambda_gt: float = 1.0  # stroke parameter L1
    lambda_w: float = 10.0  # Wasserstein
    lambda_decision: float = 10.0  # confidence BCE
    pos_weight: float = 10.0  # recall weight lambda_r of the BCE (official ``lambda_recall``)
    matching_cost: str = (
        "logit"  # logit: 1 - logit (official code) | bce: pos_weight * -log sigmoid(logit) (paper)
    )
    matcher: str = (
        "auto"  # auto | scipy | gpu (exhaustive search on the GPU for N <= 8, same optimum)
    )


PRESETS: dict[str, dict[str, float | str]] = {
    "official": {
        "lambda_pixel": 10.0,
        "lambda_gt": 1.0,
        "lambda_w": 10.0,
        "lambda_decision": 10.0,
        "pos_weight": 10.0,
        "matching_cost": "logit",
    },
    "paper_weights": {
        "lambda_pixel": 1.0,
        "lambda_gt": 1.0,
        "lambda_w": 10.0,
        "lambda_decision": 1.0,
        "pos_weight": 8.0,
        "matching_cost": "bce",
    },
}


def resolve_loss_config(cfg: PTLossConfig) -> PTLossConfig:
    """Apply the named preset (``official`` / ``paper_weights``); ``custom`` keeps the fields as
    given."""
    if cfg.preset == "custom":
        resolved = cfg
    elif cfg.preset in PRESETS:
        resolved = replace(cfg, **PRESETS[cfg.preset])  # type: ignore[arg-type]
    else:
        raise ValueError(f"preset must be one of {sorted(PRESETS)} or 'custom', got {cfg.preset!r}")
    if resolved.matching_cost not in MATCHING_COSTS or resolved.matcher not in MATCHERS:
        raise ValueError(f"matching_cost in {MATCHING_COSTS} and matcher in {MATCHERS} required")
    return resolved


# ----------------------------------------------------------------------------- stroke distances


def _sigma_sqrt(w: torch.Tensor, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``R diag(w/2, h/2) R^T`` as ``(..., 2, 2)`` (the square root of the stroke covariance, paper
    Eq. 9)."""
    c, s = torch.cos(theta), torch.sin(theta)
    a = w * c**2 / 2 + h * s**2 / 2
    b = (w - h) * c * s / 2
    d = h * c**2 / 2 + w * s**2 / 2
    return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)


def _sigma(w: torch.Tensor, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """``R diag(w^2/4, h^2/4) R^T`` as ``(..., 2, 2)``."""
    c, s = torch.cos(theta), torch.sin(theta)
    a = w * w * c**2 / 4 + h * h * s**2 / 4
    b = (w * w - h * h) * c * s / 4
    d = h * h * c**2 / 4 + w * w * s**2 / 4
    return torch.stack([torch.stack([a, b], -1), torch.stack([b, d], -1)], -2)


def gaussian_w_distance(p1: torch.Tensor, p2: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """Squared Wasserstein-2 distance between the Gaussians of two strokes ``[x, y, w, h, theta]``
    (paper Eq. 10).

    ``|mu_1 - mu_2|^2 + Tr(S_1) + Tr(S_2) - 2 Tr((S_1^{1/2} S_2 S_1^{1/2})^{1/2})`` with ``mu = (x,
    y)``,
    ``S = R(theta) diag(w^2/4, h^2/4) R(theta)^T`` and ``theta`` in units of pi (``theta * pi``
    radians). For a 2 x 2
    positive semi-definite ``M``: ``Tr(M^{1/2}) = sqrt(Tr M + 2 sqrt(det M))``, which is what is
    evaluated (as in the
    official code); the arguments of the square roots are clamped at ``eps`` to keep gradients
    finite.

    Args:
        p1, p2: ``(..., >= 5)`` broadcastable strokes.

    Returns:
        ``(...)`` distances, >= 0, symmetric, 0 for identical strokes.
    """
    pi = torch.acos(torch.tensor(-1.0, device=p1.device, dtype=p1.dtype))
    x1, y1, w1, h1, t1 = (p1[..., i] for i in range(5))
    x2, y2, w2, h2, t2 = (p2[..., i] for i in range(5))
    t1, t2 = pi * t1, pi * t2
    trace_1, trace_2 = (w1**2 + h1**2) / 4, (w2**2 + h2**2) / 4
    m = _sigma_sqrt(w1, h1, t1) @ _sigma(w2, h2, t2) @ _sigma_sqrt(w1, h1, t1)
    det = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] * m[..., 1, 0]
    trace_m = m[..., 0, 0] + m[..., 1, 1]
    trace_root = torch.sqrt((trace_m + 2 * torch.sqrt(det.clamp_min(eps))).clamp_min(eps))
    return (x1 - x2) ** 2 + (y1 - y2) ** 2 + trace_1 + trace_2 - 2 * trace_root


# ----------------------------------------------------------------------------- matching


@lru_cache(maxsize=4)
def _permutations(n: int, device: str) -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long, device=device)


def match_scipy(cost: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Hungarian matching per sample with ``linear_sum_assignment`` (as DETR and the official code).

    Args:
        cost: ``(B, N, N)`` cost of assigning prediction ``u`` (rows) to ground-truth stroke ``v``
        (columns).
        valid: ``(B, N)`` bool, valid ground-truth columns.

    Returns:
        ``(B, N)`` long: for each ground-truth column the matched prediction index (``-1`` for
        invalid columns).
    """
    costs = cost.detach().cpu().numpy()
    valids = valid.cpu().numpy()
    assign = torch.full(valid.shape, -1, dtype=torch.long)
    for b in range(cost.shape[0]):
        cols = valids[b].nonzero()[0]
        if cols.size:
            rows, picked = linear_sum_assignment(costs[b][:, cols])
            assign[b, torch.as_tensor(cols[picked])] = torch.as_tensor(rows)
    return assign.to(cost.device)


def match_gpu(cost: torch.Tensor, valid: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """Exact optimal matching by enumerating the ``N!`` assignments of the columns to the rows,
    batched on the device.

    The minimum total cost over the valid columns is the same as the Hungarian algorithm's (ties
    may pick another
    optimal assignment); no host synchronization per sample. Only for ``N <= 8`` (40,320
    permutations).
    """
    b, n, _ = cost.shape
    if n > MAX_GPU_STROKES:
        raise ValueError(f"exhaustive matching supports N <= {MAX_GPU_STROKES}, got {n}")
    perms = _permutations(n, str(cost.device))  # (P, N): row assigned to column k
    cols = torch.arange(n, device=cost.device)
    masked = cost.detach() * valid.unsqueeze(1).to(cost.dtype)  # invalid columns cost nothing
    best = torch.empty(b, n, dtype=torch.long, device=cost.device)
    for start in range(0, b, chunk):
        part = masked[start : start + chunk]
        totals = part[:, perms, cols].sum(dim=-1)  # (b, P)
        best[start : start + chunk] = perms[totals.argmin(dim=1)]
    return torch.where(valid, best, torch.full_like(best, -1))


def total_cost(cost: torch.Tensor, valid: torch.Tensor, assign: torch.Tensor) -> torch.Tensor:
    """Total matching cost ``(B,)`` of an assignment returned by :func:`match_scipy` /
    :func:`match_gpu`."""
    b, n, _ = cost.shape
    rows = assign.clamp_min(0)  # (B, N): matched row of each column
    cols = torch.arange(n, device=cost.device).expand(b, n)
    per_column = cost[torch.arange(b, device=cost.device).unsqueeze(1), rows, cols]
    return (per_column * valid.to(cost.dtype)).sum(dim=1)


# ----------------------------------------------------------------------------- the loss


@dataclass
class LossOutput:
    """Weighted loss terms (as the official ``loss_*`` attributes), the total and the rendered
    prediction."""

    total: torch.Tensor
    pixel: torch.Tensor
    gt: torch.Tensor
    w: torch.Tensor
    decision: torch.Tensor
    rec: torch.Tensor  # (B, 3, P, P) canvas with the predicted strokes painted
    assign: torch.Tensor  # (B, N) matched prediction of each valid GT stroke (-1 otherwise)


class PaintTransformerLoss:
    """Loss of a batch of patches; ``brushes`` are the ``train`` brush images
    (``load_pt_brushes("train")``)."""

    def __init__(
        self, brushes: torch.Tensor, cfg: PTLossConfig | None = None, colour_mode: str = "sampled"
    ):
        self.cfg = resolve_loss_config(cfg or PTLossConfig())
        self.brushes = brushes
        self.colour_mode = colour_mode

    def cost_matrix(
        self,
        pred: torch.Tensor,
        logits: torch.Tensor,
        gt: torch.Tensor,
    ) -> torch.Tensor:
        """``(B, N, N)`` matching cost between predictions (rows) and ground-truth strokes
        (columns)."""
        l1 = (pred.unsqueeze(2) - gt.unsqueeze(1)).abs().sum(dim=-1)
        wass = gaussian_w_distance(pred.unsqueeze(2), gt.unsqueeze(1))
        if self.cfg.matching_cost == "logit":
            conf = (1 - logits).unsqueeze(2)  # official: 1 - raw logit
        else:
            conf = (self.cfg.pos_weight * F.softplus(-logits)).unsqueeze(
                2
            )  # -lambda_r log sigmoid(c)
        return l1 + wass + conf

    def match(self, cost: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        matcher = self.cfg.matcher
        if matcher == "auto":
            matcher = "gpu" if cost.is_cuda and cost.shape[1] <= MAX_GPU_STROKES else "scipy"
        return match_gpu(cost, valid) if matcher == "gpu" else match_scipy(cost, valid)

    def render(
        self, canvas: torch.Tensor, shape: torch.Tensor, colour: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Paint the predicted strokes (binary decision with the sigmoid straight-through
        gradient)."""
        b, n, _ = shape.shape
        p = canvas.shape[-1]
        param8 = torch.cat([shape[..., :5], colour], dim=-1).reshape(b * n, 8)
        fg, alpha = pt_param2stroke(param8, p, p, self.brushes, "pt_train")
        decision = sign_with_sigmoid_grad(logits)
        return pt_composite(canvas, fg.view(b, n, 3, p, p), alpha.view(b, n, 3, p, p), decision)

    def __call__(
        self,
        shape: torch.Tensor,
        colour: torch.Tensor,
        logits: torch.Tensor,
        batch: PatchBatch,
    ) -> LossOutput:
        """Loss of predictions ``(shape (B, N, 5), colour (B, N, 3), logits (B, N))`` against
        ``batch``."""
        cfg = self.cfg
        valid = batch.gt_decision.bool()
        rec = self.render(batch.canvas, shape, colour, logits)
        pixel = F.l1_loss(rec, batch.target) * cfg.lambda_pixel
        if self.colour_mode == "predicted":
            pred_all, gt_all = (
                torch.cat([shape, colour], -1),
                torch.cat([batch.gt_param, batch.gt_color], -1),
            )
        else:
            pred_all, gt_all = shape, batch.gt_param
        with torch.no_grad():
            cost = self.cost_matrix(pred_all.detach(), logits.detach(), gt_all)
            assign = self.match(cost, valid)
        bi, vi = torch.nonzero(valid, as_tuple=True)  # every valid GT stroke
        pi = assign[bi, vi]  # its matched prediction
        paired_pred, paired_gt = pred_all[bi, pi], gt_all[bi, vi]
        target_decision = torch.zeros_like(logits)
        target_decision[bi, pi] = 1.0
        gt = F.l1_loss(paired_pred, paired_gt) * cfg.lambda_gt
        w = gaussian_w_distance(shape[bi, pi], batch.gt_param[bi, vi]).mean() * cfg.lambda_w
        pos_weight = torch.tensor(cfg.pos_weight, device=logits.device, dtype=logits.dtype)
        decision = (
            F.binary_cross_entropy_with_logits(logits, target_decision, pos_weight=pos_weight)
            * cfg.lambda_decision
        )
        return LossOutput(pixel + gt + w + decision, pixel, gt, w, decision, rec, assign)
