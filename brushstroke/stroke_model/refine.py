"""Optional per-patch optimization refinement of the predicted strokes (T-015).

The network predicts the 8 strokes of a patch in one forward pass. Here the strokes of every patch
of a layer are fine-tuned by gradient descent on the pixel loss of that patch, which is the
objective the network was trained on (Liu et al., ICCV 2021, pixel term ``L_pixel``): the strokes
are painted over the canvas patch and compared with the target patch, in the *training* rendering
(``pt_train``: binarized brush warped bilinearly, so the stroke edges carry a gradient with respect
to the geometry; the inference alpha is binary and has none).

The optimizer is the one of the M1 baseline (:mod:`brushstroke.baselines.optim`): Adam on
sigmoid-reparameterized parameters (:func:`~brushstroke.renderer.params.params_from_unconstrained`)
with the same cosine learning-rate decay. The confidence decisions are kept; only the shape and
(optionally) the colour change. A patch keeps its refined strokes only if its patch loss is lower
than with the network strokes (``keep_if_better``), so the proxy objective never gets worse.

All the patches of a layer are optimized at once (the patches are independent: the sum of the
patch losses is minimized). The proxy is not exactly the inference rendering (32 px patch, strokes
clipped at the patch border, soft alpha) so the gain on the final image has to be measured, see
``docs/design/paint_transformer.md`` section 14.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from brushstroke.baselines.optim.init import cosine_lr
from brushstroke.baselines.optim.optimize import make_optimizer
from brushstroke.renderer.params import params_from_unconstrained
from brushstroke.renderer.params import unconstrained_from_params as to_unconstrained
from brushstroke.renderer.pt import NUM_PT_PARAMS, render_pt


@dataclass
class RefineConfig:
    """``refine:`` config section (off by default)."""

    enabled: bool = False
    levels: list[int] = field(default_factory=list)  # layers to refine (empty = all, incl. final)
    iters: int = 100  # optimization steps per layer
    lr: float = 0.05  # on the unconstrained (logit) parameters (the M1 baseline uses 0.02)
    lr_final_frac: float = 0.1  # cosine decay to lr * lr_final_frac
    optimizer: str = "adam"  # adam | rmsprop
    optimize_colour: bool = True  # also optimize the colour (otherwise the sampled one is kept)
    keep_if_better: bool = True  # per patch, keep the network strokes if refinement did not help


def patch_loss(
    params: torch.Tensor,
    img_patches: torch.Tensor,
    canvas_patches: torch.Tensor,
    painted: torch.Tensor,
    brushes: torch.Tensor,
) -> torch.Tensor:
    """Per-patch L1 between the target and the canvas with the strokes painted (training
    rendering).

    Args:
        params: ``(n, N, 8)`` strokes in patch units.
        img_patches, canvas_patches: ``(n, 3, P, P)``.
        painted: ``(n, N)`` bool decisions.
        brushes: ``(2, 1, h_b, w_b)`` brush images.

    Returns:
        ``(n,)`` mean absolute error of every patch.
    """
    size = img_patches.shape[-1]
    out = render_pt(params, size, size, brushes, "pt_train", painted, canvas_patches)
    return (out - img_patches).abs().mean(dim=(1, 2, 3))


def refine_patch_strokes(
    img_patches: torch.Tensor,
    canvas_patches: torch.Tensor,
    params: torch.Tensor,
    painted: torch.Tensor,
    brushes: torch.Tensor,
    cfg: RefineConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Refine the strokes of every patch of one layer.

    Args:
        img_patches, canvas_patches: ``(n, 3, P, P)`` target and canvas patches.
        params: ``(n, N, 8)`` strokes in patch units (``x, y, w, h, theta, r, g, b``).
        painted: ``(n, N)`` bool, strokes that are painted (the others are left unchanged).
        brushes: ``(2, 1, h_b, w_b)`` brush images.
        cfg: refinement options.

    Returns:
        ``(params, stats)``: refined strokes ``(n, N, 8)`` and the mean patch loss before and after
        (``loss_before``, ``loss_after``), the fraction of patches that keep the refinement.
    """
    if params.ndim != 3 or params.shape[-1] != NUM_PT_PARAMS:
        raise ValueError(f"expected params of shape (n, N, {NUM_PT_PARAMS}), got {params.shape}")
    with torch.enable_grad():
        start = params.detach()
        init = start.clamp(min=0.0, max=1.0)  # the sigmoid reparameterization lives in [0, 1]
        init[..., 2:4] = init[..., 2:4].clamp(min=1e-3)
        raw = to_unconstrained(torch.cat([init, torch.ones_like(init[..., :1])], dim=-1))
        raw.requires_grad_(True)
        opt = make_optimizer(cfg.optimizer, [{"params": [raw], "lr": cfg.lr}], cfg.lr)
        for it in range(cfg.iters):
            for group in opt.param_groups:
                group["lr"] = cosine_lr(cfg.lr, cfg.lr_final_frac, it, cfg.iters)
            opt.zero_grad(set_to_none=True)
            current = _compose(raw, start, cfg)
            loss = patch_loss(current, img_patches, canvas_patches, painted, brushes).sum()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
        with torch.no_grad():
            refined = _compose(raw, start, cfg)
            before = patch_loss(start, img_patches, canvas_patches, painted, brushes)
            after = patch_loss(refined, img_patches, canvas_patches, painted, brushes)
            keep = (
                after < before if cfg.keep_if_better else torch.ones_like(before, dtype=torch.bool)
            )
            move = keep[:, None, None] & painted[:, :, None]  # only painted strokes change
            out = torch.where(move, refined, start)
            final = patch_loss(out, img_patches, canvas_patches, painted, brushes)
    stats = {
        "loss_before": float(before.mean()),
        "loss_after": float(final.mean()),
        "patches_kept": float(keep.float().mean()),
    }
    return out, stats


def _compose(raw: torch.Tensor, start: torch.Tensor, cfg: RefineConfig) -> torch.Tensor:
    """Strokes ``(n, N, 8)`` from the unconstrained parameters (colour optionally frozen)."""
    p = params_from_unconstrained(raw)[..., :NUM_PT_PARAMS]
    if cfg.optimize_colour:
        return p
    return torch.cat([p[..., :5], start[..., 5:]], dim=-1)
