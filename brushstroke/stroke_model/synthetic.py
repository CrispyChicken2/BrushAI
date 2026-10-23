"""On-the-fly synthetic training data of Paint Transformer (study note section 2).

Follows ``set_input`` of the official code (``train/models/painter_model.py`` at commit 7960b8e):

* ``B / 4`` background canvases of ``2P x 2P`` pixels, each with ``N`` strokes (``x, y, w, h ~
  U(0.2, 0.7)``,
  ``theta ~ U(0, 1)``, uniform colour), all painted over black, then cut into ``2 x 2`` patches of
  ``P x P``;
* for every patch, ``N`` foreground strokes sampled the same way, rendered at ``P x P`` (clipped at
  the patch
  border) over the background patch; stroke ``i`` is *invalid* (label 0, not painted) if, for some
  earlier
  valid stroke ``j``, ``sum(alpha_i * alpha_j) / sum(alpha_j) >= 0.75`` (the code's threshold; the
  paper says 60 %).

Strokes are rendered with the ``pt_train`` pipeline of :mod:`brushstroke.renderer.pt` (official
train brush images).
The random draws have the shape of the official ones (``rand(B/4, N, 12)`` then ``rand(B, N, 12)``,
12 columns
``x, y, w, h, theta, r0, g0, b0, r2, g2, b2, a`` of which ``r2, g2, b2`` and ``a`` are unused), so
that with the same
CPU generator state the batches equal the official ones (tested against the official code when it
is available).
Everything runs on the device of ``brushes`` (GPU), seeded through a ``torch.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from brushstroke.renderer.pt import pt_composite, pt_param2stroke

PARAM_COLUMNS = (
    12  # width of the official random parameter draw (x, y, w, h, theta, r0, g0, b0, r2, g2, b2, a)
)
SHAPE_PARAMS = 5


@dataclass
class SyntheticConfig:
    """``synthetic:`` config section (defaults = the official recipe)."""

    patch_size: int = 32  # P
    strokes_per_patch: int = 8  # N
    low: float = 0.2  # x, y, w, h = low + span * U(0, 1)
    span: float = 0.5
    iou_threshold: float = (
        0.75  # a stroke covering >= this fraction of an earlier valid stroke is invalid
    )


@dataclass
class PatchBatch:
    """A batch of training patches.

    Attributes:
        canvas: ``(B, 3, P, P)`` background ("old") patches, input canvas of the model.
        target: ``(B, 3, P, P)`` target patches (background + valid foreground strokes), input
        image of the model.
        gt_param: ``(B, N, 5)`` foreground shape parameters ``x, y, w, h, theta`` (all strokes,
        valid or not).
        gt_decision: ``(B, N)`` 1.0 for valid strokes, 0.0 for invalid ones.
        gt_color: ``(B, N, 3)`` colours of the foreground strokes.
    """

    canvas: torch.Tensor
    target: torch.Tensor
    gt_param: torch.Tensor
    gt_decision: torch.Tensor
    gt_color: torch.Tensor


class SyntheticStrokeGenerator:
    """Batched, seeded generator of :class:`PatchBatch` on the device of ``brushes``."""

    def __init__(self, brushes: torch.Tensor, cfg: SyntheticConfig | None = None) -> None:
        """``brushes``: ``(2, 1, h, w)`` from ``load_pt_brushes("train")``, already on the target
        device."""
        self.cfg = cfg or SyntheticConfig()
        self.brushes = brushes
        self.device = brushes.device

    def _draw(self, shape: tuple[int, int], generator: torch.Generator | None) -> torch.Tensor:
        """``(B, N, 12)`` official-style random strokes: first four columns in ``[low, low +
        span)``."""
        u = torch.rand(*shape, PARAM_COLUMNS, generator=generator, device=self.device)
        u[..., :4] = u[..., :4] * self.cfg.span + self.cfg.low
        return u

    def _render(self, raw: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Foregrounds and alphas ``(B, N, 3, size, size)`` of ``(B, N, 12)`` draws (``pt_train``
        pipeline)."""
        b, n, _ = raw.shape
        fg, alpha = pt_param2stroke(
            raw[..., :8].reshape(-1, 8), size, size, self.brushes, "pt_train"
        )
        return fg.view(b, n, 3, size, size), alpha.view(b, n, 3, size, size)

    def sample(self, batch_size: int, generator: torch.Generator | None = None) -> PatchBatch:
        """Draw one batch. ``batch_size`` must be a multiple of 4; ``generator`` seeds the draw
        (device generator)."""
        if batch_size % 4 != 0 or batch_size < 4:
            raise ValueError(f"batch_size must be a positive multiple of 4, got {batch_size}")
        p, n = self.cfg.patch_size, self.cfg.strokes_per_patch
        with torch.no_grad():
            # background canvases (2P x 2P), all strokes painted
            old_raw = self._draw((batch_size // 4, n), generator)
            fg, al = self._render(old_raw, 2 * p)
            canvas = torch.zeros(batch_size // 4, 3, 2 * p, 2 * p, device=self.device)
            canvas = pt_composite(canvas, fg, al)
            old = canvas.view(batch_size // 4, 3, 2, p, 2, p).permute(0, 2, 4, 1, 3, 5)
            old = old.reshape(batch_size, 3, p, p).contiguous()
            # foreground strokes with the validity rule
            gt_raw = self._draw((batch_size, n), generator)
            fg, al = self._render(gt_raw, p)
            decision = self._validity(al)
            target = pt_composite(old.clone(), fg, al, decision)
        return PatchBatch(
            canvas=old,
            target=target,
            gt_param=gt_raw[..., :SHAPE_PARAMS].contiguous(),
            gt_decision=decision,
            gt_color=gt_raw[..., 5:8].contiguous(),
        )

    def _validity(self, alphas: torch.Tensor) -> torch.Tensor:
        """``(B, N)`` 0/1 labels from ``(B, N, 3, P, P)`` alphas (sequential rule of the official
        code)."""
        b, n = alphas.shape[:2]
        area = alphas.sum(dim=(-3, -2, -1))  # (B, N)
        valid = torch.ones(b, n, device=alphas.device)
        for i in range(1, n):
            overlap = (alphas[:, i : i + 1] * alphas[:, :i]).sum(dim=(-3, -2, -1))  # (B, i)
            iou = (overlap + 1e-5) / (area[:, :i] + 1e-5)
            ok = (iou < self.cfg.iou_threshold) | (valid[:, :i] == 0)
            valid[:, i] = ok.all(dim=1).float()
        return valid
