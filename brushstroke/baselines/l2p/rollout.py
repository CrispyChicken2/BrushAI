"""Inference rollout of Learning to Paint with progressive canvases (official ``baseline/test.py``, commit 615c070, MIT).

For an image larger than 128 x 128 the official procedure (``--divide d``, here ``d = 4`` for a 512 x 512 image) is:

1. *Global stage*: the image is resized to 128 x 128 and the actor paints ``max_step // 2`` steps of one bundle of 5 strokes each
   on a black canvas (the step input is ``i / (max_step // 2)``);
2. *Divided stage*: the canvas is resized to ``128 d`` and cut into ``d x d`` patches of 128 x 128 (row-major), the target is
   resized and cut the same way, and the actor paints ``max_step // 2`` more steps on all patches (a batch), one bundle per patch
   and step; the patches are stitched back and the seams are smoothed (``smooth``, a 3 x 3 mean of the pixels on the borders).

Both stages are painted with the neural renderer (:func:`brushstroke.baselines.l2p.renderer.decode`). Every output is resized to
the original size with ``cv2.resize`` (linear). The channel order is whatever the caller passes (the official code feeds OpenCV's
BGR images to a network trained on OpenCV-read CelebA, so the producer flips RGB to BGR and back).

**Stroke order and counts** (evaluation protocol, section 3.3): strokes are counted for every patch; the emission order is
global-stage steps, then divided-stage step by step, within a step patch by patch (row-major), within a patch the 5 bundle strokes
in order. The canvas after ``k`` strokes is exact: the patches are independent, so a stroke of patch ``p`` is applied on the state
the actor saw (the canvas of the start of the step) with the earlier strokes of the same bundle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn

from brushstroke.baselines.l2p.actor import coord_channels
from brushstroke.baselines.l2p.renderer import decode

WIDTH = 128


@dataclass
class RolloutConfig:
    """Inference settings (to move to YAML when T-030 registers the producer).

    ``max_step = 100`` is the command documented in the official README (``test.py --max_step=100 --divide=4``): 50 steps per stage,
    ``50 x 5 + 16 x 50 x 5 = 4,250`` strokes at ``divide = 4``, so the 2,000-stroke budget of the protocol is reachable. The
    ``test.py`` default of 40 gives only 1,700 strokes (``docs/design/learning_to_paint.md`` section 2).
    """

    max_step: int = 100  # steps of the episode; halved per stage when the image is divided
    divide: int = 4  # patches per side of the divided stage (1 = no divided stage)
    bundle: int = 5  # strokes per step (the actor outputs 5 x 13 numbers)
    smooth: bool = True  # official seam smoothing of the stitched canvas


def large2small(x: np.ndarray, d: int) -> np.ndarray:
    """``(d W, d W, C)`` -> ``(d * d, W, W, C)`` patches in row-major order (official ``large2small``)."""
    c = x.shape[-1]
    x = x.reshape(d, WIDTH, d, WIDTH, c)
    return np.transpose(x, (0, 2, 1, 3, 4)).reshape(d * d, WIDTH, WIDTH, c)


def small2large(x: np.ndarray, d: int) -> np.ndarray:
    """``(d * d, W, W, C)`` -> ``(d W, d W, C)`` (official ``small2large``)."""
    x = x.reshape(d, d, WIDTH, WIDTH, -1)
    return np.transpose(x, (0, 2, 1, 3, 4)).reshape(d * WIDTH, d * WIDTH, -1)


def smooth_seams(img: np.ndarray, d: int) -> np.ndarray:
    """The official ``smooth``: in place 3 x 3 mean of the pixels on the right and bottom border of every patch and the first
    pixel of the next (order matters: later pixels read the already smoothed ones, as in the official loop)."""
    n = d * WIDTH

    def pix(tx: int, ty: int) -> None:
        if tx in (0, n - 1) or ty in (0, n - 1):
            return
        img[tx, ty] = (
            img[tx, ty]
            + img[tx + 1, ty]
            + img[tx, ty + 1]
            + img[tx - 1, ty]
            + img[tx, ty - 1]
            + img[tx + 1, ty - 1]
            + img[tx - 1, ty + 1]
            + img[tx - 1, ty - 1]
            + img[tx + 1, ty + 1]
        ) / 9

    for p in range(d):
        for q in range(d):
            x, y = p * WIDTH, q * WIDTH
            for k in range(WIDTH):
                pix(x + k, y + WIDTH - 1)
                if q != d - 1:
                    pix(x + k, y + WIDTH)
            for k in range(WIDTH):
                pix(x + WIDTH - 1, y + k)
                if p != d - 1:
                    pix(x + WIDTH, y + k)
    return img


def _resize(x: torch.Tensor, size: int) -> torch.Tensor:
    """``(3, H, W)`` tensor -> ``(3, size, size)`` with ``cv2.resize`` (linear), as the official code."""
    arr = x.permute(1, 2, 0).detach().cpu().numpy()
    out = cv2.resize(arr, (size, size))
    return torch.from_numpy(out).permute(2, 0, 1).to(x.device, torch.float32)


class Painting:
    """The recorded rollout: the strokes and every intermediate state, from which any prefix canvas is assembled."""

    def __init__(
        self,
        cfg: RolloutConfig,
        out_size: tuple[int, int],
        stage1: list[torch.Tensor],
        stage1_strokes: torch.Tensor,
        before: list[torch.Tensor],
        after: list[torch.Tensor],
        stage2_strokes: torch.Tensor,
    ) -> None:
        self.cfg, self.out_size = cfg, out_size
        self.stage1 = stage1  # canvases (3, 128, 128) after 0, 1, ... strokes of the global stage (all kept on the CPU)
        self.before = before  # per divided step: (d * d, 3, 128, 128) at the start of the step
        self.after = (
            after  # per divided step: (d * d, bundle, 3, 128, 128) after each stroke of the bundle
        )
        self.strokes = torch.cat([stage1_strokes, stage2_strokes])  # (n_total, 13), emission order
        self.n_stage1 = stage1_strokes.shape[0]

    @property
    def n_total(self) -> int:
        return int(self.strokes.shape[0])

    def canvas(self, k: int) -> torch.Tensor:
        """The ``(3, H, W)`` canvas after the first ``k`` strokes (float32, clamped to [0, 1])."""
        if not 0 <= k <= self.n_total:
            raise ValueError(f"k must lie in [0, {self.n_total}], got {k}")
        h, w = self.out_size
        if k <= self.n_stage1:
            arr = self.stage1[k].permute(1, 2, 0).cpu().numpy()
            out = cv2.resize(arr, (w, h))
            return torch.from_numpy(out).permute(2, 0, 1).clamp(0, 1)
        d, bundle = self.cfg.divide, self.cfg.bundle
        per_step = d * d * bundle
        s, r = divmod(
            k - self.n_stage1 - 1, per_step
        )  # the last applied stroke: step s, patch p, bundle index j
        p, j = divmod(r, bundle)
        patches = self.before[s].clone()
        patches[:p] = self.after[s][:p, bundle - 1]
        patches[p] = self.after[s][p, j]
        big = small2large(patches.permute(0, 2, 3, 1).cpu().numpy(), d)
        if self.cfg.smooth:
            big = smooth_seams(big, d)
        out = cv2.resize(big, (w, h))
        return torch.from_numpy(out).permute(2, 0, 1).clamp(0, 1)


@torch.no_grad()
def rollout(
    actor: nn.Module,
    renderer: nn.Module,
    target: torch.Tensor,
    cfg: RolloutConfig | None = None,
) -> Painting:
    """Run the official inference on ``target`` ``(3, H, W)`` in [0, 1] (channel order of the actor) and record everything."""
    cfg = cfg or RolloutConfig()
    dev = target.device
    out_size = (int(target.shape[-2]), int(target.shape[-1]))
    d, bundle = cfg.divide, cfg.bundle
    steps = cfg.max_step // 2 if d != 1 else cfg.max_step
    coord = coord_channels(WIDTH, dev)
    img = _resize(target, WIDTH)[None]
    canvas = torch.zeros(1, 3, WIDTH, WIDTH, device=dev)
    stage1 = [canvas[0].cpu()]
    strokes1 = []
    for i in range(steps):
        step = torch.full((1, 1, WIDTH, WIDTH), i / steps, device=dev)
        actions = actor(torch.cat([canvas, img, step, coord], 1))
        canvas, res = decode(actions, canvas, renderer, bundle)
        stage1.extend(c[0].cpu() for c in res)
        strokes1.append(actions.reshape(bundle, 13))
    before: list[torch.Tensor] = []
    after: list[torch.Tensor] = []
    strokes2 = []
    if d != 1:
        big = _resize(canvas[0], WIDTH * d)
        canv = (
            torch.from_numpy(large2small(big.permute(1, 2, 0).cpu().numpy(), d))
            .permute(0, 3, 1, 2)
            .to(dev)
        )
        tgt_big = cv2.resize(target.permute(1, 2, 0).cpu().numpy(), (WIDTH * d, WIDTH * d))
        patch_img = torch.from_numpy(large2small(tgt_big, d)).permute(0, 3, 1, 2).to(dev).float()
        coord_p = coord.expand(d * d, 2, WIDTH, WIDTH)
        for i in range(steps):
            step = torch.full((d * d, 1, WIDTH, WIDTH), i / steps, device=dev)
            actions = actor(torch.cat([canv, patch_img, step, coord_p], 1))
            before.append(canv.cpu())
            canv, res = decode(actions, canv, renderer, bundle)
            after.append(torch.stack(res, 1).cpu())
            strokes2.append(actions.reshape(d * d * bundle, 13))
    empty = torch.zeros(0, 13, device=dev)
    return Painting(
        cfg,
        out_size,
        stage1,
        torch.cat(strokes1) if strokes1 else empty,
        before,
        after,
        torch.cat(strokes2) if strokes2 else empty,
    )


def progressive_canvases(
    painting: Painting, checkpoints: Sequence[int]
) -> tuple[list[int], torch.Tensor]:
    """Canvases after ``k`` strokes for every reachable ``k`` in ``checkpoints``: ``(len(cps), 3, H, W)``."""
    cps = [int(c) for c in checkpoints if c <= painting.n_total]
    return cps, torch.stack([painting.canvas(k) for k in cps]) if cps else torch.zeros(0)
