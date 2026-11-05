"""Analytic rasterizer of the quadratic Bezier strokes of Learning to Paint (ground truth of the neural renderer).

Reimplements, on the GPU and for a batch, ``baseline/Renderer/stroke_gen.py::draw`` of the official code
(github.com/megvii-research/ICCV2019-LearningToPaint, commit 615c070, MIT): a stroke is 10 numbers in [0, 1]
``(x0, y0, x1, y1, x2, y2, z0, z2, w0, w2)`` (+ 3 colour values handled by the compositor, 13 in all):

* the control point ``(x1, y1)`` is relative: ``x1' = x0 + (x2 - x0) x1``, ``y1' = y0 + (y2 - y0) y1``;
* coordinates are quantized on a ``2 S`` canvas (``S`` = 128): ``int(v (2 S - 1) + 0.5)``; radii ``int(1 + z S // 2)``;
* 100 samples ``t = i / 100`` along the curve; each draws a filled disc of integer radius ``z(t) = int((1-t) z0 + t z2)`` centred
  at the truncated curve point, with value ``w(t) = (1 - t) w0 + t w2`` (**overwriting**, not blending: the last sample wins);
* the ``2 S`` canvas is resized to ``S`` (``cv2.resize``, linear = 2 x 2 mean) and the result is ``1 - canvas``.

``(x, y)`` of the official code are the row and column (``cv2.circle(canvas, (y, x), ...)``), kept as is.
:func:`stroke_coverage` returns the coverage ``canvas`` (``1 - draw``); :func:`draw_reference` is the official ``draw`` with cv2,
used by the tests and the equivalence check only.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

SAMPLES = 100


def _quantized(strokes: torch.Tensor, size: int) -> tuple[torch.Tensor, ...]:
    """Control points on the ``2 size`` canvas (int64 rows/cols), radii and values at the 100 samples: all ``(B, 100)``."""
    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = (strokes[:, i].double() for i in range(10))
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1
    big = 2 * size

    def norm(v: torch.Tensor) -> torch.Tensor:
        return torch.floor(v * (big - 1) + 0.5)

    x0, x1, x2, y0, y1, y2 = (norm(v) for v in (x0, x1, x2, y0, y1, y2))
    z0 = torch.floor(1 + torch.floor(z0 * size / 2))
    z2 = torch.floor(1 + torch.floor(z2 * size / 2))
    t = (torch.arange(SAMPLES, dtype=torch.float64, device=strokes.device) * (1.0 / 100))[None]

    def col(v: torch.Tensor) -> torch.Tensor:
        return v[:, None]

    xs = torch.trunc((1 - t) ** 2 * col(x0) + 2 * t * (1 - t) * col(x1) + t * t * col(x2))
    ys = torch.trunc((1 - t) ** 2 * col(y0) + 2 * t * (1 - t) * col(y1) + t * t * col(y2))
    zs = torch.trunc((1 - t) * col(z0) + t * col(z2))
    ws = (1 - t) * col(w0) + t * col(w2)
    return xs, ys, zs, ws


def stroke_coverage(strokes: torch.Tensor, size: int = 128, block: int = 32) -> torch.Tensor:
    """Coverage of ``(B, >=10)`` strokes: ``(B, size, size)`` float32 in [0, 1] (the official ``1 - draw(f)``)."""
    strokes = strokes[:, :10]
    big = 2 * size
    xs, ys, zs, ws = _quantized(strokes, size)
    grid = torch.arange(big, device=strokes.device, dtype=torch.float32)
    order = torch.arange(1, SAMPLES + 1, device=strokes.device, dtype=torch.int16)[
        None, :, None, None
    ]
    out = []
    for a in range(0, strokes.shape[0], block):
        sl = slice(a, a + block)
        dr = (grid[None, None, :] - xs[sl, :, None].float()) ** 2  # rows: (b, 100, big)
        dc = (grid[None, None, :] - ys[sl, :, None].float()) ** 2  # cols
        inside = (dr[:, :, :, None] + dc[:, :, None, :]) <= (zs[sl].float() ** 2)[:, :, None, None]
        last = (inside.to(torch.int16) * order).amax(
            dim=1
        )  # (b, big, big): 1 + index of the last covering sample, 0 = none
        w = F.pad(ws[sl].float(), (1, 0))  # index 0 -> value 0
        canvas = torch.gather(w, 1, last.long().flatten(1)).view(-1, big, big)
        out.append(F.avg_pool2d(canvas[:, None], 2)[:, 0])
    return torch.cat(out) if out else strokes.new_zeros(0, size, size)


def draw_reference(f: np.ndarray, width: int = 128) -> np.ndarray:
    """The official ``stroke_gen.draw`` (cv2), verbatim apart from formatting: ``1 - coverage``, ``(width, width)`` float32."""
    import cv2

    x0, y0, x1, y1, x2, y2, z0, z2, w0, w2 = f
    x1 = x0 + (x2 - x0) * x1
    y1 = y0 + (y2 - y0) * y1

    def normal(x: float, w: int) -> int:
        return int(x * (w - 1) + 0.5)

    x0, x1, x2 = (normal(v, width * 2) for v in (x0, x1, x2))
    y0, y1, y2 = (normal(v, width * 2) for v in (y0, y1, y2))
    z0 = int(1 + z0 * width // 2)
    z2 = int(1 + z2 * width // 2)
    canvas = np.zeros([width * 2, width * 2]).astype("float32")
    tmp = 1.0 / 100
    for i in range(100):
        t = i * tmp
        x = int((1 - t) * (1 - t) * x0 + 2 * t * (1 - t) * x1 + t * t * x2)
        y = int((1 - t) * (1 - t) * y0 + 2 * t * (1 - t) * y1 + t * t * y2)
        z = int((1 - t) * z0 + t * z2)
        w = (1 - t) * w0 + t * w2
        cv2.circle(canvas, (y, x), z, w, -1)
    return 1 - cv2.resize(canvas, dsize=(width, width))
