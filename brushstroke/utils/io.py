"""Image I/O as float tensors.

Convention used everywhere in the project: an image is a ``float32`` tensor of shape ``(3, H, W)``,
RGB, values in ``[0, 1]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image(path: str | Path, size: int | tuple[int, int] | None = None) -> torch.Tensor:
    """Load an image file as a ``(3, H, W)`` float32 tensor in [0, 1].

    Alpha is dropped and grayscale is expanded to RGB.

    Args:
        path: image file.
        size: optional ``(H, W)`` or a single int for a square output (bicubic, antialiased).
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return resize_image(img, size) if size is not None else img


def save_image(img: torch.Tensor, path: str | Path) -> None:
    """Save a ``(3, H, W)`` tensor as 8-bit RGB (clamped to [0, 1]); creates parent dirs."""
    _check_image(img)
    arr = img.detach().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="RGB").save(out)


def resize_image(img: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Resize a ``(3, H, W)`` tensor to ``size`` (``(H, W)`` or int); bicubic, antialiased."""
    _check_image(img)
    hw = (size, size) if isinstance(size, int) else size
    out = F.interpolate(img[None], size=hw, mode="bicubic", align_corners=False, antialias=True)[0]
    return out.clamp(0.0, 1.0)


def resize_short_side(img: torch.Tensor, short: int) -> torch.Tensor:
    """Resize keeping the aspect ratio so that the shorter side is ``short`` pixels."""
    _check_image(img)
    h, w = img.shape[-2:]
    scale = short / min(h, w)
    return resize_image(img, (max(1, round(h * scale)), max(1, round(w * scale))))


def _check_image(img: torch.Tensor) -> None:
    if img.ndim != 3 or img.shape[0] != 3:
        raise ValueError(f"expected an image of shape (3, H, W), got {tuple(img.shape)}")
