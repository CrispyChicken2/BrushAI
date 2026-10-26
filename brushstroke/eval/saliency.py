"""Evaluation saliency maps (T-025): the InSPyReNet evaluator and the spectral-residual control.

Spec: ``docs/design/evaluation.md`` section 4 (saliency-weighted LPIPS, circularity rule). A map ``s`` is a float32
array ``(H, W)`` in [0, 1] of the **target** image, independent of any painting.

* ``spectral_residual``: Hou and Zhang, "Saliency Detection: A Spectral Residual Approach", CVPR 2007. Model-free:
  the image is reduced to 64 x 64 greyscale, the log amplitude spectrum minus its 3 x 3 local average is
  recombined with the original phase, and the squared magnitude of the inverse transform is smoothed by a Gaussian
  (sigma 8 at 64 x 64), then resized and min-max normalized.
* ``InSPyReNet``: Kim et al., "Revisiting Image Pyramid Structure for High Resolution Salient Object Detection",
  ACCV 2022, SwinB backbone trained on DUTS-TR only. The code (MIT) and the checkpoint are fetched at a pinned commit
  and a pinned SHA-256 (``configs/eval/saliency.yaml``) by ``scripts/fetch_inspyrenet.py``; nothing is committed and the
  checkpoint is never redistributed. **Reserved for evaluation: the contribution (M4) must not use it** (the
  circularity rule of section 4.2).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

CONFIG = Path("configs/eval/saliency.yaml")
ROOT = Path("data/raw/inspyrenet")
SR_SIZE = 64  # side of the image the spectral residual is computed on (Hou and Zhang)
SR_SIGMA = 8.0  # Gaussian smoothing of the saliency map at that size
SR_AVERAGE = 3  # side of the local-average filter of the log spectrum


def minmax(x: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant map becomes zeros (nothing is more salient than anything else)."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def spectral_residual(image: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency of an ``uint8`` RGB image ``(H, W, 3)``: float32 ``(H, W)`` in [0, 1].

    Deterministic (CPU, float64 FFT). A uniform image gives a zero map. As with every spectral-residual map, a lone
    object on a perfectly flat background or next to the border of the image can lose its peak to a ghost on the border
    (the transform is periodic): the map is a control, judged by its sign agreement with the evaluation model.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 (H, W, 3) image, got {image.dtype} {image.shape}")
    h, w = image.shape[:2]
    grey = torch.from_numpy(image.astype(np.float64)).mean(dim=2)[None, None]  # (1, 1, H, W), 0-255
    small = F.interpolate(grey, size=(SR_SIZE, SR_SIZE), mode="area")[0, 0].numpy()
    if float(np.ptp(small)) < 1e-6:  # a flat image has no spectrum to take a residual of
        return np.zeros((h, w), dtype=np.float32)
    spectrum = np.fft.fft2(small)
    log_amp = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amp - uniform_filter(log_amp, size=SR_AVERAGE, mode="reflect")
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = gaussian_filter(sal, sigma=SR_SIGMA, mode="reflect")
    big = F.interpolate(
        torch.from_numpy(sal)[None, None], size=(h, w), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    return minmax(big)


def map_sha256(saliency: np.ndarray) -> str:
    """SHA-256 of the float32 bytes of a map (C order)."""
    if saliency.dtype != np.float32:
        raise ValueError(f"maps are float32, got {saliency.dtype}")
    return hashlib.sha256(np.ascontiguousarray(saliency).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_spec(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def verify_files(root: Path = ROOT, spec: dict[str, Any] | None = None) -> list[str]:
    """The problems of the fetched code and checkpoint (missing file or SHA-256 different from the pinned one)."""
    spec = spec or load_spec()
    bad = []
    for rel, want in spec["code"]["files"].items():
        path = root / "code" / rel
        if not path.exists():
            bad.append(f"{rel}: missing")
        elif sha256_file(path) != want:
            bad.append(f"{rel}: SHA-256 differs from the pinned one")
    ckpt = root / spec["checkpoint"]["file"]
    if not ckpt.exists():
        bad.append(f"{ckpt.name}: missing")
    elif sha256_file(ckpt) != spec["checkpoint"]["sha256"]:
        bad.append(f"{ckpt.name}: SHA-256 differs from the pinned one")
    return bad


class InSPyReNetSaliency:
    """The InSPyReNet SwinB (DUTS-TR) evaluator on 512 px targets.

    Follows ``run/Inference.py`` of the pinned commit: a 512 px target is below the pyramid threshold, so the model sees the
    384 px bilinear resize (ImageNet mean and std), returns its own min-max normalized map, which is resized back with
    ``align_corners=True``. TF32 is off, as in the reference. The code and the checkpoint are verified against their pinned
    SHA-256 before anything is imported or loaded.
    """

    def __init__(self, device: str | torch.device = "cuda", root: Path = ROOT) -> None:
        spec = load_spec()
        problems = verify_files(root, spec)
        if problems:
            raise RuntimeError(
                "InSPyReNet files are not the pinned ones (run python -m scripts.fetch_inspyrenet):\n  "
                + "\n  ".join(problems)
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        code = str((root / "code").resolve())
        sys.path.insert(0, code)
        try:
            from lib import InSPyReNet_SwinB  # type: ignore[import-not-found]  # the pinned reference code
        finally:
            sys.path.remove(code)
        inf = spec["inference"]
        self.base_size = tuple(inf["base_size"])
        self.mean = torch.tensor(inf["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(inf["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.device = torch.device(device)
        model = InSPyReNet_SwinB(
            depth=64, pretrained=False, base_size=list(self.base_size), threshold=inf["threshold"]
        )
        state = torch.load(root / spec["checkpoint"]["file"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device)
        self.model.eval()

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)[None]
        return ((x - self.mean) / self.std).to(self.device)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Saliency of an ``uint8`` RGB image ``(H, W, 3)`` with sides of at most 512: float32 ``(H, W)`` in [0, 1]."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected a uint8 (H, W, 3) image, got {image.dtype} {image.shape}")
        h, w = image.shape[:2]
        if min(h, w) > 512:
            raise ValueError(
                "images above 512 px would use the high-resolution pyramid, which is not part of the protocol"
            )
        pil = Image.fromarray(image)
        sample = {
            "image": self._tensor(pil),
            "image_resized": self._tensor(
                pil.resize(self.base_size[::-1], Image.Resampling.BILINEAR)
            ),
        }
        pred = self.model(sample)["pred"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)
        return pred[0, 0].float().cpu().numpy().astype(np.float32)
