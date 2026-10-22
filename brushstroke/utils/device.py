"""Device selection and GPU memory helpers."""

from __future__ import annotations

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Return the torch device.

    Args:
        prefer: ``"auto"`` (CUDA if available, else CPU), ``"cuda"`` or ``"cpu"``.

    Raises:
        RuntimeError: ``"cuda"`` requested but unavailable.
        ValueError: unknown ``prefer`` value.
    """
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device preference: {prefer!r}")


def reset_peak_vram() -> None:
    """Reset the CUDA peak-memory counter (no-op without CUDA)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mb() -> float:
    """Peak allocated CUDA memory in MiB since :func:`reset_peak_vram` (0.0 without CUDA)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)
