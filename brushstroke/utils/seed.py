"""Global seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int, deterministic: bool = False, cudnn_deterministic: bool = False
) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices).

    Args:
        seed: non-negative integer seed.
        deterministic: if True, additionally force deterministic kernels
            (``torch.use_deterministic_algorithms(True)``, cuDNN deterministic, benchmark off);
            ops without a deterministic implementation then raise.
            Call it before any CUDA work: ``CUBLAS_WORKSPACE_CONFIG`` is only read at the first
            cuBLAS call (otherwise set it in the entry script). If False, these flags are
            reset to their permissive defaults.
        cudnn_deterministic: only force deterministic cuDNN kernels (convolution backward, e.g. the
            LPIPS/VGG term of the baseline loss) without ``use_deterministic_algorithms``. On the
            optimization baseline this alone makes GPU runs bitwise reproducible at full speed
            (T-007); the full flag is ~3x slower and refuses ``grid_sample``'s CUDA backward.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (global legacy RNG, seeded on purpose)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Only affects child processes: hash randomization of this process is fixed at startup.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic or cudnn_deterministic
    torch.backends.cudnn.benchmark = False
