"""pytest configuration: auto-skip tests marked ``gpu`` when CUDA is unavailable."""

from __future__ import annotations

import pytest
import torch


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
