"""Tests of the Hugging Face staging folder (dry run) and of the training report helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from scripts.pt_training_report import PRE_DECAY, mean_std, window_gain
from scripts.upload_pt_to_hub import build_staging


def test_staging_folder_has_weights_only_card_and_config_hash(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.resolved.yaml").write_text("seed: 3\n", encoding="utf-8")
    state = {
        "step": 100,
        "model": {"w": torch.zeros(4, 2)},
        "optimizer": {"big": 1},
        "config": {"seed": 3},
    }
    torch.save(state, run / "latest.pt")
    out = build_staging(run / "latest.pt", tmp_path / "hub", "test", {"total": 1.5})
    assert set(torch.load(out / "pytorch_model.pth", weights_only=True)) == {"w"}
    digest = hashlib.sha256((run / "config.resolved.yaml").read_bytes()).hexdigest()
    assert (out / "training_config_sha256.txt").read_text().strip() == digest
    card = (out / "README.md").read_text(encoding="utf-8")
    assert (
        digest in card and "license: mit" in card and "total 1.500" in card and "seed\n  3" in card
    )


def test_plateau_criterion_and_mean_std() -> None:
    steps = list(range(0, 1001, 20))
    flat = [5.0 + 0.0001 * (1000 - s) / 1000 for s in steps]
    falling = [5.0 * (1 - 0.0005 * s) for s in steps]
    assert window_gain(steps, flat, PRE_DECAY) < 0.01 < window_gain(steps, falling, PRE_DECAY)
    # a median ignores the isolated evaluation spikes
    spiky = list(flat)
    spiky[40] = 9.0  # step 800, inside the first pre-decay window
    assert window_gain(steps, spiky, PRE_DECAY) < 0.01
    assert mean_std([1.0, 3.0]) == (2.0, 2**0.5)
    assert mean_std([4.0]) == (4.0, 0.0)
