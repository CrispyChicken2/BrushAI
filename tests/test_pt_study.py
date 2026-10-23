"""Sanity checks of the pinned references of the Paint Transformer study tool (offline)."""

from __future__ import annotations

import re
from pathlib import Path

from scripts import pt_reference_study as study

ROOT = Path(__file__).parents[1]


def test_pinned_commit_and_checksums_are_well_formed() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", study.COMMIT)
    assert study.RAW.endswith(study.COMMIT)
    assert re.fullmatch(r"[0-9a-f]{64}", study.WEIGHTS_SHA256)
    assert all(re.fullmatch(r"[0-9a-f]{64}", h) for h in study.FILES.values())
    for rel in ("inference/inference.py", "train/models/painter_model.py", "LICENSE"):
        assert rel in study.FILES


def test_design_note_cites_the_same_commit_and_weights() -> None:
    note = (ROOT / "docs" / "design" / "paint_transformer.md").read_text(encoding="utf-8")
    assert study.COMMIT in note and study.WEIGHTS_SHA256 in note
    assert "9,072,902" in note and "Apache" in note


def test_official_lr_schedule_arithmetic_quoted_in_the_note() -> None:
    """Re-derive the schedule numbers of the note from the official lambda rule."""

    def multiplier(epoch: int, n_epochs: int = 200, epoch_count: int = 1) -> float:
        return 0.3 ** max(0, (epoch + epoch_count - n_epochs) // 5)

    # training epoch k (1-indexed) uses the multiplier of scheduler epoch k - 1
    assert multiplier(203) == 1.0 and multiplier(204) == 0.3
    assert abs(multiplier(219) - 0.0081) < 1e-12 and abs(1e-4 * multiplier(214) - 2.7e-6) < 1e-12
    assert (16384 // 64) * 220 == 56320 and 56320 * 64 == 3_604_480
