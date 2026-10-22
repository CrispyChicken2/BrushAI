"""Tests for the StrokeSequence data model and its serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from brushstroke.renderer import BrushBank, RenderConfig, render_progressive
from brushstroke.strokes import SCHEMA_VERSION, BrushSpec, StrokeSequence

SPEC = BrushSpec(kind="oil", bank_size=4, mask_height=32, mask_width=64, seed=3)


def make_seq(n: int = 20, seed: int = 0, size: tuple[int, int] = (48, 64)) -> StrokeSequence:
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, 9, generator=g)
    p[:, 2:4] = 0.05 + 0.3 * p[:, 2:4]
    return StrokeSequence(
        params=p,
        canvas_size=size,
        brush=SPEC,
        background=(0.9, 0.8, 0.7),
        background_ref="paper.png",
        scale_level=torch.arange(n) % 3,
        patch_id=torch.arange(n) * 2,
        source=[f"prod{i % 2}" for i in range(n)],
        meta={"seed": seed, "config_hash": "abc123", "timings": {"opt": 1.5}, "tags": ["a", "b"]},
    )


def test_construction_defaults_and_validation() -> None:
    seq = StrokeSequence(torch.rand(5, 9), canvas_size=(32, 32))
    assert len(seq) == 5 and seq.params.dtype == torch.float32
    assert seq.schema_version == SCHEMA_VERSION
    assert seq.brush_index is not None and seq.brush_index.shape == (5,)
    assert seq.scale_level is not None and (seq.scale_level == -1).all()
    assert seq.patch_id is not None and (seq.patch_id == -1).all()
    assert seq.source == [""] * 5
    assert len(StrokeSequence.empty((16, 16))) == 0
    with pytest.raises(ValueError):
        StrokeSequence(torch.rand(5, 8))
    with pytest.raises(ValueError):
        StrokeSequence(torch.rand(5, 9), scale_level=torch.zeros(4))
    with pytest.raises(ValueError):
        StrokeSequence(torch.rand(5, 9), source=["a"])
    with pytest.raises(ValueError):
        StrokeSequence(torch.rand(5, 9), brush_index=torch.full((5,), 99))
    with pytest.raises(ValueError):
        StrokeSequence(torch.rand(5, 9), canvas_size=(0, 8))
    with pytest.raises(TypeError):
        StrokeSequence(torch.rand(5, 9), meta={"x": object()})


def test_float64_params_are_stored_as_float32() -> None:
    assert StrokeSequence(torch.rand(3, 9, dtype=torch.float64)).params.dtype == torch.float32


def test_default_brush_index_is_prefix_stable() -> None:
    a = StrokeSequence(torch.rand(10, 9))
    b = StrokeSequence(torch.rand(25, 9))
    assert a.brush_index is not None and b.brush_index is not None
    assert torch.equal(a.brush_index, b.brush_index[:10])


# ------------------------------------------------------------------ serialization


def test_save_load_roundtrip_bitwise(tmp_path: Path) -> None:
    seq = make_seq()
    path = seq.save(tmp_path / "sub" / "seq")
    assert path.suffix == ".npz" and path.exists()
    back = StrokeSequence.load(path)
    assert torch.equal(back.params, seq.params) and back.params.dtype == torch.float32
    assert back.params.numpy().tobytes() == seq.params.numpy().tobytes()
    for name in ("brush_index", "scale_level", "patch_id"):
        assert torch.equal(getattr(back, name), getattr(seq, name))
    assert back.source == seq.source and back.meta == seq.meta
    assert back.brush == seq.brush and back.background == seq.background
    assert back.canvas_size == seq.canvas_size and back.background_ref == "paper.png"
    assert back.equals(seq)
    assert torch.equal(back.render(), seq.render())


def test_roundtrip_special_values_and_empty(tmp_path: Path) -> None:
    p = torch.rand(3, 9)
    p[0, 0] = float(np.float32(1e-38))
    seq = StrokeSequence(p, meta={"x": 0.1 + 0.2, "t": (1, 2), "n": None})
    assert seq.meta["t"] == [1, 2]  # tuples are normalized by the JSON round trip
    back = StrokeSequence.load(seq.save(tmp_path / "a.npz"))
    assert back.equals(seq) and back.meta["x"] == 0.1 + 0.2
    empty = StrokeSequence.empty((8, 8))
    assert StrokeSequence.load(empty.save(tmp_path / "e.npz")).equals(empty)


def test_load_rejects_future_schema_and_foreign_files(tmp_path: Path) -> None:
    path = make_seq().save(tmp_path / "s.npz")
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    header = json.loads(bytes(arrays["header"].tobytes()).decode())
    header["schema_version"] = SCHEMA_VERSION + 1
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    np.savez(tmp_path / "future.npz", **arrays)
    with pytest.raises(ValueError, match="schema_version"):
        StrokeSequence.load(tmp_path / "future.npz")
    np.savez(tmp_path / "other.npz", foo=np.zeros(3))
    with pytest.raises(ValueError, match="not a StrokeSequence"):
        StrokeSequence.load(tmp_path / "other.npz")
    header["schema_version"] = 0  # no migration registered
    arrays["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    np.savez(tmp_path / "old.npz", **arrays)
    with pytest.raises(ValueError, match="migration"):
        StrokeSequence.load(tmp_path / "old.npz")


# ------------------------------------------------------------------ operations


def test_slicing_keeps_tags() -> None:
    seq = make_seq(10)
    assert seq.scale_level is not None and seq.patch_id is not None and seq.brush_index is not None
    head = seq[:4]
    assert len(head) == 4 and head.meta == seq.meta and head.brush == seq.brush
    assert torch.equal(head.params, seq.params[:4])
    assert head.source == seq.source[:4]  # type: ignore[index]
    assert head.scale_level is not None and torch.equal(head.scale_level, seq.scale_level[:4])
    assert head.patch_id is not None and torch.equal(head.patch_id, seq.patch_id[:4])
    assert head.brush_index is not None and torch.equal(head.brush_index, seq.brush_index[:4])
    assert len(seq[3:3]) == 0 and len(seq[::2]) == 5
    picked = seq[[7, 1]]
    assert torch.equal(picked.params, seq.params[[7, 1]])
    assert picked.source == [seq.source[7], seq.source[1]]  # type: ignore[index]
    assert len(seq[torch.arange(10) % 2 == 0]) == 5
    with pytest.raises(IndexError):
        seq[[10]]
    with pytest.raises(ValueError):
        seq[torch.ones(3, dtype=torch.bool)]


def test_reorder_identity_inverse_and_invalid() -> None:
    seq = make_seq(15)
    assert seq.reorder(list(range(15))).equals(seq)
    g = torch.Generator().manual_seed(4)
    perm = torch.randperm(15, generator=g)
    shuffled = seq.reorder(perm)
    assert not shuffled.equals(seq)
    assert torch.equal(shuffled.params, seq.params[perm])
    assert shuffled.source == [seq.source[i] for i in perm]  # type: ignore[index]
    inverse = torch.argsort(perm)
    assert shuffled.reorder(inverse).equals(seq)
    for bad in ([0, 0, *range(2, 15)], list(range(14)), list(range(1, 16)), [-1, *range(1, 15)]):
        with pytest.raises(ValueError):
            seq.reorder(bad)
    with pytest.raises(ValueError):
        seq.reorder(torch.zeros(15))  # float tensor
    with pytest.raises(ValueError):
        seq.reorder(torch.zeros(15, 1, dtype=torch.long))


def test_concat() -> None:
    a, b = make_seq(6, seed=1), make_seq(4, seed=2)
    c = StrokeSequence.concat([a, b])
    assert len(c) == 10 and torch.equal(c.params, torch.cat([a.params, b.params]))
    assert c.source == a.source + b.source and c.meta == a.meta  # type: ignore[operator]
    assert a.patch_id is not None and b.patch_id is not None and c.patch_id is not None
    assert torch.equal(c.patch_id, torch.cat([a.patch_id, b.patch_id]))
    assert c[:6].equals(a) and StrokeSequence.concat([a]).equals(a)
    with pytest.raises(ValueError):
        StrokeSequence.concat([])
    with pytest.raises(ValueError):
        StrokeSequence.concat([a, make_seq(3, size=(10, 10))])


def test_to_device_roundtrip() -> None:
    seq = make_seq(5)
    assert seq.to("cpu").equals(seq)


@pytest.mark.gpu
def test_to_cuda_and_render_matches_cpu() -> None:
    seq = make_seq(12)
    g = seq.to("cuda")
    assert g.params.is_cuda
    assert g.brush_index is not None and g.brush_index.is_cuda
    assert g.scale_level is not None and g.scale_level.is_cuda
    assert torch.allclose(g.render().cpu(), seq.render(), atol=1e-4)
    assert g.to("cpu").equals(seq)


# ------------------------------------------------------------------ rendering


def test_prefix_render_equals_progressive_checkpoint() -> None:
    seq = make_seq(17)
    assert seq.brush_index is not None
    cps = [0, 1, 5, 17]
    bank = BrushBank.build(
        SPEC.kind, SPEC.bank_size, (SPEC.mask_height, SPEC.mask_width), SPEC.seed
    )
    prog = render_progressive(
        seq.params[None],
        bank,
        48,
        64,
        cps,
        canvas=seq.background,
        cfg=RenderConfig(chunk_size=4),
        brush_index=seq.brush_index[None],
    )[0]
    for i, k in enumerate(cps):
        assert torch.allclose(seq[:k].render(cfg=RenderConfig(chunk_size=4)), prog[i], atol=1e-6), k


def test_render_defaults_and_validation() -> None:
    seq = make_seq(6)
    img = seq.render()
    assert img.shape == (3, 48, 64) and img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    assert seq.render(30, 20).shape == (3, 30, 20)
    flat = StrokeSequence.empty((4, 4), background=(0.1, 0.2, 0.3)).render()[:, 0, 0]
    assert torch.allclose(flat, torch.tensor([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        seq.render(height=32)
    custom = torch.full((3, 48, 64), 0.5)
    assert torch.equal(seq[:0].render(canvas=custom), custom)


def test_render_at_double_resolution_matches_downsampled() -> None:
    """The 2x render, box-downsampled to 1x, stays close to the 1x render.

    Differences come from bilinear sampling of the brush mask on another pixel grid and from
    antialiasing at stroke edges, so they shrink with resolution. Measured (40 strokes, seeds
    0-4) at 128x192: mean error 2.6e-3..3.4e-3, 99th percentile 0.03..0.05, max 0.13..0.28; at
    256x384 the mean is ~8e-4. Documented tolerance: mean < 6e-3 and 99th percentile < 8e-2 (the
    maximum is not bounded tightly: a one-pixel shift of a high-contrast edge can be large).
    """
    seq = make_seq(40, seed=5, size=(128, 192))
    one = seq.render()
    two = seq.render(256, 384)
    down = F.avg_pool2d(two[None], 2)[0]
    assert down.shape == one.shape
    err = (down - one).abs()
    assert float(err.mean()) < 6e-3
    assert float(torch.quantile(err.flatten(), 0.99)) < 8e-2
