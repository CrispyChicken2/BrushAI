"""Tests of the Paint Transformer multi-scale inference, adapter and refinement (T-015)."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from brushstroke.renderer.pt import load_pt_brushes, patch_to_canvas, pt_param2stroke
from brushstroke.stroke_model import (
    LayerStrokes,
    PaintTransformer,
    PTInferenceConfig,
    PTModelConfig,
    RefineConfig,
    layers_to_sequence,
    load_pt_model,
    paint,
    paint_layer,
    refine_patch_strokes,
    replay_sequence,
)
from brushstroke.stroke_model.inference import (
    crop_square,
    extract_patches,
    num_layers,
    pad_square,
    to_window_units,
)
from brushstroke.stroke_model.pt_sequence import PHASES, SOURCE_FORCED, window_geometry
from brushstroke.strokes import StrokeSequence
from scripts import paint_pt
from scripts.pt_reference_study import official_paint
from tests import pt_official as po

needs_brushes = pytest.mark.skipif(
    not po.have_brushes(),
    reason="PT brush assets missing: run `python -m scripts.fetch_pt_brushes`",
)
needs_official = pytest.mark.skipif(
    not (po.have_official() and po.have_weights() and po.have_brushes()),
    reason="official code/weights missing: run `python -m scripts.pt_reference_study fetch`",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # as the official script


class ToyPredictor:
    """Deterministic stand-in for the network: plausible random strokes, colour read from the
    target at the centre. ``negative_h`` is the probability of a negative height; ``absolute``
    returns ``|h|`` instead (same draws), so that ``abs_h`` can be compared with it."""

    def __init__(self, seed: int = 0, negative_h: float = 0.0, absolute: bool = False) -> None:
        self.seed, self.negative_h, self.absolute = seed, negative_h, absolute

    def strokes(
        self, img: torch.Tensor, canvas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = img.shape[0]
        g = torch.Generator().manual_seed(self.seed + 7919 * b)
        shape = torch.rand(b, 8, 5, generator=g)
        shape[..., :2] = 0.15 + 0.7 * shape[..., :2]
        shape[..., 2:4] = 0.3 + 0.7 * shape[..., 2:4]
        sign = torch.where(torch.rand(b, 8, generator=g) < self.negative_h, -1.0, 1.0)
        shape[..., 3] = shape[..., 3] * (1.0 if self.absolute else sign)
        if self.absolute:
            shape[..., 3] = shape[..., 3].abs()
        logits = 0.8 + torch.randn(b, 8, generator=g)
        shape, logits = shape.to(img.device), logits.to(img.device)
        return shape, PaintTransformer.sample_colour(img, shape), logits


def smooth_image(height: int, width: int, seed: int = 0) -> torch.Tensor:
    """Deterministic smooth RGB test image ``(3, H, W)`` in [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(1, 3, 6, 8, generator=g)
    return F.interpolate(low, (height, width), mode="bicubic", align_corners=False)[0].clamp(0, 1)


@pytest.fixture(scope="module")
def brushes() -> torch.Tensor:
    return load_pt_brushes("inference", po.BRUSHES, "cpu")


# ------------------------------------------------------------------ geometry helpers


def test_number_of_layers_follows_the_official_formula() -> None:
    assert [num_layers(s, s) for s in (16, 32, 33, 64, 512, 513, 1024)] == [0, 0, 1, 1, 4, 5, 5]
    assert num_layers(300, 512) == 4 and num_layers(512, 100) == 4


def test_pad_and_crop_are_inverse_and_centre_like_the_official_code() -> None:
    img = torch.rand(1, 3, 5, 8)
    padded = pad_square(img, 12)
    assert padded.shape == (1, 3, 12, 12)
    # (12 - 5) // 2 = 3 rows above, 4 below; (12 - 8) // 2 = 2 columns left and right
    assert torch.equal(padded[..., 3:8, 2:10], img)
    assert torch.count_nonzero(padded) == torch.count_nonzero(img)  # zeros elsewhere
    assert torch.equal(crop_square(padded, 5, 8), img)
    with pytest.raises(ValueError):
        pad_square(img, 6)


def test_patches_are_in_raster_order() -> None:
    x = torch.arange(2 * 64 * 96, dtype=torch.float32).view(1, 2, 64, 96).repeat(1, 3, 1, 1)[:, :3]
    patches = extract_patches(x, 32)
    assert patches.shape == (6, 3, 32, 32)
    assert torch.equal(patches[0], x[0, :, :32, :32]) and torch.equal(
        patches[1], x[0, :, :32, 32:64]
    )
    assert torch.equal(patches[3], x[0, :, 32:, :32]) and torch.equal(patches[5], x[0, :, 32:, 64:])


# ------------------------------------------------------------------ window painter


@needs_brushes
@pytest.mark.parametrize("grid", [1, 2, 3, 4])
def test_paint_layer_equals_a_naive_window_loop(grid: int, brushes: torch.Tensor) -> None:
    """Gather/scatter implementation against explicit loops over the windows (odd grids too)."""
    torch.manual_seed(grid)
    size, n_stroke = 32 * grid, 3
    canvas = torch.rand(1, 3, size, size)
    params = torch.rand(1, grid, grid, n_stroke, 8)
    params[..., 2:4] = 0.2 + 0.5 * params[..., 2:4]
    decision = torch.rand(1, grid, grid, n_stroke) < 0.7
    out = paint_layer(canvas, params, decision, brushes)

    patch, window = size // grid, 2 * (size // grid)
    expected = F.pad(canvas, [patch // 2] * 4).clone()
    for phase_y, phase_x in PHASES:
        for i in range(phase_y, grid, 2):
            for j in range(phase_x, grid, 2):
                y0, x0 = patch * i, patch * j
                win = expected[:, :, y0 : y0 + window, x0 : x0 + window].clone()
                for s in range(n_stroke):
                    if not bool(decision[0, i, j, s]):
                        continue
                    fg, al = pt_param2stroke(params[0, i, j, s][None], window, window, brushes)
                    win = fg * al + win * (1 - al)
                expected[:, :, y0 : y0 + window, x0 : x0 + window] = win
    expected = expected[:, :, patch // 2 : -(patch // 2), patch // 2 : -(patch // 2)]
    assert torch.allclose(out, expected, atol=1e-6)
    assert not torch.equal(out, canvas)


@needs_brushes
def test_paint_layer_without_decision_is_the_identity(brushes: torch.Tensor) -> None:
    canvas = torch.rand(1, 3, 64, 64)
    params = torch.rand(1, 2, 2, 8, 8) * 0.5 + 0.2
    out = paint_layer(canvas, params, torch.zeros(1, 2, 2, 8, dtype=torch.bool), brushes)
    assert torch.equal(out, canvas)


# ------------------------------------------------------------------ adapter


def _toy_layer(level: int, grid: int, canvas_size: int, shifted: bool, seed: int) -> LayerStrokes:
    g = torch.Generator().manual_seed(seed)
    params = torch.rand(grid, grid, 2, 8, generator=g)
    params[..., 2:4] = 0.2 + 0.5 * params[..., 2:4]
    logits = torch.randn(grid, grid, 2, generator=g)
    painted = logits != 0 if shifted else logits > 0
    return LayerStrokes(level, grid, canvas_size, shifted, params, logits, painted, logits > 0)


def test_sequence_order_tags_and_coordinates() -> None:
    """(layer, phase, stroke index, patch raster order), independent explicit loops."""
    layers = [_toy_layer(0, 1, 32, False, 0), _toy_layer(1, 3, 96, False, 1)]
    layers.append(_toy_layer(2, 3, 96, True, 2))
    seq, dropped = layers_to_sequence(layers, (96, 96), 96, drop_invisible=False)
    assert dropped == 0
    expected: list[tuple[int, int, int]] = []  # (level, patch id, stroke)
    for layer in layers:
        for py, px in PHASES:
            for s in range(2):
                for i in range(py, layer.grid, 2):
                    for j in range(px, layer.grid, 2):
                        keep = layer.painted[i, j, s]
                        if keep:
                            expected.append((layer.level, i * layer.grid + j, s))
    assert len(seq) == len(expected)
    assert seq.scale_level is not None and seq.patch_id is not None
    got = list(zip(seq.scale_level.tolist(), seq.patch_id.tolist(), strict=True))
    assert got == [(a, b) for a, b, _ in expected]
    # coordinates: x = (32 col + 32 x_p [- 16 if shifted]) / canvas size, sizes 32 w_p / canvas size
    for k, (level, pid, s) in enumerate(expected):
        layer = layers[level]
        row, col = divmod(pid, layer.grid)
        p = layer.params[row, col, s]
        off = -16.0 if layer.shifted else 0.0
        want_x = (32 * col + off + 32 * float(p[0])) / layer.canvas_size
        want_h = 32 * float(p[3]) / layer.canvas_size
        assert float(seq.params[k, 0]) == pytest.approx(want_x, abs=1e-6)
        assert float(seq.params[k, 3]) == pytest.approx(want_h, abs=1e-6)
        assert float(seq.params[k, 8]) == 1.0
    # forced strokes: only the shifted layer paints strokes with a non positive logit
    forced = [i for i, src in enumerate(seq.source or []) if src == SOURCE_FORCED]
    assert all(seq.scale_level[i] == 2 for i in forced)
    shifted = layers[2]
    assert len(forced) == int((shifted.painted & ~shifted.confident).sum()) > 0
    gated, _ = layers_to_sequence(layers, (96, 96), 96, variant="gated", drop_invisible=False)
    assert len(gated) == len(seq) - len(forced)
    with pytest.raises(ValueError):
        layers_to_sequence(layers, (96, 96), 96, variant="nope")


def test_sequence_maps_padded_square_to_image_fractions_and_drops_invisible_strokes() -> None:
    # image 32 x 96 padded to a 96 square: 32 rows of padding above and below
    layer = _toy_layer(1, 3, 96, False, 3)
    layer.painted[:] = True
    seq, dropped = layers_to_sequence([layer], (32, 96), 96)
    full, none_dropped = layers_to_sequence([layer], (32, 96), 96, drop_invisible=False)
    assert none_dropped == 0 and len(full) == 18 and len(seq) + dropped == 18 and dropped > 0
    # a kept stroke: y_image = (y_square * 96 - 32) / 32
    first = full.params[0]
    p = layer.params[0, 0, 0]
    y_square = float(p[1]) * 32 / 96
    assert float(first[1]) == pytest.approx((y_square * 96 - 32) / 32, abs=1e-5)
    assert float(first[3]) == pytest.approx(float(p[3]) * 32 / 96 * 96 / 32, abs=1e-5)


def test_window_geometry_matches_the_pipeline_windows() -> None:
    level = torch.tensor([0, 2, 2, 5])  # K + 1 = 5 layers 0..4, then the shifted one (level 5)
    patch_id = torch.tensor([0, 5, 15, 17 * 3 + 4])
    rect, side = window_geometry(level, patch_id, 512, 5, 32)
    assert side.tolist() == [1024, 256, 256, 64]
    assert rect[0].tolist() == [-256, 768, -256, 768]  # one 1024 px window around the canvas
    assert rect[1].tolist() == [128 - 64, 128 - 64 + 256, 128 - 64, 128 - 64 + 256]  # (1, 1)
    assert rect[2].tolist() == [3 * 128 - 64, 3 * 128 + 192, 3 * 128 - 64, 3 * 128 + 192]
    # shifted layer: patch (3, 4) covers [32 * 3 - 16, +32) and its window half a patch more
    assert rect[3].tolist() == [32 * 3 - 32, 32 * 3 + 32, 32 * 4 - 32, 32 * 4 + 32]


def test_patch_to_canvas_roundtrip_of_the_adapter() -> None:
    p = torch.tensor([[0.3, 0.6, 0.5, 0.4, 0.1, 0.2, 0.3, 0.4]])
    out = patch_to_canvas(p, 2, 1, 128)
    assert out[0, 0] == pytest.approx((32 * 1 + 32 * 0.3) / 128)
    assert out[0, 1] == pytest.approx((32 * 2 + 32 * 0.6) / 128)
    assert torch.equal(to_window_units(p)[0, 4:], p[0, 4:])


# ------------------------------------------------------------------ pipeline


@needs_brushes
def test_replay_reproduces_the_pipeline_canvas(brushes: torch.Tensor) -> None:
    """Padded (64 x 96 in a 128 square), odd shifted grid (5 x 5), large strokes clipped at their
    windows: the layer-aware replay equals the pipeline canvas (float rounding of the parameter
    round trip only)."""
    image = smooth_image(64, 96)
    result = paint(ToyPredictor(), image, brushes)
    seq = result.sequence
    assert (
        seq.canvas_size == (64, 96) and seq.meta["levels"] == 3 and seq.meta["padded_size"] == 128
    )
    replay = replay_sequence(seq, brushes)
    diff = (replay - result.canvas).abs()
    assert float(diff.mean()) < 2e-5
    assert float((diff.amax(0) > 1 / 255).float().mean()) < 2e-4
    # the generic PT renderer is close (same strokes) but not equal (texture sampling, no clipping)
    generic = seq.render(cfg=_pt_render_config(po.BRUSHES))
    assert 1e-4 < float((generic - result.canvas).abs().mean()) < 0.05


def _pt_render_config(brush_dir: Path) -> object:
    from brushstroke.renderer.pipeline import RenderConfig

    return RenderConfig(mode="pt", pt_alpha="pt_infer", pt_brush_dir=str(brush_dir))


@needs_brushes
def test_pipeline_structure_and_determinism(brushes: torch.Tensor) -> None:
    image = smooth_image(64, 64)
    a = paint(ToyPredictor(), image, brushes)
    b = paint(ToyPredictor(), image, brushes)
    assert torch.equal(a.canvas, b.canvas) and torch.equal(a.sequence.params, b.sequence.params)
    assert a.stats["layers"] == 2 and [layer.grid for layer in a.layers] == [1, 2, 3]
    assert a.layers[-1].shifted and a.layers[-1].canvas_size == 64
    assert len(a.layer_canvases) == 3 and a.layer_canvases[0].shape == (3, 64, 64)
    # the first layer paints a single patch, the canvas changes at every layer
    assert not torch.equal(a.layer_canvases[0], a.layer_canvases[1])
    assert a.stats["strokes_official"] == len(a.sequence) == sum(a.stats["painted_per_layer"])  # type: ignore[arg-type]
    assert (
        a.canvas.shape == (3, 64, 64) and 0 <= float(a.canvas.min()) <= float(a.canvas.max()) <= 1
    )
    # image smaller than a patch: K = 0, only the first and the shifted layers
    tiny = paint(ToyPredictor(), smooth_image(20, 28), brushes)
    assert tiny.stats["layers"] == 1 and tiny.canvas.shape == (3, 20, 28)
    with pytest.raises(ValueError):
        paint(ToyPredictor(), image[None], brushes)
    with pytest.raises(ValueError):
        paint(ToyPredictor(), image, brushes, PTInferenceConfig(final_layer="nope"))


@needs_brushes
def test_final_layer_variants_official_and_confidence_gated(brushes: torch.Tensor) -> None:
    image = smooth_image(64, 64, seed=1)  # no padding: nothing is dropped as invisible
    result = paint(ToyPredictor(1), image, brushes)
    final = result.layers[-1]
    forced = int((final.painted & ~final.confident).sum())
    assert forced > 0
    assert result.stats["strokes_official"] - result.stats["strokes_gated"] == forced  # type: ignore[operator]
    sources = result.sequence.source or []
    assert sum(s == SOURCE_FORCED for s in sources) == forced
    gated = paint(ToyPredictor(1), image, brushes, PTInferenceConfig(final_layer="gated"))
    assert len(gated.sequence) == result.stats["strokes_gated"]
    assert SOURCE_FORCED not in (gated.sequence.source or [])
    assert torch.equal(gated.canvas, result.canvas) and torch.equal(
        gated.canvas_gated, result.canvas_gated
    )
    assert not torch.equal(result.canvas, result.canvas_gated)
    # the gated canvas is what the gated sequence replays
    replay = replay_sequence(gated.sequence, brushes)
    assert float((replay - result.canvas_gated).abs().mean()) < 2e-5
    # the official variant paints the strokes with a negative logit (exactly zero logits aside)
    assert bool(((final.logits != 0) == final.painted).all())


@needs_brushes
def test_negative_heights_are_made_positive_before_painting(brushes: torch.Tensor) -> None:
    """|h|-safe adapter: negative predicted heights never reach the canvas nor the sequence."""
    image = smooth_image(64, 96, seed=2)
    negative = ToyPredictor(2, negative_h=0.5)
    result = paint(negative, image, brushes)
    assert float(result.stats["negative_h_fraction"]) > 0.3  # type: ignore[arg-type]
    assert float(result.sequence.params[:, 3].min()) > 0
    assert all(float(layer.params[..., 3].min()) > 0 for layer in result.layers)
    # same draws, heights already positive: identical painting
    reference = paint(ToyPredictor(2, negative_h=0.5, absolute=True), image, brushes)
    assert torch.equal(result.canvas, reference.canvas)
    assert torch.equal(result.sequence.params, reference.sequence.params)
    assert reference.stats["negative_h_fraction"] == 0.0
    # without |h| the negative heights are kept (and mirror the brush): different painting
    raw = paint(negative, image, brushes, PTInferenceConfig(abs_h=False))
    assert float(raw.sequence.params[:, 3].min()) < 0
    assert not torch.equal(raw.canvas, result.canvas)
    # the replay of a sequence with positive heights equals its canvas
    assert float((replay_sequence(result.sequence, brushes) - result.canvas).abs().mean()) < 2e-5


@needs_brushes
def test_patch_batching_does_not_change_the_result(brushes: torch.Tensor) -> None:
    image = smooth_image(96, 96, seed=3)
    net = PaintTransformer().eval().requires_grad_(False)
    cfg = PTInferenceConfig()
    full = paint(net, image, brushes, cfg)
    chunked = paint(net, image, brushes, replace(cfg, max_patches_per_batch=3))
    assert float((full.canvas - chunked.canvas).abs().max()) < 1e-3
    assert [int(x) for x in full.stats["painted_per_layer"]] == [  # type: ignore[attr-defined]
        int(x)
        for x in chunked.stats["painted_per_layer"]  # type: ignore[attr-defined]
    ]


# ------------------------------------------------------------------ official parity


def _save_png(image: torch.Tensor, path: Path) -> None:
    arr = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


@needs_official
@pytest.mark.parametrize("size", [(256, 256), (160, 224)])
def test_inference_equals_the_official_code_with_the_official_weights(
    size: tuple[int, int], tmp_path: Path
) -> None:
    """The key test: official weights in our network, our pipeline == official ``inference.py``
    (float output, same device), also for a non-square image that needs padding."""
    path = tmp_path / "input.png"
    _save_png(smooth_image(*size, seed=size[1]), path)
    official, counts = official_paint(path)
    image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0)
    image = image.permute(2, 0, 1).contiguous().to(DEVICE)
    net, _ = load_pt_model(po.OFFICIAL / "model.pth", DEVICE)
    brushes = load_pt_brushes("inference", po.BRUSHES, DEVICE)
    result = paint(net, image, brushes)
    assert result.canvas.shape == official.shape
    diff = float((result.canvas.cpu() - official).abs().max())
    print(f"{size}: max abs difference to the official output {diff:.2e}")
    assert diff <= 1 / 255
    assert [int(n) for n in result.stats["painted_per_layer"]] == [c[1] for c in counts]  # type: ignore[attr-defined]
    assert float(result.sequence.params[:, 3].min()) > 0  # the official weights predict h > 0


# ------------------------------------------------------------------ refinement


@needs_brushes
def test_refinement_lowers_the_patch_loss_and_keeps_unpainted_strokes(
    brushes: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    n = 6
    img = smooth_image(32, 32 * n).view(3, 32, n, 32).permute(2, 0, 1, 3).contiguous()
    canvas = torch.zeros_like(img)
    shape, colour, logits = ToyPredictor(5).strokes(img, canvas)
    params = torch.cat([shape, colour], dim=-1)
    painted = logits > 0
    cfg = RefineConfig(enabled=True, iters=30)
    out, stats = refine_patch_strokes(img, canvas, params, painted, brushes, cfg)
    assert out.shape == params.shape and torch.isfinite(out).all()
    assert stats["loss_after"] < stats["loss_before"]
    assert torch.equal(out[~painted], params[~painted])  # unpainted strokes untouched
    assert 0 < stats["patches_kept"] <= 1
    again, _ = refine_patch_strokes(img, canvas, params, painted, brushes, cfg)
    assert torch.equal(again, out)  # deterministic
    # keeping the colour: the colour columns do not move
    fixed, _ = refine_patch_strokes(
        img, canvas, params, painted, brushes, replace(cfg, optimize_colour=False)
    )
    assert torch.equal(fixed[..., 5:], params[..., 5:])
    # zero iterations: nothing changes beyond the clamping into the parameter ranges
    same, stats0 = refine_patch_strokes(
        img, canvas, params, painted, brushes, replace(cfg, iters=0)
    )
    assert stats0["loss_after"] <= stats0["loss_before"] + 1e-6 and same.shape == params.shape


@needs_brushes
def test_refinement_flag_in_the_pipeline_improves_the_painting(brushes: torch.Tensor) -> None:
    image = smooth_image(64, 64, seed=4)
    base = paint(ToyPredictor(4), image, brushes)
    refined = paint(
        ToyPredictor(4),
        image,
        brushes,
        refine=RefineConfig(enabled=True, iters=40),
    )
    assert refined.sequence.meta["refined"] is True and base.sequence.meta["refined"] is False
    assert len(refined.stats["refine_layers"]) == 3  # type: ignore[arg-type]
    assert base.stats["refine_layers"] == []
    l1 = lambda r: float((r.canvas - image).abs().mean())  # noqa: E731
    assert l1(refined) < l1(base)
    # refined strokes are still replayable
    assert float((replay_sequence(refined.sequence, brushes) - refined.canvas).abs().mean()) < 2e-5
    only_last = paint(
        ToyPredictor(4),
        image,
        brushes,
        refine=RefineConfig(enabled=True, iters=5, levels=[2]),
    )
    assert [r["level"] for r in only_last.stats["refine_layers"]] == [2.0]  # type: ignore[attr-defined]
    off = paint(ToyPredictor(4), image, brushes, refine=RefineConfig(enabled=False, iters=5))
    assert torch.equal(off.canvas, base.canvas)


# ------------------------------------------------------------------ checkpoints and script


def test_load_pt_model_accepts_training_checkpoints_and_plain_state_dicts(tmp_path: Path) -> None:
    net = PaintTransformer(PTModelConfig(strokes=8)).eval()
    training = tmp_path / "latest.pt"
    torch.save(
        {"step": 7, "model": net.state_dict(), "config": {"model": asdict(PTModelConfig())}},
        training,
    )
    plain = tmp_path / "model.pth"
    torch.save({f"module.{k}": v for k, v in net.state_dict().items()}, plain)  # DataParallel
    loaded, info = load_pt_model(training)
    assert info == {"kind": "training", "step": 7} and not loaded.training
    other, info2 = load_pt_model(plain)
    assert info2["kind"] == "state_dict"
    img, canvas = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(loaded(img, canvas)[0], other(img, canvas)[0])
        assert torch.equal(loaded(img, canvas)[0], net(img, canvas)[0])
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(FileNotFoundError):
        load_pt_model(tmp_path / "missing.pt")


class _Payload:
    """Unpickling it would call ``print``: a stand-in for arbitrary code in a checkpoint."""

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (print, ("executed",))


def test_load_pt_model_never_unpickles_arbitrary_objects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "evil.pt"
    torch.save({"model": _Payload(), "config": {}}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_pt_model(path)
    assert "executed" not in capsys.readouterr().out


@needs_brushes
def test_paint_pt_script_writes_the_outputs(tmp_path: Path) -> None:
    net = PaintTransformer().eval()
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {"step": 1, "model": net.state_dict(), "config": {"model": asdict(PTModelConfig())}}, ckpt
    )
    img_path = tmp_path / "in.png"
    _save_png(smooth_image(70, 90, seed=5), img_path)
    out = tmp_path / "run"
    metrics = paint_pt.main(
        [
            "--input",
            str(img_path),
            "--out",
            str(out),
            "--checkpoint",
            str(ckpt),
            "device=cpu",
            "height=64",
            "width=64",
            "inference.final_layer=gated",
            f"inference.brush_dir={po.BRUSHES}",
            f"render.pt_brush_dir={po.BRUSHES}",
            "refine.enabled=true",
            "refine.iters=3",
        ]
    )
    for name in (
        "sequence.npz",
        "final.png",
        "target.png",
        "metrics.json",
        "config.resolved.yaml",
        "progress/layer_00.png",
        "progress/layer_02.png",
        "variants/final_official.png",
        "variants/final_gated.png",
    ):
        assert (out / name).exists(), name
    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["exported_variant"] == "gated" and saved["refine"] is True
    assert saved["checkpoint_info"]["kind"] == "training"
    assert {"l1", "psnr", "ssim", "lpips_vgg"} <= set(saved["official"])
    assert saved["strokes_exported"] == metrics["strokes_exported"]
    assert saved["strokes_exported"] + saved["dropped_invisible"] == saved["strokes_gated"]
    seq = StrokeSequence.load(out / "sequence.npz")
    assert len(seq) == saved["strokes_exported"] and seq.canvas_size == (64, 64)
    assert saved["sequence_vs_canvas"]["replay_mean_abs"] < 1e-4
