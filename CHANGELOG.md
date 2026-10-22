# Changelog

All notable changes, one entry per ticket (`[T-NNN]`, GitHub Issue #NNN). Newest first.

## [T-005] Baseline (a): optimization-only stroke decomposition

- `brushstroke/baselines/optim/`: progressive coarse-to-fine optimization of strokes (grids 1 x 1 to 20 x 20, new strokes
  initialized from the residual error and optimized jointly per level, earlier levels frozen by default), through the
  T-003 renderer. Output: a `StrokeSequence` with `scale_level` and `patch_id` tags and timings in `meta`.
- Losses, each behind a weight: pixel L1, LPIPS-VGG, debiased log-domain Sinkhorn OT on pooled intensities.
- `configs/baseline_optim.yaml` and `scripts/paint_optim.py` (writes the sequence, PNGs, metrics and loss curves).
- Result on a 512 x 512 public-domain image (1794 strokes): L1 0.036, PSNR 26.1 dB, LPIPS-VGG 0.44, 149 s, 6.7 GB peak.
  The M1 target (LPIPS < 0.25 within 2000 strokes) is not met; the design note reports the tuning runs.
- Design note `docs/design/optim_baseline.md`, with the differences from Stylized Neural Painting.

## [T-004] Stroke sequence data model and serialization

- `brushstroke/strokes.py`: `StrokeSequence` (ordered strokes `(N, 9)` in the renderer convention, per-stroke tags
  `brush_index`, `scale_level`, `patch_id`, `source`, metadata, background, schema version) and `BrushSpec` (which brush
  bank the strokes were made for).
- Operations: slicing / selection, `reorder` (validated permutation), `concat`, `to(device)`, `render` and
  `render_progressive` (re-rendering at any resolution), `equals`.
- `from_batch`: the producer path. It keeps the brush variants the renderer drew, so a sequence re-renders bitwise like
  the producer's batch. The renderer's default brush draw is now one rule shared by both (seeded per painting, so
  independent of the batch size); this changed the default draw for batches and non-zero seeds (`docs/design/renderer.md`).
- Single-file `.npz` save/load with an embedded JSON header; a newer schema version is rejected, older ones go through
  registered migrations (none yet).
- Design note `docs/design/stroke_sequence.md`.

## [T-003] Differentiable batched textured stroke renderer

- `brushstroke/renderer/params.py`: stroke layout `(B, N, 9)` = x, y, w, h, theta, r, g, b, a (Paint Transformer's
  `param2stroke` layout plus opacity), clamping, sigmoid re-parameterization and its inverse, `canonicalize` (strokes
  become at least as long as thick in pixels, so the brush texture follows the long side).
- `brushstroke/renderer/rasterize.py`: affine warp of the brush masks with `affine_grid` + `grid_sample`, same matrix as
  Paint Transformer; `sampler: gather` option for a deterministic CUDA backward.
- `brushstroke/renderer/compositing.py`: over operator, naive sequential reference and closed-form parallel version.
- `brushstroke/renderer/pipeline.py`: `render`, `render_progressive` (one pass, canvases at chosen stroke counts),
  chunking and optional checkpointing (`render:` config section, typed `RenderConfig`).
- `scripts/bench_renderer.py` and the benchmark table in `docs/design/renderer.md` (512², 1024 strokes forward+backward:
  0.76 s, 0.65 GB peak with checkpointing; about 11 GB without).
- Tests: geometry, compositing equivalence, progressive rendering, gradcheck in float64 over all 9 parameters, determinism.

## [T-002] Procedural textured brush masks and canvas texture

- `brushstroke/renderer/brushes.py`: `make_brush` with kinds `flat` (antialiased rectangle, ablation baseline), `oil`
  (bristle streaks along the main axis, ragged soft sides, tapered and dry ends) and `round` (soft dab with mild texture);
  `BrushBank` (K seeded variants as one `(K, 1, h, w)` tensor, memoized per config).
- `brushstroke/renderer/canvas.py`: `make_canvas_texture` (woven-canvas field with irregular threads and low-frequency
  noise) and `apply_canvas` (shading, used by the animation in T-006).
- Config: `brush:` and `canvas:` sections in `configs/default.yaml`, typed as `brushstroke.config.RunConfig`.
- `scripts/preview_brushes.py`: contact sheet of the brush banks and the canvas (written to `outputs/`).
- Design note `docs/design/brushes.md`: conventions (main stroke axis = local x; renderers must canonicalize to `w >= h`),
  references (Stylized Neural Painting, Paint Transformer, Learning to Paint).

## [T-001] Project scaffold, pinned environment, config system, CI, M1 plan

- Package layout `brushstroke/` (renderer, stroke_model, contribution, animation, generation, eval, utils) with typed package.
- Utilities: YAML config with `key.sub=value` overrides and resolved-config dump (`brushstroke.utils.config`),
  `seed_everything` (`brushstroke.utils.seed`), device selection and peak VRAM (`brushstroke.utils.device`),
  image I/O to float tensors in [0, 1] with resize helpers (`brushstroke.utils.io`).
- Pinned environment: `requirements.txt`, `requirements-dev.txt`, full transitive closure in `constraints.txt`
  (PyTorch 2.11.0 from the CUDA 12.8 index, for RTX 50-series GPUs).
- Tooling: ruff (lint + format), mypy strict on `brushstroke/`, pytest with `gpu` (auto-skip without CUDA) and `slow` markers.
- CI (`.github/workflows/ci.yml`): Ubuntu, Python 3.11, CPU torch; ruff, mypy, `pytest -m "not gpu and not slow"`.
- Documentation: README (setup, layout, configuration, checks), `docs/README.md` index, `docs/plan/M1.md`, MIT license.
