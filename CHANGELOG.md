# Changelog

All notable changes, one entry per ticket (`[T-NNN]`, GitHub Issue #NNN). Newest first.

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
