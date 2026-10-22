# Changelog

All notable changes, one entry per ticket (`[T-NNN]`, GitHub Issue #NNN). Newest first.

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
