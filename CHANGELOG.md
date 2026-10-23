# Changelog

All notable changes, one entry per ticket (`[T-NNN]`, GitHub Issue #NNN). Newest first.

## [T-013] Paint Transformer training pipeline

- `scripts/train_pt.py` and `brushstroke/stroke_model/train.py`: the official schedule (56,320 steps of batch 64, Adam lr 1e-4,
  betas (0.5, 0.999), lr x0.3 every 5 epochs at the end), all set in `configs/stroke_model/pt.yaml`.
- Fixed-seed synthetic validation set, CSV logs, checkpoints with exact resume (a resumed CPU run is bit-identical to an
  uninterrupted one), optional bf16 (default fp32: no speed gain, 0.85 GB peak VRAM at batch 64) and a step profiler.
- Speed: the official code spends 40 % of its step in a per-sample matching loop; the batched cost matrix brings our step to
  about 38 ms against 237 ms for the official code, so the full schedule should take about 36 to 40 minutes on the RTX 5070.
- A 3000-step validation run (`docs/media/pt_train_curve.png`); the full run has not been launched.
- Design note section 13 with the step-time table.

## [T-012] Paint Transformer model and Hungarian-matched losses

- `brushstroke/stroke_model/model.py`: `PaintTransformer`, the official architecture with the same parameter names: 9,072,902
  trainable parameters; the official released weights load strictly, and their outputs match the official code exactly
(max difference 0.0; this test runs when the official files are present).
- `brushstroke/stroke_model/losses.py`: pixel loss (through the PT renderer of T-011), stroke L1, Wasserstein distance between
  Gaussian stroke representations, and confidence BCE with Hungarian matching (SciPy reference, exhaustive GPU matcher for up to 8
  strokes). Both matchers agree with a brute-force search in the tests.
- Two presets from YAML (`configs/pt/loss_official.yaml`, `loss_paper.yaml`): `official` (the default, the weights of the official
  code) and `paper_weights` (the weights of the paper). The model config is `configs/pt/model.yaml`.
- Tests: an overfitting run on one small batch drives the total loss down by more than 95 %; every loss term has non-zero gradients.

## [T-011] Paint Transformer-compatible rendering mode and synthetic stroke data generator

- `render.mode: pt` (`brushstroke/renderer/pt.py`): renders our `(B, N, 9)` strokes as the official Paint Transformer code does
  (two brush images chosen by `h > w`, binarized and eroded alpha in `pt_infer` / `pt_train`, dilated foreground, black canvas,
  no opacity). Bit-identical to the official inference `param2stroke` in the tests (max difference 0); the training pipeline
  agrees to 1.8e-7. The default procedural mode is unchanged.
- Parameter adapters between Paint Transformer and our convention, following the study note.
- `scripts/fetch_pt_brushes.py`: the four official brush images, fetched with SHA-256 checks into `data/raw/pt_brush/` (not committed).
  Without them the PT mode fails with a message naming the command.
- `brushstroke/stroke_model/synthetic.py`: GPU synthetic stroke generator following the official `set_input` distribution. Its
  batches equal the official ones with the same random stream; 13.6 k patches/s at batch 64 on the RTX 5070, against 1.07 k/s for
  the official code.
- Tests skip with a reason when the official files or brush images are absent.

## [T-010] Paint Transformer study note, reproduction targets, M2 plan

- `docs/design/paint_transformer.md`: the parameterization and renderer of the official code, the synthetic training data, the model,
  the losses (with the paper and code weights side by side), training, inference, reproduction targets with feasibility verdicts,
  licences and a list of paper-vs-code discrepancies. Each claim cites the paper or the official file and function, or is marked as measured.
- `docs/plan/M2.md`: the M2 plan (T-010 to T-017), measured estimates and proposed refinements, none applied silently.
- `scripts/pt_reference_study.py`: runs the official code and weights at a pinned commit with SHA-256 checks, outside git. Measured,
  for example, a 9,072,902-parameter model and about 3.7 h for the official training recipe on this hardware.
- No change to the package.

## [T-008] M1 gate review: fresh-clone reproduction and milestone audit

- Fresh clone of `main`, new venv, README commands only: setup, checks (190 tests), README snippets and the three demos all
  run. The regenerated demos are byte-identical to the committed ones (sequence hashes, GIFs, comparison images).
- README: status updated (M1 reviewed; three of four criteria met, LPIPS criterion not met), licence table, LPIPS weight
  download and proxy notes, the demo runtime shown in minutes so a rerun does not change the tracked table.
- `docs/plan/M1_retro.md`: criterion-by-criterion table with evidence, measured cost per stage, limitations, and ten proposed
  tickets for M2 and later.

## [T-007] Three M1 demo GIFs, reproducible with one command

- `python -m scripts.make_m1_demos`: fetch, paint (baseline defaults) and animate three public-domain inputs (portrait,
  landscape, photo); writes `docs/media/m1_{portrait,landscape,photo}.gif`, side-by-side images and `m1_results.md`.
- Results at 512 px, 2037 strokes: portrait and photo meet the revised target (LPIPS-VGG 0.253 and 0.251, LPIPS-Alex 0.217
  and 0.289); the landscape (The Starry Night) does not (0.358 / 0.432). Two full runs gave identical sequences, GIFs and metrics.
- Animation: `easing: by_level` (time per coarse-to-fine level ~ strokes ** 0.5).
- Reproducibility: `cudnn_deterministic` switch for the baseline (the GPU non-determinism came from cuDNN convolution backward
  in the LPIPS-VGG loss); SSIM and shared image metrics in `brushstroke/eval/metrics.py`.
- Demo inputs reuse the T-009 manifest and checksums; documented in `docs/data_sources.md`.

## [T-009] Investigation of the optimization-baseline quality gap; new baseline defaults

- Sweep of hypotheses H1 to H7 (`scripts/exp_optim_sweep.py`, `configs/experiments/optim_sweep.yaml`), with held-out
  LPIPS-AlexNet reported for every run, screening on 2 images and confirmation on 5 images x 3 seeds. Verdicts and tables in
  `docs/experiments/optim_sweep.md` and `optim_sweep_tables.md`.
- New baseline defaults in `configs/baseline_optim.yaml`: grids 1 to 20 with more strokes at the fine levels (2037 strokes),
  more steps at the fine levels (`iters_per_level`), joint fine-tuning of earlier strokes (`finetune_all`). Mean over 5 images x 3 seeds:
  LPIPS-VGG 0.416 -> 0.308, held-out LPIPS-AlexNet 0.465 -> 0.359, about 9.4 min and 3.3 GB per 512 px image.
- Code additions needed by the sweep: `iters_per_level`, `init_mode`, `init_size_mode`, `final_pass_iters`, LPIPS-AlexNet.
- The M1 target (LPIPS-VGG < 0.25) is **not met**. A revised target is proposed in the report, pending the chef's decision.
- Data: `scripts/fetch_data.py` with checksums and `docs/data_sources.md` (public-domain images, licenses recorded).

## [T-006] Stroke-by-stroke animation with partial reveal and ffmpeg export

- `brushstroke/animation/`: deterministic timeline (overlap, minimum reveal frames, uniform or decelerating slots, seeded
  stroke directions), partial reveal of each stroke along its main axis computed from the renderer's own affine grid
  (`reveal_front` / `rasterize_flat(reveal=...)` in `renderer/rasterize.py`; a finished stroke is bit-identical to the
  normal render), and an incremental frame renderer (finished prefix + strokes in flight).
- Canvas texture (T-002) applied to every frame.
- Export through ffmpeg: MP4 (H.264), WebM (VP9), GIF (two-pass palette). Clear error when ffmpeg is missing.
- `configs/animation.yaml` and `scripts/animate.py`; design note `docs/design/animation.md` with the timing measurements
  (1000 strokes, 512 x 512, 10 s at 30 fps: 2.5 s for MP4 + WebM, 89 MB peak VRAM).

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
