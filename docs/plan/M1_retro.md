# M1 retrospective and gate audit (T-008)

Scope of M1: differentiable renderer, optimization-only decomposition (baseline a), animation export, three demo GIFs
([plan](M1.md)). This document audits M1 as an outsider would: a fresh clone of `main` (commit `135bb92`), setup from the README only,
all checks, regeneration of the demos, then every success criterion.

## 1. Fresh-clone reproduction

Machine: Windows 11, Python 3.11.9, RTX 5070 (12 GB), driver 610.88, ffmpeg 9.0.1. The clone was made with `git clone` of the local
`main` repository into `outputs/fresh_clone/` (the repository has no remote), with a new venv inside it.

| step (README commands, in order) | result |
|----------------------------------|--------|
| `git clone ... outputs/fresh_clone` | 97 tracked files, `.git` 5.5 MB |
| `py -3.11 -m venv .venv` + `pip install -U pip` | **fails** with `CERTIFICATE_VERIFY_FAILED` on this machine (TLS-intercepting proxy); fixed by the README troubleshooting entry (`pip.ini` with `trusted-host` inside the venv), then OK |
| `pip install -c constraints.txt -r requirements.txt -r requirements-dev.txt` | OK, 75 s (pip's cache held the torch wheels), torch `2.11.0+cu128`, CUDA available, capability `(12, 0)`; the Windows path limit was not hit with the clone 76 characters deep |
| `ruff check .` / `ruff format --check .` / `mypy brushstroke` | all green (71 files formatted, 32 files type-checked) |
| `pytest -q` | **190 passed**, 20.4 s (GPU and slow tests included); CI subset `-m "not gpu and not slow"`: 176 passed, 14 deselected, 15.3 s |
| README snippets (configuration, render, `StrokeSequence`, `preview_brushes`) | all run as written |
| `python -m scripts.fetch_data --insecure`, then the README's fast `paint_optim` example | OK (52 s, L1 0.0317); `animate` examples produce MP4/WebM/GIF (0.51 / 0.23 / 0.52 MB for 345 frames) |
| `python -m scripts.make_m1_demos --insecure` | **OK, about 29 min** (3 x 566 s painting + animation); inputs fetched and checksum-verified |

**The demos are reproduced exactly**: the three sequence hashes of the fresh run (`ee4dd8ac823f`, `c7f02dc260f5`, `67904fe4faf9`) equal those
of the committed table, and the three GIFs and three side-by-side images are byte-identical (`cmp`) to the committed files, from a different
venv and clone. The regenerated `m1_results.md` differed from the committed one only by the runtime column (565 s vs 566 s), which made a
rerun dirty a tracked file; the column is now shown in minutes (9.4), identical on both runs.

### Friction points found, and what was done
| # | friction | fix |
|---|----------|-----|
| 1 | TLS-intercepting proxy: pip, the image download and the torchvision weights all fail with certificate errors | already in the README (pip); `--insecure` for `make_m1_demos` now stated next to the command; weights note added to the troubleshooting section |
| 2 | LPIPS needs the VGG16 (528 MB) and AlexNet (233 MB) ImageNet weights, downloaded on first use; not mentioned anywhere (they were already cached on the development machine, so the clone did not hit it) | added to the README troubleshooting section |
| 3 | The README's `paint_optim` example reads `data/raw/mona_lisa.jpg`, which exists only after `fetch_data` | README now says to run `python -m scripts.fetch_data` first |
| 4 | Rerunning `make_m1_demos` modified the tracked `docs/media/m1_results.md` (runtime 566 s vs 565 s) | runtime column in minutes, tested; regenerated table committed |
| 5 | README status said "M1 in progress", the license section said licenses "will be listed", the layout described `eval/` as M3-only | updated (status, a license table, layout) |
| 6 | The one-command demo run needs ~29 min and a GPU: there is no CPU path for the demos (CPU is supported by tests only) | documented here (limitation, section 5) |

No code bug was found in the fresh clone; every acceptance command worked.

## 2. Success criteria of the M1 plan

| # | criterion (docs/plan/M1.md) | verdict | evidence |
|---|-----------------------------|---------|----------|
| 1 | Renderer unit-tested: shapes, gradients (including gradcheck), determinism, parallel == sequential compositing | **met** | `tests/test_renderer.py` (39 tests): float64 `gradcheck` on all 9 parameters; finite non-zero gradients for every parameter (CPU and GPU); bitwise CPU determinism; parallel/chunked compositing == sequential at atol 1e-5 for chunk sizes 1 to 64, also with saturated alphas; progressive == prefix renders; CPU vs GPU at 1e-4; geometry (centroid < 1 px, extents < 2 px, rotation periodicity) |
| 2 | Optimization baseline reaches LPIPS < 0.25 on 512² demo images within 2000 strokes | **not met** (reported honestly) | T-005 default 0.416 mean LPIPS-VGG (5 images x 3 seeds); after the T-009 investigation **0.308 ± 0.053** (held-out LPIPS-Alex 0.359 ± 0.102) at 2037 strokes; 2 of 5 images within 0.003 of 0.25. A plain 1-px Gaussian blur of these images already scores 0.268 / 0.318, and no cited method reports LPIPS ([docs/experiments/optim_sweep.md](../experiments/optim_sweep.md)). **Revised target** used from T-007 on: LPIPS-VGG <= 0.35 and LPIPS-Alex <= 0.40 per image, <= 2000 strokes, <= 10 min. Demos: portrait 0.253 / 0.217 and photo 0.251 / 0.289 meet it; **landscape (The Starry Night) 0.358 / 0.432 misses it** |
| 3 | Animation shows per-stroke partial reveal, no visible brush | **met** | `tests/test_animation.py` (22 tests): reveal coverage is monotone in progress, exactly 1 at the end, the front moves along the stroke axis; incremental frames equal the brute-force composite of every frame (atol 1e-5); last frame == full render (+ canvas) within 1/255; frames inspected by eye for all three demos (strokes laid down with a soft front, no brush or cursor) |
| 4 | 3 demo GIFs, <= 4 MB each, reproducible with one command | **met** | 1.17 / 1.32 / 1.23 MB, 480 px, 15 fps, 12 s (10 s + 2 s hold); `python -m scripts.make_m1_demos`; reproduced bit for bit twice in the development tree and once from a fresh clone and new venv (section 1) |

Ticket-level acceptance criteria were checked by the reviewer for each of T-001 to T-007 and T-009 before merge; none is reopened by this audit.

## 3. Repository hygiene

| check | result |
|-------|--------|
| files in git | 97 (at the audited commit), none larger than 1.32 MB (the three GIFs); no file > 4 MB, no blob > 4 MB anywhere in the history |
| weights / datasets / media | none tracked: no `*.pt, *.pth, *.ckpt, *.safetensors, *.bin, *.npz`; the only binaries are 3 GIFs, 3 side-by-side JPEGs and 3 experiment figures (all <= 105 KB except the GIFs); inputs live in `data/raw/` (gitignored), outputs in `outputs/` |
| secrets | `git grep` for key/token/password patterns: no hit |
| `.gitignore` | `.venv/`, outputs, runs, `data/raw/`, weights, videos and GIFs except `docs/media/*.gif`, caches, `.handoff/`, `tasks/` |
| licenses | code MIT; third-party and data licenses listed in the README license table and [docs/data_sources.md](../data_sources.md) (images: public domain with source URL and SHA-256 each; weights downloaded at run time, never committed) |
| reproducibility of the environment | full transitive pin set in `constraints.txt` (verified equal to the resolved set by a dry run in T-001/T-009), installs cleanly from scratch (above) |

## 4. Measured cost per stage (RTX 5070, 12 GB; VRAM = peak allocated)

| stage | measured | M1 plan estimate |
|-------|----------|------------------|
| brush bank (3 kinds x 8 variants, 64 x 128), canvas texture 512² | 44 ms, 18 ms (CPU) | n/a |
| renderer, 512², 1024 strokes, forward + backward | **0.76 s, 0.65 GB with checkpointing**; 11.3 GB without it (activations ~44 bytes per stroke pixel) | 2-4 GB (chunked compositing): true only with checkpointing, which is therefore the right default for large N |
| renderer, 1024², 4096 strokes, forward + backward | 15.5 s, 4.1 GB (checkpointing, chunk 16) | n/a |
| baseline, T-005 default (1794 strokes, L1 + LPIPS-VGG) | 149 s, 6.7 GB | +1.5 GB for LPIPS on top of the renderer |
| baseline, new default (2037 strokes, joint fine-tuning) | **566 s (9.4 min), 3.3 GB** | n/a |
| animation, 1000 strokes, 512², 10 s @ 30 fps + 1.5 s hold, MP4 + WebM | 2.5 s, 89 MB | < 1 GB: confirmed |
| full demo pipeline (3 demos) | about 29 min | n/a |
| test suite | 190 tests in 20 s (15 s without GPU/slow tests) | n/a |

## 5. Known limitations and open issues

1. **Speed**: every stroke is rasterized over the whole canvas at every optimization step (cost O(H·W) per stroke) although strokes at the fine
   levels cover about 1/500 of it. Bounding-box cropping would cut the 9.4 min substantially and allow more strokes/steps.
2. **Quality gap**: LPIPS-VGG 0.308 mean (target 0.25 in the plan); textured, high-frequency inputs (Starry Night, Great Wave) stay at 0.35-0.38.
   The remaining gap is a stroke-budget/abstraction effect, not an optimization bug (T-009), so it is a baseline property for M3, not an M1 defect.
3. **GPU reproducibility is conditional**: bit-identical runs need `cudnn_deterministic: true` (cuDNN's convolution backward, used by the LPIPS loss, is
   non-deterministic otherwise; two runs of one seed differed by up to 0.7 in stroke parameters). It is on for the demos but off in `configs/baseline_optim.yaml`;
   results are only guaranteed identical on the same GPU/driver/torch build.
4. **GPU required for the demos** (and a recent driver for sm_120); CI runs CPU-only tests (176 of 190), so GPU and LPIPS-weight paths are not covered by CI.
5. **First-use downloads** (ImageNet weights, input images) need internet access; behind a TLS-intercepting proxy they need manual steps (README).
6. **Data model gaps**: `StrokeSequence` stores a background colour only (no background image), `seq[0]` is not supported (use `seq[[0]]`), and the brush bank
   is regenerated from a spec rather than stored.
7. **Paint Transformer compatibility**: our brush alpha is soft and single-image; Paint Transformer uses binarized, eroded alpha and two brush images
   (selection by `h > w`). Needed for the M2 reproduction.
8. **Protocol limits of T-009**: screening on 2 images/1 seed, interactions along one path, the Learning-to-Paint metric not re-checked in the literature review.
9. **Platform**: developed and tested on Windows only; the Linux setup path is documented but was exercised only through CI (CPU).

## 6. Proposed tickets for M2 and later

| # | ticket | milestone | rationale |
|---|--------|-----------|-----------|
| 1 | Bounding-box (per-stroke crop) rasterization in the renderer, with equivalence tests against the full-canvas path | M2 (enabler) | removes the main cost of the baseline and of any trainer that renders strokes; the M2 stroke-predictor training loop needs a fast renderer |
| 2 | `pt_brush` option: binarized/eroded alpha and two brush images selected by `h > w`, plus a Paint Transformer parameter adapter | M2 | exact compatibility for the reproduction, identified in T-002/T-003 |
| 3 | Paint Transformer reimplementation, synthetic-stroke self-supervised training, multi-scale inference (the M2 ticket set) | M2 | brief component (B) |
| 4 | Evaluation harness: held-out metrics (LPIPS-Alex, SSIM, add DISTS), multi-seed protocol, anytime-quality curve (LPIPS vs stroke count via `render_progressive`) | M3 | the contribution is evaluated by this; `eval/metrics.py` and `StrokeSequence.render_progressive` already exist |
| 5 | Decide the multi-image evaluation set (PD/CC0, diverse textures) and fix per-image targets | M3 | T-009 showed results depend strongly on the image |
| 6 | Ticket for the Learning-to-Paint (RL) baseline and a published-number check of its metric | M3 | baseline (c); closes the unchecked literature point |
| 7 | Optional CPU/Linux path for the demos (smaller budget) and a CI job that runs one tiny painting end to end | tooling | lets non-GPU contributors reproduce a demo |
| 8 | Background image support and single-stroke indexing in `StrokeSequence`; storing the brush bank hash | M2/M3 | known data-model gaps |
| 9 | Content-aware animation pacing (slower on salient regions) | M4 | uses the saliency allocation of the contribution |
| 10 | Pin down GPU determinism in CI-friendly form (document `cudnn_deterministic`, add it to the baseline config for experiments) | tooling | limitation 3 |
