# BrushStroke

Neural stroke-based painting: turn a text prompt or an image into a painting made of an ordered sequence
of brush strokes, replayed as a stroke-by-stroke animation. The research focus is semantic-aware stroke
allocation, painterly stroke ordering and an "anytime quality" metric.

**Status: M1 reviewed (gate, T-008); M2 reviewed (T-018, [docs/plan/M2_retro.md](docs/plan/M2_retro.md)).** Three of the four M1 success criteria are met; the LPIPS < 0.25 criterion of the plan is
not met and a revised target is used (plan: [docs/plan/M1.md](docs/plan/M1.md); retrospective and criterion table:
[docs/plan/M1_retro.md](docs/plan/M1_retro.md)). M2 (Paint Transformer reimplementation) is reviewed at its gate, and the closing decision is the chef's: the study note is
[docs/design/paint_transformer.md](docs/design/paint_transformer.md), the plan is [docs/plan/M2.md](docs/plan/M2.md), and the
Paint Transformer-compatible renderer mode, synthetic stroke generator (T-011), model and Hungarian-matched losses (T-012) and the
training pipeline (T-013) are in place, three full training runs have been done (T-014, see
[docs/experiments/pt_training.md](docs/experiments/pt_training.md)), and the multi-scale inference pipeline reproduces the official
inference exactly with the official weights (T-015, see [Painting with the Paint Transformer](#painting-with-the-paint-transformer-m2)),
and the reproduction study (T-016) checks the paper's results: the ablation and real-image claims are mostly not reproduced (see
[Reproducing the Paint Transformer results](#reproducing-the-paint-transformer-results-m2)), and the stroke-convention study (T-019)
gives an invariant metric and options that are off by default (see [Training the Paint Transformer](#training-the-paint-transformer-m2)).
Documentation index: [docs/README.md](docs/README.md).
Changes per ticket: [CHANGELOG.md](CHANGELOG.md).

## M1 demos

Three paintings made by the optimization baseline from public-domain images, each animated stroke by stroke (10 s of
painting and a 2 s hold; the GIFs are 480 px wide, about 1.2 MB each). Settings and reproduction: [Painting an image](#painting-an-image-baseline-a).

| Portrait: *Girl with a Pearl Earring* | Landscape: *The Starry Night* | Photo: *Earth from Apollo 17* |
|---|---|---|
| ![Portrait demo](docs/media/m1_portrait.gif) | ![Landscape demo](docs/media/m1_landscape.gif) | ![Photo demo](docs/media/m1_photo.gif) |

| demo | strokes | L1 | LPIPS-VGG | LPIPS-Alex (held out) | target met |
|------|---------|----|-----------|------------------------|------------|
| portrait | 2037 | 0.024 | 0.253 | 0.217 | yes |
| landscape | 2037 | 0.071 | 0.358 | 0.432 | **no** |
| photo | 2037 | 0.055 | 0.251 | 0.289 | yes |

The target is the revised one adopted for the demos (LPIPS-VGG ≤ 0.35 and LPIPS-Alex ≤ 0.40 at ≤ 2000 strokes and ≤ 10 min
per 512 px image). The landscape misses it: its swirling high-frequency brushwork is hard to reproduce with about 2000 strokes.
Full table with runtime, VRAM, SSIM and hashes: [docs/media/m1_results.md](docs/media/m1_results.md). Regenerate everything with
`.venv\Scripts\python -m scripts.make_m1_demos` (about 29 minutes on an RTX 5070; it downloads and verifies the three inputs itself;
behind a TLS-intercepting proxy add `--insecure`, and see the weights note in the troubleshooting section). A rerun from a fresh clone
reproduces the committed GIFs byte for byte on the same GPU.

## M2 demos

The same three inputs painted by the Paint Transformer (this repository's model, T-014 checkpoint) and animated with the
same animation module; on the right of each side-by-side GIF, the M1 optimization baseline on the left. 3,866 to 5,020 strokes
in under a second, against 2,037 strokes in 9.4 minutes for the baseline, at a clearly lower fidelity (see the table).

| Portrait | Landscape | Photo |
|---|---|---|
| ![Portrait, M1 vs PT](docs/media/m2_portrait_vs_m1.gif) | ![Landscape, M1 vs PT](docs/media/m2_landscape_vs_m1.gif) | ![Photo, M1 vs PT](docs/media/m2_photo_vs_m1.gif) |

| demo | method | strokes | L1 | LPIPS-VGG | LPIPS-Alex | runtime |
|---|---|---|---|---|---|---|
| portrait | M1 optimization | 2037 | 0.024 | 0.253 | 0.217 | 9.4 min |
| portrait | Paint Transformer | 3866 | 0.036 | 0.541 | 0.416 | < 1 s |
| landscape | M1 optimization | 2037 | 0.071 | 0.358 | 0.432 | 9.4 min |
| landscape | Paint Transformer | 5020 | 0.091 | 0.651 | 0.538 | < 1 s |
| photo | M1 optimization | 2037 | 0.055 | 0.251 | 0.289 | 9.4 min |
| photo | Paint Transformer | 4351 | 0.074 | 0.532 | 0.382 | < 1 s |

Full table, the PT-only GIFs (`m2_<name>.gif`) and the input | M1 | PT images: [docs/media/m2_results.md](docs/media/m2_results.md).
Regenerate with `.venv\Scripts\python -m scripts.make_m2_demos` (needs `outputs/pt_long/seed2/latest.pt`, or
`--checkpoint data/raw/pt_official/model.pth` for the released weights; the M1 baselines are painted first if `outputs/m1/` is
empty, about 28 minutes). The GIFs are reproduced byte for byte on a rerun. Milestone review: [docs/plan/M2_retro.md](docs/plan/M2_retro.md).

## Setup

Requires Python 3.11. The pinned PyTorch build comes from the CUDA 12.8 index, which an NVIDIA RTX 50-series
GPU (Blackwell, sm_120) needs.

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -c constraints.txt -r requirements.txt -r requirements-dev.txt
```

Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -c constraints.txt -r requirements.txt -r requirements-dev.txt
```

`constraints.txt` pins the full transitive closure (reproducible installs); `requirements.txt` lists the direct
dependencies and already contains `--extra-index-url https://download.pytorch.org/whl/cu128`.
For a CPU-only machine (as in CI), install torch first from the CPU index:
`pip install -c constraints.txt torch==2.11.0+cpu torchvision==0.26.0+cpu --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple`,
then the install command above.

Do not `pip install -e .`: run scripts from the repo root (`python -m scripts.<name>`); pytest uses
`pythonpath = ["."]`. The commands below use the venv interpreter explicitly (`.venv\Scripts\python` on Windows,
`.venv/bin/python` on Linux); activating the venv and calling `python` works the same way.

### Troubleshooting

- `CERTIFICATE_VERIFY_FAILED` from pip (TLS interception on some machines): add a `pip.ini` **inside the venv** (machine-specific,
  never committed) with
  `[global]` / `trusted-host = download.pytorch.org pypi.org files.pythonhosted.org` (one host per line).
- Windows path limit: create the venv under a short path; torch has very deep paths and the install breaks otherwise.
- First use of LPIPS (the default baseline loss, the metrics of the demos and several tests) downloads the torchvision ImageNet weights of
  VGG16 (528 MB) and AlexNet (233 MB) into the torch hub cache (`~/.cache/torch/hub/checkpoints`) from `download.pytorch.org`; behind a
  TLS-intercepting proxy this download fails with the same certificate error. Pre-download them once (for example with the
  `lpips.LPIPS(net="vgg")` and `net="alex"` constructors with certificate checks relaxed for that one command) or run on a machine with
  direct access. Tests that need them are marked `slow`; the CI subset does not.
- Painting needs an input image: run `python -m scripts.fetch_data` first (add `--insecure` behind such a proxy); it fills `data/raw/`.

## Layout

```
brushstroke/          Python package
  renderer/           brush masks and canvas texture (T-002); differentiable batched stroke renderer (T-003);
                      Paint Transformer-compatible mode `render.mode: pt` (T-011)
  stroke_model/       stroke predictor (M2): Paint Transformer model (T-012), losses (T-012), training pipeline (T-013, T-014),
                      synthetic stroke generator (T-011)
  contribution/       semantic allocation and painterly ordering (M4)
  animation/          stroke-by-stroke replay with partial reveal, ffmpeg export (T-006)
  generation/         text/image front-ends (M5)
  eval/               image metrics: L1, PSNR, SSIM, LPIPS-VGG / LPIPS-Alex (T-007); evaluation harness (M3)
  baselines/optim/    baseline (a): optimization-only stroke decomposition (T-005)
  strokes.py          StrokeSequence: ordered strokes + metadata, save/load (T-004)
  utils/              config (YAML + dot-overrides), seeding, device/VRAM helpers, image I/O
configs/              YAML run configurations (every hyperparameter lives here)
scripts/              entry points, one per experiment
tests/                pytest suite
docs/                 plan/ (milestones), design/ (component notes), media/ (optimized demo GIFs)
report/               technical report
```

## Configuration

Run settings are typed dataclasses loaded from YAML in `configs/`, with `key.sub=value` overrides from the
command line. The root schema is `brushstroke.config.RunConfig` (sections `brush:`, `canvas:` and `render:`; the
YAML values override the dataclass defaults, and unknown keys or wrong types raise an error).

```python
from brushstroke.config import RunConfig
from brushstroke.utils.config import dump_config, load_config

cfg = load_config(RunConfig, "configs/default.yaml", overrides=["brush.kind=round"])
dump_config(cfg, cfg.output_dir)  # writes outputs/default/config.resolved.yaml
```

Brushes and canvas: `brush.kind` is `flat`, `oil` or `round` (procedural masks, see
[docs/design/brushes.md](docs/design/brushes.md)). Preview them as a contact sheet (written to `outputs/`, not committed):

```
.venv\Scripts\python -m scripts.preview_brushes output_dir=outputs/brush_preview
```

Rendering strokes: a stroke set is a tensor `(B, N, 9)` (`x, y, w, h, theta, r, g, b, a`, documented in
[docs/design/renderer.md](docs/design/renderer.md)). `render` composites the N strokes in order onto a white (or given)
canvas and returns `(B, 3, H, W)`; `render_progressive` returns the canvas after several stroke counts in one pass.

```python
import torch

from brushstroke.config import RunConfig
from brushstroke.renderer import BrushBank, render
from brushstroke.utils.config import load_config

cfg = load_config(RunConfig, "configs/default.yaml")
bank = BrushBank.from_config(cfg.brush)
params = torch.rand(2, 32, 9)  # (B, N, 9); w and h should be kept away from 0
image = render(params, bank, 256, 256, cfg=cfg.render)  # (2, 3, 256, 256)
```

Rendering settings are in the `render:` section (`chunk_size`, `use_checkpoint`, `sampler`, `dtype`). The
benchmark (time and peak VRAM) is `.venv\Scripts\python -m scripts.bench_renderer`, which writes
`outputs/bench_renderer.md`.

Stroke sequences: a painting as an ordered list of strokes with its metadata is `brushstroke.strokes.StrokeSequence`
(schema, `.npz` format and versioning in [docs/design/stroke_sequence.md](docs/design/stroke_sequence.md)). It is the
format every producer should emit and every consumer should read. A producer's batch painting `b` is wrapped with
`from_batch`, which keeps the brush variants the renderer used, so the sequence re-renders identically.

```python
import torch

from brushstroke.strokes import BrushSpec, StrokeSequence

params = torch.rand(2, 20, 9)  # batch of 2 paintings from a producer, 20 strokes each
seq = StrokeSequence.from_batch(
    params, 1, BrushSpec(kind="oil", index_seed=0), canvas_size=(256, 256)
)
image = seq.render()  # (3, 256, 256)
seq.save("outputs/painting.npz")
same = StrokeSequence.load("outputs/painting.npz")
```

Selection uses slices, lists or boolean masks (`seq[:k]`, `seq[[0, 3]]`); a single stroke by a bare integer
(`seq[0]`) is not supported yet, use `seq[[0]]`.

Seeding for reproducible runs: `brushstroke.utils.seed.seed_everything(seed, deterministic=...)` (call it before any
CUDA work). Peak GPU memory is measured with `brushstroke.utils.device.reset_peak_vram()` and `peak_vram_mb()`.

## Painting an image (baseline a)

`scripts/paint_optim.py` decomposes an image into strokes by progressive optimization (Stylized Neural Painting style,
see [docs/design/optim_baseline.md](docs/design/optim_baseline.md)). Settings are in `configs/baseline_optim.yaml`; any
key can be overridden on the command line. The input image is not in the repository: use a public-domain or
self-made image, for example in `data/raw/` (gitignored). The evaluation images can be fetched with
`python -m scripts.fetch_data` (see [docs/data_sources.md](docs/data_sources.md)).

```
.venv\Scripts\python -m scripts.paint_optim --input data/raw/mona_lisa.jpg --out outputs/optim_run
.venv\Scripts\python -m scripts.paint_optim --input data/raw/mona_lisa.jpg --out outputs/optim_fast optim.iters_per_level=[20,20,20,20,20,20] loss.lpips=0
```

`optim.iters` is used only when `optim.iters_per_level` is empty; the default sets the steps per level explicitly.

The output directory gets `final.png`, `target.png`, `progress/level_XX.png` (canvas after each level),
`sequence.npz` (the stroke sequence), `metrics.json` and `loss_curve.json`. The default schedule (2037 strokes at 512 x 512,
about 9.4 minutes and 3.3 GB peak VRAM on an RTX 5070) reaches a mean LPIPS-VGG of 0.308 and a held-out LPIPS-AlexNet of
0.359 over 5 images x 3 seeds (the T-005 settings gave 0.416 and 0.465). The original M1 target (LPIPS-VGG below 0.25 within
2000 strokes) is **not met** on average; the revised target used for the demos is stated in the M1 demos section above.

## Experiments

The baseline study (hypotheses H1 to H7, 5 images x 3 seeds, held-out metric) is in
[docs/experiments/optim_sweep.md](docs/experiments/optim_sweep.md), with the generated tables in
`docs/experiments/optim_sweep_tables.md`. The stored run files live in `outputs/exp_optim` (gitignored, not in the
repository). Regenerating the tables and figures from them takes seconds:

```
.venv\Scripts\python -m scripts.exp_optim_sweep --aggregate-only --figures
```

A full re-run of the sweep takes about 10 hours on an RTX 5070 and is resumable: `.venv\Scripts\python -m scripts.exp_optim_sweep`
(`--stage NAME` runs one stage, `--force` recomputes existing runs). The images are fetched first with
`.venv\Scripts\python -m scripts.fetch_data`; add `--insecure` only behind a TLS-intercepting proxy.

## Animating a painting

`scripts/animate.py` replays a stroke sequence stroke by stroke: each stroke is revealed along its length on a
canvas-textured background, with no brush visible (see [docs/design/animation.md](docs/design/animation.md)). Export
needs **ffmpeg** on the PATH (Windows: `winget install Gyan.FFmpeg`; Linux: `apt install ffmpeg`; macOS: `brew install ffmpeg`),
or set its full path with `ffmpeg=...`. Settings are in `configs/animation.yaml`.

```
.venv\Scripts\python -m scripts.animate --sequence outputs/optim_run/sequence.npz --out outputs/anim
.venv\Scripts\python -m scripts.animate --sequence outputs/optim_run/sequence.npz --out outputs/anim formats=[mp4,webm,gif] duration=8 gif_width=360
```

Each format is written as `<out>/<sequence name>.<format>` (MP4, WebM or GIF), with `config.resolved.yaml` and `timings.json`
(render and encode time, peak VRAM). The video has `round(fps * duration) + round(fps * hold)` frames; the last frame is the
finished painting.

## Checks

Run from the repo root, with the venv interpreter:

```
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m mypy brushstroke
.venv\Scripts\python -m pytest -q                              # all tests (GPU tests auto-skip without CUDA)
.venv\Scripts\python -m pytest -q -m "not gpu and not slow"    # what CI runs
```

On Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

The Paint Transformer parity tests compare against the official code and brush images. They need the official files
(`python -m scripts.pt_reference_study fetch`) and the brush images (`python -m scripts.fetch_pt_brushes`); without them they are
skipped, and each skip says which command to run.

## Paint Transformer compatible mode (M2)

`render.mode: pt` renders the strokes exactly as the official Paint Transformer code does: two brush images chosen by `h > w`,
binarized and eroded alpha, a dilated `gray * colour` foreground, a black canvas and no opacity (the ninth parameter `a` is the
0/1 stroke decision). `render.pt_alpha` selects the inference (`pt_infer`) or training (`pt_train`) pipeline. The default
procedural mode is unchanged. The brush images are not committed: fetch them first with `python -m scripts.fetch_pt_brushes`
(Apache-2.0, see [docs/data_sources.md](docs/data_sources.md)); without them the mode raises an error that names the command.

```python
import torch

from brushstroke.config import RunConfig
from brushstroke.renderer import BrushBank, render
from brushstroke.utils.config import load_config

cfg = load_config(RunConfig, "configs/default.yaml", overrides=["render.mode=pt"])
bank = BrushBank.from_config(cfg.brush)  # ignored in pt mode
params = torch.rand(2, 32, 9)  # read as PT strokes
image = render(params, bank, 256, 256, cfg=cfg.render)  # (2, 3, 256, 256), black canvas
```

The mapping between Paint Transformer's parameters and ours, and the derivation of the mode, are in
[docs/design/paint_transformer.md](docs/design/paint_transformer.md) (sections 1 and 11). The synthetic stroke generator
(`brushstroke.stroke_model.SyntheticStrokeGenerator`) follows the official training data distribution and runs on the GPU.

The model is `brushstroke.stroke_model.PaintTransformer`, the official architecture with the same parameter names, so the
released weights load strictly. It has 9,072,902 trainable parameters:

```python
from brushstroke.stroke_model import PaintTransformer, count_parameters

model = PaintTransformer()
count_parameters(model)  # 9072902
```

The losses (pixel, stroke L1, Wasserstein between Gaussian stroke representations, and confidence BCE with Hungarian matching)
are `PaintTransformerLoss`. Two presets are selectable from `configs/pt/loss_*.yaml`: `official` (the weights of the official code,
the default) and `paper_weights` (the weights of the paper). The paper and code disagree in several places; see
[docs/design/paint_transformer.md](docs/design/paint_transformer.md) (section 9 and section 12).

## Training the Paint Transformer (M2)

`scripts/train_pt.py` trains the model on synthetic strokes generated on the fly, with the official schedule (Adam, lr 1e-4,
betas (0.5, 0.999), batch 64, 56,320 steps = 3.60 M samples, lr x0.3 every 5 epochs at the end). Everything is set in
`configs/stroke_model/pt.yaml`; any key can be overridden on the command line.

```
.venv\Scripts\python -m scripts.train_pt --config configs/stroke_model/pt.yaml
.venv\Scripts\python -m scripts.train_pt --config configs/stroke_model/pt.yaml schedule.max_steps=500 resume=auto precision=bf16
.venv\Scripts\python -m scripts.train_pt --profile precision=bf16
```

It needs the official brush images (`python -m scripts.fetch_pt_brushes`). The outputs go to `outputs/pt_train/` (never
committed): `train.csv` and `val.csv` (losses, the held-out validation set, step times, peak VRAM), `latest.pt`, `summary.json`
and the resolved config. `resume=auto` continues from `latest.pt`; a resumed run is bit-identical to an uninterrupted one on CPU.

Speed on an RTX 5070 with batch 64: T-013 measured 38 ms per step, but the reproduction study (T-016) re-measured **94 ms per step**
alone on an idle GPU, twice, and did not find the cause. At 94 ms the official schedule takes about 1.5 hours (the official code
takes 3.7 to 8.7 hours on the same GPU). Peak VRAM is about 0.85 GB in fp32. Three seeds of the official schedule took 83 minutes each when three
ran at the same time on the GPU.

**Results (T-014).** The official schedule is not fully flat before the learning-rate decay (0.5 to 1.05 % per 10 % of the run),
so the recipe was also run stretched to 112,640 steps (same shape, decay at 92 % of the run); all three stretched seeds meet the
plateau criterion. On a common 4096-patch held-out set the stretched seeds 1 and 2 reach a total loss of 3.550 and 3.547, about
8 % below the official pretrained weights (3.864). The step-by-step results, the official-code reference run and the findings are
in [docs/experiments/pt_training.md](docs/experiments/pt_training.md); the tables and figures regenerate with
`.venv\Scripts\python -m scripts.pt_training_report report` from the logs in `docs/experiments/data/` (no GPU needed).

**Use `|h|`, not `h`, downstream.** Seed 0 learned a negative stroke height for every stroke (the loss and the renderer accept
it, so nothing penalises it), and its loss is higher partly for that reason (with `|h|` it is still the highest of the three seeds). The checkpoint selected for the next steps
(`outputs/pt_long/seed2/latest.pt`) has no such sign problem. Code that reads the predicted height must take `|h|`.
Checkpoints are kept outside the repository.

**Convention options (T-019, off by default).** The sign and the angle of the stroke parameters have equivalent forms: a stroke with
`(w, h, theta + 1)` is the stroke `(-w, -h, theta)`, exactly; the mirror `h -> -h` is only an approximation (it changes about 14 % of the
stroke area). Three options are behind flags: `model.convention=positive` (A, softplus on `w` and `h`), `loss.l1_invariance=mirror` (B)
and `loss.lambda_sign=10` (C, a penalty on negative sizes). On the official recipe, A and C remove the outlier seeds at no cost to the
held-out loss. On the stretched recipe, A is unstable: two of three seeds diverge in the second half of training, so no option is
validated there. The metric to report from now on is the mirror-invariant `D_L1`
([docs/experiments/pt_training.md](docs/experiments/pt_training.md), section T-019; the design is in
[docs/design/pt_convention.md](docs/design/pt_convention.md)). Regenerate with `.venv\Scripts\python -m scripts.exp_pt_convention report` (no GPU).

Hugging Face upload is prepared but not run: `scripts/upload_pt_to_hub.py` builds the staging folder with the weights, the
config hash and a model card (MIT), and only uploads with `--push`.

Details, the step profile and the validation curves are in [docs/design/paint_transformer.md](docs/design/paint_transformer.md)
(section 13), [docs/media/pt_train_curve.png](docs/media/pt_train_curve.png) and the two curve figures in
[docs/media/](docs/media/) (`pt_training_official_recipe.png`, `pt_training_long.png`).

## Painting with the Paint Transformer (M2)

`scripts/paint_pt.py` paints an image with the multi-scale Paint Transformer inference: the official procedure (layers of
decreasing patch size, four-phase windows, a shifted final layer), producing a `StrokeSequence`. The settings are in
`configs/paint_pt.yaml`, and the checkpoint is passed on the command line (the released `model.pth` also works).

```
.venv\Scripts\python -m scripts.paint_pt --input data/raw/mona_lisa.jpg --out outputs/pt_run --checkpoint outputs/pt_long/seed2/latest.pt
.venv\Scripts\python -m scripts.paint_pt --input data/raw/mona_lisa.jpg --out outputs/pt_refined --checkpoint outputs/pt_long/seed2/latest.pt refine.enabled=true
```

The output directory gets `sequence.npz`, `final.png`, `variants/final_official.png` and `variants/final_gated.png`,
`progress/layer_XX.png`, `target.png`, `config.resolved.yaml` and `metrics.json`. `inference.final_layer` chooses which variant
goes into `sequence.npz`: `official` paints every stroke of the last layer, `gated` only those the network is confident about.

What is verified (details in [docs/experiments/pt_inference.md](docs/experiments/pt_inference.md)):

- **Exact reproduction**: with the official weights, our output equals the official `inference.py` bit for bit (maximum absolute
  difference 0.0) on the five evaluation images at 512 x 512 and on one image at 1024 x 1024.
- **Stroke counts and cost**: 3,787 to 4,991 strokes per 512 px image with the official weights. One image takes about 0.23 s and
  1.05 GB of peak GPU memory on the RTX 5070.
- **The sequence reproduces the painting**: `replay_sequence` re-renders the exported sequence to the pipeline canvas up to float
  rounding. The generic `StrokeSequence` render in PT mode is close but not identical; use the layer-aware replay for exact output.
- **Per-patch refinement** (`refine.enabled=true`, off by default): 100 optimization steps per layer lower the L1 error by 27 %
  (Mona Lisa) and 35 % (The Great Wave), and the held-out LPIPS-AlexNet by 0.06 to 0.08. This is measured on two images only.

**Two caveats.** First, the T-014 checkpoint has a lower loss on the synthetic validation data, but on the five real images it is
not better than the official weights (mean L1 0.071 against 0.067, LPIPS-AlexNet 0.496 against 0.496). The synthetic loss is not a
good guide to real-image quality. Second, some models predict negative stroke heights; the pipeline takes `|h|` right after the
network (`inference.abs_h: true`, on by default), so the sequence and every downstream consumer only see positive heights. For
checkpoints that predict negative heights this can change the painted result slightly, since the brush is mirrored.

The forced strokes of the last layer are nearly useless: dropping them costs little quality and removes 15 to 18 % of the
strokes. Which variant should be the default for animations is still open (see the design note, section 14).

## Reproducing the Paint Transformer results (M2)

The reproduction study (T-016) fixed its protocol before running anything; the results, the verdicts and the gap analysis are in
[docs/experiments/pt_reproduction.md](docs/experiments/pt_reproduction.md). Summary, against the paper's own tables and figures:

| item | verdict |
|------|---------|
| Table 1, real images (PT against the Optim baseline) | **not reproduced**: our optimization baseline is stronger than PT on every real-image metric (L_pixel 0.050 against 0.073) |
| Table 1, stroke metrics | **partly**: the stretched recipe passes every criterion; the official recipe is 32 % above the paper on D_W |
| Table 1, ablations (per term, 3 seeds each) | **L_pixel matters** (removing it raises the pixel metrics by 1.2 to 1.4x, the paper 1.25 to 1.6x); **D_W partly** (only its own metric and D_L1 move); **D_L1 and D_bce not reproduced**: removing D_bce slightly *improves* the real-image pixel loss (0.95x). The ordering agrees only in coarse form (Kendall tau +0.33), and the effects are smaller than the paper's; the paper's loss weights were not tried |
| Table 2, time ratios | **partly**: ratios within a factor of 2, but PT's time grows 1.5x from 128 to 512 px, against 5.5x in the paper |
| Table 2, training time | **reproduced (better)**: 1.48 h for the official schedule, against 3.8 to 8.7 h for the official code on the same GPU |
| Figures 5 and 7 | **reproduced** |
| Figure 6 | **partly**: only the pixel-loss ablation is visibly worse |

Two caveats set the scale of these numbers. The absolute value of the perceptual loss depends on a layer choice that the paper does
not publish, so only orderings and ratios are compared. Some models predict negative stroke parameters; the convention study (T-019)
found that the `|h|` fold alone is a diagnostic rather than an exact symmetry, and that the mirror-invariant `D_L1` is the metric to
read. The conclusion for the project, on our five images: a good
optimization baseline beats the Paint Transformer on fidelity, and the argument for the Paint Transformer is speed.

The two tables regenerate from the stored data with no GPU:

```
.venv\Scripts\python -m scripts.exp_pt_table1 report
.venv\Scripts\python -m scripts.exp_pt_table2 report
```

The three figures (`scripts/exp_pt_figures.py`) run the models and need a GPU and the checkpoints in `outputs/`, which are not in the
repository.

## License

Code: MIT (see [LICENSE](LICENSE)). Third-party components are not redistributed in this repository; their licenses are:

| component | used for | license |
|-----------|----------|---------|
| PyTorch 2.11, torchvision 0.26 | tensors, rendering, VGG/AlexNet code | BSD-3-Clause |
| NumPy, SciPy, OmegaConf, LPIPS (code and linear heads) | arrays, config, perceptual metric | BSD (3-Clause; LPIPS 2-Clause) |
| Pillow, PyYAML, tqdm, matplotlib | image I/O, YAML, progress, figures | MIT-CMU, MIT, MPL-2.0/MIT, PSF-based |
| pytest, ruff, mypy | development | MIT |
| ffmpeg (external executable, not bundled) | MP4/WebM/GIF export | LGPL/GPL depending on the build you install |
| VGG16 and AlexNet ImageNet weights | LPIPS backbones, downloaded at run time into the torch cache | torchvision pretrained weights (trained on ImageNet; subject to the ImageNet terms of use) |
| evaluation and demo images | experiments and demos | public domain / NASA, per image in [docs/data_sources.md](docs/data_sources.md) |
| Paint Transformer official code (Apache-2.0) and pretrained weights (no license stated) | study only, a reference point outside git; the code is not vendored | Apache-2.0; weights are never redistributed (see [docs/data_sources.md](docs/data_sources.md)) |

The demo GIFs in `docs/media/` are paintings generated from the public-domain inputs above. Model weights and datasets are never committed.
