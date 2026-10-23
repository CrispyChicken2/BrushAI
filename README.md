# BrushStroke

Neural stroke-based painting: turn a text prompt or an image into a painting made of an ordered sequence
of brush strokes, replayed as a stroke-by-stroke animation. The research focus is semantic-aware stroke
allocation, painterly stroke ordering and an "anytime quality" metric.

**Status: M1 in progress** (see [docs/plan/M1.md](docs/plan/M1.md)). Documentation index: [docs/README.md](docs/README.md).
Changes per ticket: [CHANGELOG.md](CHANGELOG.md).

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

## Layout

```
brushstroke/          Python package
  renderer/           brush masks and canvas texture (T-002); differentiable batched stroke renderer (T-003)
  stroke_model/       stroke predictor (M2)
  contribution/       semantic allocation and painterly ordering (M4)
  animation/          stroke-by-stroke replay with partial reveal, ffmpeg export (T-006)
  generation/         text/image front-ends (M5)
  eval/               metrics and evaluation harness (M3)
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
command line. The root schema is `brushstroke.config.RunConfig` (sections `brush:` and `canvas:` so far; the
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
0.359 over 5 images x 3 seeds (the T-005 settings gave 0.416 and 0.465). The M1 target (LPIPS-VGG below 0.25 within 2000
strokes) is **not met**; a revised target is proposed in the experiment report, pending the chef's decision.

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

CI (`.github/workflows/ci.yml`) runs on Ubuntu with Python 3.11 and CPU torch: ruff check, ruff format check,
mypy and `pytest -m "not gpu and not slow"`.

## License

Code: MIT (see [LICENSE](LICENSE)). Pretrained models and datasets used by the project have their own
licenses; they will be listed here as they are introduced.
