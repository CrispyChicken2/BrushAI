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
  renderer/           differentiable stroke renderer (M1)
  stroke_model/       stroke predictor (M2)
  contribution/       semantic allocation and painterly ordering (M4)
  animation/          stroke-by-stroke replay and export (M1)
  generation/         text/image front-ends (M5)
  eval/               metrics and evaluation harness (M3)
  utils/              config (YAML + dot-overrides), seeding, device/VRAM helpers, image I/O
configs/              YAML run configurations (every hyperparameter lives here)
scripts/              entry points, one per experiment
tests/                pytest suite
docs/                 plan/ (milestones), design/ (component notes), media/ (optimized demo GIFs)
report/               technical report
```

## Configuration

Run settings are typed dataclasses loaded from YAML in `configs/`, with `key.sub=value` overrides from the
command line. The YAML values override the dataclass defaults; unknown keys and wrong types raise an error.

```python
from dataclasses import dataclass, field

from brushstroke.utils.config import dump_config, load_config


@dataclass
class OptimCfg:
    lr: float = 0.01


@dataclass
class RunCfg:
    seed: int = 0
    deterministic: bool = False
    device: str = "auto"
    output_dir: str = "outputs/default"
    optim: OptimCfg = field(default_factory=OptimCfg)


cfg = load_config(RunCfg, "configs/default.yaml", overrides=["optim.lr=0.05"])
dump_config(cfg, cfg.output_dir)  # writes outputs/default/config.resolved.yaml
```

Seeding for reproducible runs: `brushstroke.utils.seed.seed_everything(seed, deterministic=...)` (call it before any
CUDA work). Peak GPU memory is measured with `brushstroke.utils.device.reset_peak_vram()` and `peak_vram_mb()`.

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
