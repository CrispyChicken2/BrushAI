# BrushStroke

Neural stroke-based painting: turn a text prompt or an image into a painting made of an ordered sequence
of brush strokes, replayed as a stroke-by-stroke animation. The research focus is semantic-aware stroke
allocation, painterly stroke ordering and an "anytime quality" metric.

**Status: M1 in progress** (see [docs/plan/M1.md](docs/plan/M1.md)).

## Setup

Requires Python 3.11. The pinned PyTorch build comes from the CUDA 12.8 index, which an NVIDIA RTX 50-series
GPU (Blackwell, sm_120) needs.

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

Linux:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

`requirements.txt` already contains `--extra-index-url https://download.pytorch.org/whl/cu128`.
For a CPU-only machine (as in CI), install torch first from the CPU index:
`pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu`,
then the two requirements files.

Do not `pip install -e .`: run scripts from the repo root (`python -m scripts.<name>`); pytest uses
`pythonpath = ["."]`.

## Checks

```
python -m ruff check .
python -m ruff format --check .
python -m mypy brushstroke
python -m pytest -q                         # all tests (GPU tests auto-skip without CUDA)
python -m pytest -q -m "not gpu and not slow"   # what CI runs
```

## License

Code: MIT (see [LICENSE](LICENSE)). Pretrained models and datasets used by the project have their own
licenses; they will be listed here as they are introduced.
