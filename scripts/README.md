# scripts

Entry points, run from the repo root: `python -m scripts.<name>` or `python scripts/<name>.py`.

| Script | Purpose |
|--------|---------|
| `preview_brushes` | contact sheet of the brush masks and the canvas texture (T-002) |
| `bench_renderer` | renderer time and peak VRAM table (T-003) |
| `paint_optim` | paint an image with baseline (a) (T-005) |
| `animate` | stroke-by-stroke animation of a `StrokeSequence` to MP4 / WebM / GIF (T-006) |
| `fetch_data` | download the evaluation images and verify their checksums (T-009) |
| `exp_optim_sweep` | the baseline sweep: runs and regenerates `docs/experiments/` tables and figures (T-009) |
