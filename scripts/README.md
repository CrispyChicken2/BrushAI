# scripts

Entry points, run from the repo root: `python -m scripts.<name>` or `python scripts/<name>.py`.

| Script | Purpose |
|--------|---------|
| `preview_brushes` | contact sheet of the brush masks and the canvas texture (T-002) |
| `bench_renderer` | renderer time and peak VRAM table (T-003) |
| `paint_optim` | paint an image with baseline (a) (T-005) |
| `animate` | stroke-by-stroke animation of a `StrokeSequence` to MP4 / WebM / GIF (T-006) |
| `fetch_data` | download the evaluation images and verify their checksums (T-009) |
| `fetch_demo_inputs` | download the three M1 demo inputs (a subset of the T-009 manifest) and verify their checksums (T-007) |
| `make_m1_demos` | the M1 deliverable: fetch, paint and animate the three demos, write `docs/media/` (T-007) |
| `exp_optim_sweep` | the baseline sweep: runs and regenerates `docs/experiments/` tables and figures (T-009) |
| `pt_reference_study` | study of the official Paint Transformer code and weights (`fetch`, `reference`, `datastats`, `trainspeed`): downloads and runs third-party code, outside git (T-010) |
