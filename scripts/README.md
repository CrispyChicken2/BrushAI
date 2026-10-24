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
| `fetch_pt_brushes` | download the four official Paint Transformer brush images (Apache-2.0) and verify their checksums (T-011) |
| `train_pt` | train the Paint Transformer stroke predictor: official schedule, validation, resume, bf16 option, step profile (T-013) |
| `paint_pt` | paint an image with the multi-scale Paint Transformer inference: exact official procedure, `StrokeSequence` output, optional per-patch refinement (T-015) |
| `pt_inference_study` | parity, variants, refinement and layer measurements of the inference pipeline; `report` regenerates `docs/experiments/pt_inference.md` without a GPU (T-015) |
| `exp_pt_table1` | Table 1 of the Paint Transformer reproduction: stages `strokes`, `real`, `optim`; `report` regenerates the table block of `docs/experiments/pt_reproduction.md` from the stored JSON, no GPU (T-016) |
| `exp_pt_table2` | Table 2 (inference and training time): stages `run`, `training` (re-times the training step), `report` regenerates its block from the stored JSON, no GPU (T-016) |
| `exp_pt_ablations` | the 12 loss-ablation trainings (4 terms x 3 seeds, 56,320 steps each, `--parallel` runs share the GPU); needs the GPU (T-016) |
| `exp_pt_figures` | Figures 5 to 7 of the paper on our models; runs the models, so it needs a GPU and the checkpoints in `outputs/` (T-016) |
| `pt_training_report` | evaluate, archive and regenerate the tables and figures of the Paint Transformer training runs from the stored logs (`report` needs no GPU) (T-014) |
| `pt_official_train` | train the official Paint Transformer code for a direct comparison, outside the package (T-014) |
| `upload_pt_to_hub` | stage the trained weights, config hash and model card for Hugging Face; uploads only with `--push` (T-014) |
| `pt_reference_study` | study of the official Paint Transformer code and weights (`fetch`, `reference`, `datastats`, `trainspeed`): downloads and runs third-party code, outside git (T-010) |
