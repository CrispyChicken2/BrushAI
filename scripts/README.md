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
| `make_m2_demos` | the M2 deliverable: Paint Transformer paintings of the three demo inputs, animated, side by side with the M1 baseline, write `docs/media/m2_*` (T-018) |
| `exp_optim_sweep` | the baseline sweep: runs and regenerates `docs/experiments/` tables and figures (T-009) |
| `fetch_pt_brushes` | download the four official Paint Transformer brush images (Apache-2.0) and verify their checksums (T-011) |
| `train_pt` | train the Paint Transformer stroke predictor: official schedule, validation, resume, bf16 option, step profile (T-013) |
| `paint_pt` | paint an image with the multi-scale Paint Transformer inference: exact official procedure, `StrokeSequence` output, optional per-patch refinement (T-015) |
| `pt_inference_study` | parity, variants, refinement and layer measurements of the inference pipeline; `report` regenerates `docs/experiments/pt_inference.md` without a GPU (T-015) |
| `exp_pt_table1` | Table 1 of the Paint Transformer reproduction: stages `strokes`, `real`, `optim`; `report` regenerates the table block of `docs/experiments/pt_reproduction.md` from the stored JSON, no GPU (T-016) |
| `exp_pt_table2` | Table 2 (inference and training time): stages `run`, `training` (re-times the training step), `report` regenerates its block from the stored JSON, no GPU (T-016) |
| `exp_pt_ablations` | the 12 loss-ablation trainings (4 terms x 3 seeds, 56,320 steps each, `--parallel` runs share the GPU); needs the GPU (T-016) |
| `exp_pt_convention` | stroke-convention study (T-019): `symmetry` (equivalence classes, verified by rendering), `train` (options A, B, C; the stretched recipe for the best one), `evaluate`, and `report` (no GPU), which regenerates the T-019 blocks of `docs/experiments/pt_training.md` and of `pt_reproduction.md` |
| `exp_pt_bn_validation` | validation BatchNorm diagnostic (T-020): `run` (8000 steps, every mode scored in one run), `bitwise` (weights identical across modes, CPU), `report` (block of `pt_training.md`) |
| `pt_convention_report` | the module behind the T-019 report (same output; `report` needs no GPU) (T-019) |
| `exp_pt_paper_weights` | the five ablations with the paper's loss weights and BCE matching cost (T-021): `train` (15 runs), `evaluate`, `report` (block of `pt_reproduction.md`, no GPU) |
| `exp_pt_instability` | instability of convention option A on the stretched recipe (T-023): `trace` (per-step diagnostics), `branch` (single-factor reruns), `report` (block `T023` of `pt_training.md`; needs the per-run `train.csv`, `trace.csv` and `summary.json` of the runs in `outputs/`, which are not in git, so it does not regenerate from the repository alone) |
| `build_eval_set`, `fetch_eval_set`, `generate_eval_sdxl` | the 100-image evaluation set (T-024): seeded curation (`curate`, `sheets`, `check`, `card`), reproducible fetch with SHA-256 verification, SDXL generation (GPU) with hash verification |
| `audit_eval_set` | audit of the evaluation set (T-034): every machine-checkable rule on every prepared image, report `audit.json`, exit 1 on any violation (run after `fetch_eval_set`) |
| `exp_pt_ablation_verdict` | per-term verdicts and the ordering of the loss terms against the paper, from the stored ablation evaluations; `report` rewrites the `ABLATION_VERDICT` block of `docs/experiments/pt_reproduction.md`, no GPU (T-017) |
| `exp_pt_figures` | Figures 5 to 7 of the paper on our models; runs the models, so it needs a GPU and the checkpoints in `outputs/` (T-016) |
| `pt_training_report` | evaluate, archive and regenerate the tables and figures of the Paint Transformer training runs from the stored logs (`report` needs no GPU) (T-014) |
| `pt_official_train` | train the official Paint Transformer code for a direct comparison, outside the package (T-014) |
| `upload_pt_to_hub` | stage the trained weights, config hash and model card for Hugging Face; uploads only with `--push` (T-014) |
| `pt_reference_study` | study of the official Paint Transformer code and weights (`fetch`, `reference`, `datastats`, `trainspeed`): downloads and runs third-party code, outside git (T-010) |
