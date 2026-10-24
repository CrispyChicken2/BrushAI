# M2 retrospective and gate audit (T-018)

Scope of M2: Paint Transformer reimplementation (PT-compatible renderer and synthetic data T-011, model and losses T-012,
training pipeline T-013, full runs T-014, multi-scale inference T-015), reproduction of the paper results and gap analysis (T-016),
loss ablations (T-017), demo and gate (T-018); [plan](M2.md). Like the M1 review, this document audits M2 as an outsider
would: a fresh clone of the branch `agents/018` (commit `bc1fea3`, which contains everything of M2 and no code change after the
audit except this document), setup from the README only, all checks, regeneration of a table and of the demos, then every success
criterion.

## 1. Fresh-clone reproduction

Machine: Windows 11, Python 3.11, RTX 5070 (12 GB), torch `2.11.0+cu128`. The clone was made with `git clone` of the local repository into
`outputs/fresh_clone/` (no remote), with a new venv inside it; the audit script is a PowerShell transcript of the README commands.

| step (README commands, in order) | result |
|---|---|
| `git clone` | 200 tracked files |
| `py -3.11 -m venv .venv`, `pip install -U pip`, `pip install -c constraints.txt -r requirements.txt -r requirements-dev.txt` | OK after the README troubleshooting entry (`pip.ini` with `trusted-host`, TLS-intercepting proxy on this machine; **the first attempt without it failed with `CERTIFICATE_VERIFY_FAILED`**, as documented in M1); install 130 s (pip cache) |
| `ruff check .` / `ruff format --check .` / `mypy brushstroke` | all green |
| `pytest -q` **before** any data download | **237 passed, 53 skipped** (the skipped tests need the official PT code and brush images and say how to fetch them) |
| `python -m scripts.fetch_data`, `pt_reference_study fetch`, `fetch_pt_brushes` (each with `--insecure` behind the proxy) | OK (14 s); checksums verified |
| `pytest -q` **with** the data | **290 passed** (151 s): the official-parity tests run and pass |
| regenerate the tables from the stored data (no GPU): `exp_pt_table1 report`, `exp_pt_table2 report`, `exp_pt_ablation_verdict report`, `pt_inference_study report`, `pt_training_report report` | `git diff` empty after the five commands: every committed table and figure block is reproduced from the stored JSON (on Windows `git status` lists `docs/experiments/pt_reproduction.md` as modified because of a line-ending rewrite only, with an empty content diff) |
| `python -m scripts.make_m2_demos --insecure` (committed `m2_*` removed first; our checkpoint copied into the clone; the M1 baselines painted from scratch in the clone) | exit 0 in 31 min (the three M1 baselines painted in 9.8 min each). The 6 GIFs and 3 JPEGs are byte-identical to the committed ones; `m2_results.md` differs only in the measured M1 runtime label (9.8 min against 9.4 min committed: wall-clock of the machine, not reproducible; the committed value is the original run's) |
| `python -m scripts.make_m2_demos --checkpoint data/raw/pt_official/model.pth --only portrait` (released weights, no training needed) | exit 0 in 30 s: `m2_portrait.gif` 0.59 MB, `m2_portrait_vs_m1.gif` 1.11 MB, compare image 87 KB |

### Friction points found, and what was done
| # | friction | fix |
|---|---|---|
| 1 | The M2 demos need the T-014 checkpoint (`outputs/pt_long/seed2/latest.pt`), which is not committed (about 100 MB, kept out of the repository); a fresh clone cannot run `make_m2_demos` with the default config | the README and the script docstring now give the alternative `--checkpoint data/raw/pt_official/model.pth` (released weights, fetched by `pt_reference_study fetch`); retraining the checkpoint is `python -m scripts.train_pt` (T-013, about 3 h) |
| 2 | The M2 demos paint the three M1 baselines first when `outputs/m1/` is empty (about 28 min, GPU) | stated in the README and in the script docstring |
| 3 | TLS-intercepting proxy: every download needs `--insecure`, pip needs the `pip.ini` | already in the README (M1), re-verified |
| 4 | The first version of the M2 results table contained the PT runtime in seconds (0.2 s to 0.8 s between reruns), which dirtied the tracked file on a rerun (the same issue as M1) | coarse runtime labels (`< 1 s`, `9.4 min`), tested |
| 5 | The README said "M2 is under way" | updated |

No code bug was found in the fresh clone.

## 2. Success criteria of the M2 plan

Criteria 2 and 3 are the versions refined in [M2.md](M2.md) (refinements 9 and 10, accepted).

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | Trained model converges (validation-loss plateau documented, curves committed as small figures) | **met** | 3 seeds of the official recipe (56,320 steps) and of a stretched one (112,640 steps): all three stretched seeds meet the pre-registered plateau criterion (pre-decay gain < 1 %), one official-recipe seed does not (1.05 %); curves `docs/media/pt_training_official_recipe.png`, `pt_training_long.png` (< 100 KB); our seed 0 and the official code trained with seed 0 end at the same loss (3.950 vs 3.944) and make the same sign choice ([pt_training.md](../experiments/pt_training.md)). Caveat: a sign convention problem (`h < 0` on seed 0) is handled by `|h|` downstream |
| 2 | Inference produces a full painting at 512² with a stroke count in the range of the released model (3.8 to 5.0 k at 512²) | **met** | with the official weights our inference is bit-identical to the official `inference.py` (maximum difference 0.0, five images at 512 and one at 1024); with our checkpoint the three demo paintings have 3,866 / 5,020 / 4,351 strokes; about 0.2 s and 1.05 GB per image ([pt_inference.md](../experiments/pt_inference.md), [m2_results.md](../media/m2_results.md)) |
| 3 | Every reproducible paper number is reproduced, or the gap is quantified and explained (T1b, T1c, T2a, T2b, Fig. 5/6/7; T1a absolute numbers are not reproducible) | **met in form, mixed in substance** | all items are measured, with pre-registered tolerances and every gap quantified ([pt_reproduction.md](../experiments/pt_reproduction.md)): T1b partly (the stretched recipe passes every tolerance, the official recipe misses D_W by +32 %, the released weights by +33 %), T1c mostly not reproduced (7 of 20 cells; the ablation ordering agrees with the paper only coarsely, Kendall tau +0.33; removing D_bce *improves* the real-image metrics), T1a ordering inverted (our optimization baseline is stronger than the paper's), T2a ratios reproduced but PT growth 1.5x vs 5.5x, T2b better than the official code (1.5 h vs 3.8 to 8.7 h), Fig. 5 and 7 reproduced, Fig. 6 partly. Several explanations are **hypotheses, marked untested** (loss weights of the code vs the paper, the paper's evaluation protocol, the mechanism of the D_bce effect, the 8 % gap to the released weights on real images) |
| 4 | Reproducible from one command per table | **met** | `exp_pt_table1 report`, `exp_pt_table2 report`, `exp_pt_ablation_verdict report` regenerate the committed blocks byte for byte from the stored JSON, without GPU (section 1); the measurement stages are separate commands (GPU) |
| 5 | Demo GIF from the PT pipeline via our animation module | **met** | `python -m scripts.make_m2_demos`: 3 PT GIFs and 3 side-by-sides with the M1 baseline, 0.67 to 1.28 MB each; byte-identical on a rerun and in the fresh clone (the M1 runtime label of the results table excepted: 9.8 min against 9.4 min). The animation draws the PT strokes with a new `PTFrameRenderer` (the animation module only knew the procedural brushes): its last frame differs from the pipeline canvas by 0.005 to 0.009 mean absolute difference (generic render, [paint_transformer.md](../design/paint_transformer.md) section 14) |

**Gate verdict.** All five criteria are met as written; criterion 3 is met because every gap is quantified, but the substance is that
**the paper's real-image and ablation claims mostly do not reproduce on our setup** (explanations are partly hypotheses). The deliverable
that M3 needs is solid: a trained, exactly-compatible stroke predictor and a fast inference pipeline. The Paint Transformer paints in
under a second what the optimization baseline needs 9.4 minutes for, at lower fidelity (LPIPS-VGG 0.53 to 0.65 against 0.25 to 0.36).

## 3. Repository hygiene

| check | result |
|---|---|
| files in git | 200 at the audited commit; the largest are the media files: no file larger than 1.3 MB (the GIFs), no blob > 4 MB in the history (`git rev-list --objects --all` size check, see the handoff) |
| weights / datasets | none tracked: no `*.pt, *.pth, *.ckpt, *.safetensors, *.bin, *.npz`; checkpoints in `outputs/` (git-ignored); the official code and brush PNGs are fetched into `data/raw/` (ignored, pinned by SHA-256, Apache-2.0) |
| secrets | `git grep` for key / token / password patterns: no hit |
| media | 6 GIFs (m2) + 3 (m1) <= 1.4 MB, JPEG and PNG figures <= 100 KB each |
| licenses | code MIT; the official Paint Transformer code and brush images are Apache-2.0, downloaded not committed; the README license table and `docs/data_sources.md` cover them |

## 4. Measured cost (RTX 5070, 12 GB)

| stage | measured |
|---|---|
| model | 9,072,902 parameters (equal to the official model; official `state_dict` loads and reproduces its outputs) |
| training, one run alone | **94 ms per step, 1.5 h** for the official recipe, about 3 h stretched; peak 0.85 GB (fp32). The T-013 measurement of 38 ms was **not reproduced** (corrected in T-016). Three runs sharing the GPU: about 88 ms per step each (83 min for 56,320 steps) |
| the official code on the same GPU | 241 ms per step in the T-010 benchmark (3.8 h), 555 ms in the full retrain (8.7 h) |
| T-014: 3 seeds x 2 recipes + official-code retrain | official recipe 83 min and stretched recipe 181 min for each batch of three runs in parallel; the official code retrain 8.7 h |
| T-016/T-017: 12 ablation trainings | 9.1 h of wall clock (3 in parallel, partly contended with the Optim timing) |
| inference, 512², one image | 0.23 s (T-015), 1.05 GB; 0.15 / 0.20 / 0.23 s at 128 / 256 / 512 px (launch-bound at small sizes: growth 1.5x against 5.5x in the paper) |
| optimization baseline (M1), same hardware | 150 / 155 / 587 s at 128 / 256 / 512 px: PT is 770 to 2,500 times faster |
| animation of a PT sequence, 12 s at 15 fps, 480 px | a few seconds per GIF |
| test suite | 290 tests in 151 s with the data, 237 + 53 skipped in 40 s without |

## 5. Known limitations and open issues

1. **Fidelity.** On our three demo inputs the PT paintings are clearly below the optimization baseline (L1 0.036 / 0.091 / 0.074 against 0.024 / 0.071 / 0.055; LPIPS-VGG 0.54 / 0.65 / 0.53 against 0.25 / 0.36 / 0.25) and leave unpainted black patches on dark regions (the official procedure does not cover a patch without confident stroke). PT's advantage is time, not quality.
2. **8 to 9 % behind the released weights** on real-image L_pixel although the stroke metrics are equal or better; not explained (T-016).
3. **Stroke-parameter convention.** Some runs learn negative `h` (seed 0 of the full model, 4 of 12 ablations) or negative `w`: the renderer accepts it, only D_L1 sees it; downstream code must use `|h|` (done in the inference), and `w` is not folded.
4. **Ablation findings.** The paper's ranking of the loss terms is only coarsely reproduced; D_L1 and D_bce do not matter in our recipe, and D_bce hurts the real-image L_pixel. The ablations use the code's loss weights, not the paper's (12 more trainings needed).
5. **Small-size inference is launch-bound** (host synchronizations in `paint_layer`).
6. **The T-014 checkpoint is not published** (a Hugging Face upload is prepared in `scripts/upload_pt_to_hub.py`, not run); the demos need it or the released weights.
7. **GPU required for the demos**; GPU results are bit-reproducible only on the same GPU/driver/torch build (verified on this machine).
8. **L_pcpt** is defined by us (relu3_3 of VGG16): its scale is 20x the paper's, so only orderings are comparable. T1a absolute numbers cannot be reproduced (datasets unavailable).
9. **Animation of PT strokes** uses the generic render (brush resized to the canvas), not the exact window replay of the pipeline (differences of 0.5 to 0.9 % mean absolute).
10. **Platform**: Windows only was exercised; the Linux path is documented only.

## 6. Proposed tickets for M3 and later

| # | ticket | milestone | rationale |
|---|---|---|---|
| 1 | PT as initializer + per-patch refinement of the M1 optimizer (T-015 refinement already reduces L1 by 27 to 35 %): quality/time trade-off curve | M3 | PT gives 4 k strokes in 0.2 s, the optimizer closes the fidelity gap |
| 2 | Evaluation harness with held-out metrics, multi-image set, anytime-quality curve (LPIPS vs stroke count) over M1, PT and RL | M3 | the contribution is evaluated by it; reuses `render_progressive` and the T-016 metrics |
| 3 | Learning-to-Paint (RL) baseline, checked against published numbers | M3 | baseline (c); the paper's RL row is still copied |
| 4 | Ablations with the paper's loss weights and matching cost (3 seeds x 4) | M2 follow-up | tests the main hypothesis for the missing ablation effects (T-016 item 4) |
| 5 | A fix for the (w, h, theta) parameter convention (loss or reparameterization) and a convention-invariant D_L1 | M2 follow-up | removes the seed-dependent sign outliers; makes D_L1 meaningful |
| 6 | Reduce host synchronizations in `paint_layer` | tooling | 2 to 3x at 128 px |
| 7 | Investigate the 94 ms vs 38 ms training-step discrepancy and the 8 % real-image gap to the released weights | M2 follow-up | two unexplained observations |
| 8 | Publish the checkpoint (Hugging Face upload already staged) and add a CPU/Linux path or CI job for one tiny PT painting | tooling | lets others run the demos without training |
| 9 | Semantic-aware stroke allocation on top of PT (the project's contribution): allocate patches/strokes by saliency, cover dark regions | M4 | uses PT's layers; fixes the unpainted dark patches |
