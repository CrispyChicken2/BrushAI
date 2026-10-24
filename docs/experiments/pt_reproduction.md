# Paint Transformer: reproduction of the paper results and gap analysis (T-016)

**Hypothesis.** Our reimplementation (T-011 to T-015) reproduces the reproducible results of Liu et al. (ICCV 2021) listed in the feasibility table of
[`docs/design/paint_transformer.md`](../design/paint_transformer.md) section 7 (Table 1 stroke part and ablations, Table 2 times, Fig. 5 to 7) within the tolerances
fixed below; every gap is quantified and explained, starting from the paper-versus-code list of section 9.

The **Protocol** section was written and committed **before any run of this ticket**; the results sections are appended after. Where a choice had to be made because the paper does not specify it, the choice and its reason are stated
here and were not changed after seeing results.

## Protocol

### Rows and what is compared
- **paper**: the numbers printed in the paper (Tables 1 and 2), as extracted in section 7 of the study note.
- **official model**: the released pretrained weights run through our (bit-identical, T-015) pipeline on our machine: the reference point, not a target.
- **ours**: our trained models, **3 seeds, mean ± std** (std over seeds, population-corrected `n - 1`): the T-014 runs, *official recipe* (56,320 steps, the budget of the paper and of the ablations below). The stretched 112,640-step runs
  (T-014 "long") are reported as an extra row. Inference always uses `|h|` (T-015); where `h` matters (stroke distances) both the raw and the `|h|` value are given.
- **Optim**: our optimization baseline (M1, `paint_optim`), the closest thing we have to the paper's "Optim" [39] (Stylized Neural Painting). It is *stronger* than SNP (our renderer, LPIPS term, T-009 tuning), so the paper's ordering between PT and Optim is **not expected to
  reproduce**; this expectation is recorded here before the runs. For Table 1 "same number of strokes as Ours" it is run with 4,261 strokes (levels `[1,2,4,8,16,20]`, strokes per patch `[1,1,1,3,8,5]`, the official PT mean is 4,336); the variant without the LPIPS term (L1 only, the loss of SNP is L1 + OT) is also reported.
- **RL** [10] (Learning to Paint): not available before M3; its columns are copied from the paper and marked n/a for us.

### Metrics
- `L_pixel`: mean absolute error between the painting and the target over RGB pixels in [0, 1] (paper Eq. 7).
- `L_pcpt`: the paper cites the VGG perceptual loss of Johnson et al. (ECCV 2016) without layer or normalization. **Pinned: feature reconstruction loss at `relu3_3` of the torchvision VGG16 (ImageNet weights, the same backbone as LPIPS-VGG), images in [0, 1]
  normalized with the ImageNet mean/std, full 512 x 512 input, squared error averaged over channels and positions** (Johnson et al. use `relu3_3` for their feature loss). The absolute scale of the paper (0.74 to 0.86) cannot be matched with an unspecified definition: the comparison is of orderings and relative changes.
  The held-out LPIPS-Alex (never used in a loss) is reported next to it, as in M1/T-009.
- Stroke metrics on synthesized strokes (paper Table 1 right part): `L_pixel` between the canvas with the predicted strokes painted and the target patch, `D_L1` the parameter L1 over the matched stroke pairs, `D_W` the Gaussian Wasserstein distance over the matched pairs
  (matching = the Hungarian matching of the training loss, official cost), **unweighted** terms of `PaintTransformerLoss` (the training lambdas divided out).

### Data
- **Real images (Table 1 left part, ablations, Fig. 5 to 7):** the five public-domain images of `docs/data_sources.md`, centre-cropped to 512 x 512, `K = 4`, final layer = the official variant (all shifted-layer strokes painted), black initial canvas (official). The paper's 3 x 100 images (Kaggle, WikiArt, FFHQ) are not available:
  **the absolute numbers of Table 1 left (T1a) are not reproducible and are not compared**; five images give per-image numbers and an indication of ordering, not a statistical comparison.
- **Synthesized strokes (T1b, T1c right part):** the paper does not give the generator, the number of images or the settings. **Pinned: the official training distribution** (`set_input`, our bit-equivalent generator of T-011), **4,096 held-out patches (seed 999, the T-014 evaluation set)**, patch = 32 x 32, 8 foreground strokes of which the valid ones are the ground truth,
  the model receives the target patch and the background canvas patch (the training task).
- **Optim on synthesized strokes:** for each patch the strokes the official-weights PT predicts confidently define the number of strokes `n` ("same number of strokes as Ours"); `n` strokes are initialized at random (`x, y ~ U(0.2, 0.8)`, `w, h ~ U(0.2, 0.7)`, `theta ~ U(0, 1)`, colour = target at the centre) and optimized
  (Adam, lr 0.05, 300 iterations, cosine decay) on the patch L1 in the training rendering with the M1 optimizer pieces (`brushstroke.stroke_model.refine`), then scored like the network outputs. This is an assumption: the paper's Optim is a different optimizer on its own renderer.

### Ablations (Table 1 lower part, Fig. 6)
Four extra training runs per seed with one loss term removed, **everything else = the official recipe** (56,320 steps, batch 64, official loss weights and matching): without `L_pixel` (`lambda_pixel = 0`), without `D_L1` (`lambda_gt = 0`), without `D_W` (`lambda_w = 0`), without `D_bce` (`lambda_decision = 0`),
seeds 0, 1, 2 (12 runs). The full model is the T-014 official-recipe run of the same seed. Each ablated model is scored with both protocols above (real images with the inference pipeline, synthesized strokes).

### Evaluation of the ablation runs
**Amendment (after an incident, before the final evaluation):** a run is evaluated only when its checkpoint is at its final step (56,320 or 112,640); an earlier evaluation of partial `latest.pt` files was discarded.

### Table 2
- **T2a, inference time at 128 / 256 / 512 px** (`K` = 2, 3, 4): PT through our pipeline, PT through the official script (time from the end of the model loading to the end of the painting), our optimization baseline (M1 default, 2,037 strokes, scaled to the image size by keeping the same levels). Median of 5 runs per image over the five images
  after 2 warm-up runs for PT, GPU idle (no training running). **Timing scope:** the official script's timer starts at `read_img` (it includes the PNG read and the resize); ours starts from a tensor already on the GPU; the official time is therefore overstated by the I/O (about a few ms), which flatters ours slightly at 512 px and understates our slowness at 128 px; the conclusions do not depend on it. **Amendment before running Table 2:** one run of the optimization baseline takes 2.5 to 10 minutes (measured while preparing the script: 151 s at 128 px, 584 s at 512 px), so it is timed once per size on Mona Lisa (no warm-up needed at these durations). RL: n/a. **Absolute times are not compared** (RTX 5070 against 2080 Ti): the comparison is the Optim / PT ratio and the growth of PT from 128 to 512.
- **T2b, training time (h):** our pipeline (T-013/T-014 measurement), the official training code on our GPU (T-014: 8.68 h for 56,320 steps), the paper (3.79 h). The RL and renderer training costs of the paper are not reproducible and not needed.

### Figures (each <= 500 KB)
Fig. 5: effect of the number of layers `K` (the official procedure applied to the image resized to `32 * 2^K`, K = 1 to 4, shown at 512 px). Fig. 6: the four ablated models and the full model on one image. Fig. 7: the same trained network with the brush images replaced by a rectangle and by a disc
(binary masks of the same size and orientation convention), against the original brush.

### Tolerances (fixed now) and verdict rules
- **T1b (stroke metrics, PT):** reproduced if each of `L_pixel`, `D_L1`, `D_W` of ours (|h|, mean over seeds) is within **±25 %** of the paper's PT value (0.042 / 0.083 / 0.018) and PT beats Optim on all three, as in the paper. A larger deviation is a gap to explain.
- **T1c (ablations):** the paper's claim is directional; per ablated term and per column the verdict is "same direction as the paper (worse than the full model)" or not; the size of each effect (ratio ablated / full) is reported next to the paper's ratio and a ratio within a factor 2 of the paper's counts as the same magnitude.
- **T1a (real, relative):** reproduced if the ordering of `L_pixel` and `L_pcpt` between PT and Optim is the paper's (PT better than Optim); expected to fail, see above. The official model and ours must agree with each other within the seed spread (an observation, not a pass/fail).
- **T2a:** reproduced if the Optim / PT ratio is within a factor 2 of the paper's at each size (1,391 / 1,303 / 1,715) **and** the growth of PT time from 128 to 512 is within a factor 2 of the paper's (5.5x). **T2b:** reproduced if our pipeline's time is of the order of the official code's measured on the same GPU or better (it is expected to be much better, T-013).
- Every comparison keeps negative results. A result outside tolerance is analysed (section "Gap analysis") with the discrepancy list of section 9 of the study note as the first suspects.

### Commands (one per table)
`python -m scripts.exp_pt_table1 {strokes,real,optim,report}`, `python -m scripts.exp_pt_table2 {run,training,report}`, `python -m scripts.exp_pt_figures`, `python -m scripts.exp_pt_ablations --parallel 3`; the ablation trainings are `python -m scripts.train_pt seed=S loss.preset=custom loss.<term>=0 output_dir=outputs/pt_abl/<name>/seedS`.

<!-- BEGIN TABLE2 -->

**T2a, inference time (s)** at 128 / 256 / 512 px (NVIDIA GeForce RTX 5070; paper: 2080 Ti). PT = median of 25 runs (5 images x 5), Optim = one run on Mona Lisa.

| method | 128 | 256 | 512 | growth 128 -> 512 |
|---|---|---|---|---|
| paper PT | 0.055 | 0.124 | 0.304 | 5.5x |
| paper RL | 0.242 | 0.266 | 0.322 | 1.3x |
| paper Optim | 76.5 | 161.6 | 521.4 | 6.8x |
| **ours PT** (official weights) | 0.153 | 0.201 | 0.231 | 1.5x |
| official PT code (same weights) | 0.100 | 0.166 | 0.467 | 4.7x |
| **ours Optim** (M1 baseline, 2,037 strokes) | 149.9 | 155.3 | 587.3 | 3.9x |

| ratio Optim / PT | 128 | 256 | 512 |
|---|---|---|---|
| paper | 1391 | 1303 | 1715 |
| ours (Optim / our PT) | 979 | 773 | 2545 |
| ours (Optim / official PT code) | 1499 | 933 | 1257 |

**T2b, training time of the stroke predictor** (56,320 steps, batch 64):

| | step (ms) | total (h) |
|---|---|---|
| paper PT (2080 Ti) | | 3.79 |
| official code, our GPU, T-010 benchmark | 241 | 3.78 |
| official code, our GPU, full retrain (T-014) | 555 | 8.68 |
| **our pipeline**, one run alone on the GPU | 94.3 | 1.48 |

<!-- END TABLE2 -->

<!-- BEGIN TABLE1 -->

**T1a, real images** (five public-domain images at 512 x 512; paper rows are on different datasets: orderings only; our rows are mean ± std over seeds of the mean over the images):

| row | L_pixel | L_pcpt | LPIPS-Alex | strokes |
|---|---|---|---|---|
| paper RL (own datasets) | 0.04 | 0.737 | n/a | n/a |
| paper Optim (own datasets) | 0.059 | 0.856 | n/a | n/a |
| paper PT (own datasets) | 0.056 | 0.807 | n/a | n/a |
| official model (pretrained weights) | 0.0671 | 15.172 | 0.495 | 4336 |
| **ours**, official recipe, 3 seeds | 0.0726 ± 0.0014 | 16.532 ± 0.400 | 0.521 ± 0.011 | 4589 ± 32 |
| ours, stretched recipe (112,640 steps), 3 seeds | 0.0735 ± 0.0021 | 16.607 ± 0.653 | 0.502 ± 0.011 | 4538 ± 77 |
| ours Optim (M1 baseline, L1 + 0.1 LPIPS), 4,261 strokes | 0.0496 | 8.074 | 0.352 | 4261 |
| ours Optim, L1 only, 4,261 strokes | 0.0396 | 8.408 | 0.355 | 4261 |

**T1b, synthesized strokes** (4,096 held-out patches of the official training distribution):

| row | L_pixel | D_L1 | D_W | painted strokes / patch | matched strokes with h < 0 | with w < 0 |
|---|---|---|---|---|---|---|
| paper RL | 0.058 | n/a | n/a | n/a | n/a | n/a |
| paper Optim | 0.073 | 0.137 | 0.057 | n/a | n/a | n/a |
| paper PT | 0.042 | 0.083 | 0.018 | n/a | n/a | n/a |
| official model (`|w|`, `|h|`) | 0.0410 | 0.0978 | 0.0240 | 7.72 | 0.0 % | 0.0 % |
| official model (`|h|`) | 0.0410 | 0.0978 | 0.0240 | 7.72 | 0.0 % | 0.0 % |
| official model (raw) | 0.0410 | 0.0978 | 0.0240 | 7.72 | 0.0 % | 0.0 % |
| **ours**, official recipe, 3 seeds (`|w|`, `|h|`) | 0.0461 ± 0.0016 | 0.1013 ± 0.0035 | 0.0237 ± 0.0010 | 7.68 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| **ours**, official recipe, 3 seeds (`|h|`) | 0.0461 ± 0.0016 | 0.1013 ± 0.0035 | 0.0237 ± 0.0010 | 7.68 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| **ours**, official recipe, 3 seeds (raw) | 0.0458 ± 0.0011 | 0.1476 ± 0.0836 | 0.0237 ± 0.0009 | 7.68 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| ours, stretched recipe, 3 seeds (`|w|`, `|h|`) | 0.0421 ± 0.0007 | 0.0971 ± 0.0027 | 0.0223 ± 0.0013 | 7.64 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| ours, stretched recipe, 3 seeds (`|h|`) | 0.0421 ± 0.0007 | 0.0971 ± 0.0027 | 0.0223 ± 0.0013 | 7.64 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| ours, stretched recipe, 3 seeds (raw) | 0.0417 ± 0.0002 | 0.1445 ± 0.0848 | 0.0220 ± 0.0009 | 7.64 ± 0.01 | 33.3 ± 57.7 % | 0.0 ± 0.0 % |
| **ours Optim** (random init, 300 it., same count as the official PT) | 0.0438 | 0.1294 | 0.0587 | 7.72 | n/a | n/a |

**T1c, loss ablations** (official recipe, `|h|`, mean ± std over seeds):

| model | L_pixel (real) | L_pcpt (real) | L_pixel (strokes) | D_L1 | D_W |
|---|---|---|---|---|---|
| paper full | 0.056 | 0.807 | 0.042 | 0.083 | 0.018 |
| **ours** full (n = 3) | 0.073 ± 0.001 | 16.532 ± 0.400 | 0.046 ± 0.002 | 0.101 ± 0.003 | 0.024 ± 0.001 |
| paper w/o L_pixel | 0.081 | 1.012 | 0.068 | 0.241 | 0.024 |
| **ours** w/o L_pixel (n = 3) | 0.088 ± 0.001 | 19.465 ± 0.517 | 0.066 ± 0.002 | 0.098 ± 0.002 | 0.022 ± 0.001 |
| paper w/o D_L1 | 0.074 | 0.941 | 0.077 | 0.267 | 0.019 |
| **ours** w/o D_L1 (n = 3) | 0.073 ± 0.002 | 16.448 ± 0.506 | 0.045 ± 0.001 | 0.192 ± 0.138 | 0.020 ± 0.004 |
| paper w/o D_W | 0.069 | 0.947 | 0.046 | 0.113 | 0.034 |
| **ours** w/o D_W (n = 3) | 0.070 ± 0.002 | 16.348 ± 0.383 | 0.045 ± 0.000 | 0.110 ± 0.002 | 0.030 ± 0.001 |
| paper w/o D_bce | 0.071 | 0.928 | 0.052 | 0.093 | 0.021 |
| **ours** w/o D_bce (n = 3) | 0.069 ± 0.001 | 15.777 ± 0.008 | 0.043 ± 0.001 | 0.147 ± 0.086 | 0.024 ± 0.001 |

Effect of removing a term (ablated / full; > 1 = worse):

| removed term | L_pixel (real) | L_pcpt (real) | L_pixel (strokes) | D_L1 | D_W |
|---|---|---|---|---|---|
| w/o L_pixel | paper 1.45, ours 1.21 | paper 1.25, ours 1.18 | paper 1.62, ours 1.44 | paper 2.90, ours 0.97 | paper 1.33, ours 0.94 |
| w/o D_L1 | paper 1.32, ours 1.00 | paper 1.17, ours 0.99 | paper 1.83, ours 0.98 | paper 3.22, ours 1.90 | paper 1.06, ours 0.83 |
| w/o D_W | paper 1.23, ours 0.97 | paper 1.17, ours 0.99 | paper 1.10, ours 0.97 | paper 1.36, ours 1.08 | paper 1.89, ours 1.26 |
| w/o D_bce | paper 1.27, ours 0.95 | paper 1.15, ours 0.95 | paper 1.24, ours 0.94 | paper 1.12, ours 1.46 | paper 1.17, ours 1.02 |

**D_L1 per seed, `|h|` -> `|w|`, `|h|`** (the raw predictions of the outlier runs paint as well as the full model, so their high D_L1 is a different parameter convention; the `|w|`, `|h|` fold is a diagnostic, it changes the rendered strokes and is not an exact symmetry of the renderer; `w < 0` = share of matched strokes with a negative width, raw):

| model | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| full | 0.105 -> 0.105 (w < 0: 0 %) | 0.100 -> 0.100 (w < 0: 0 %) | 0.099 -> 0.099 (w < 0: 0 %) |
| w/o L_pixel | 0.099 -> 0.099 (w < 0: 0 %) | 0.096 -> 0.096 (w < 0: 0 %) | 0.098 -> 0.098 (w < 0: 0 %) |
| w/o D_L1 | 0.120 -> 0.120 (w < 0: 0 %) | 0.106 -> 0.106 (w < 0: 0 %) | 0.352 -> 0.200 (w < 0: 100 %) |
| w/o D_W | 0.112 -> 0.112 (w < 0: 0 %) | 0.108 -> 0.108 (w < 0: 0 %) | 0.110 -> 0.110 (w < 0: 0 %) |
| w/o D_bce | 0.095 -> 0.095 (w < 0: 0 %) | 0.101 -> 0.101 (w < 0: 0 %) | 0.247 -> 0.101 (w < 0: 100 %) |

<!-- END TABLE1 -->

## Results and verdicts

Verdicts apply the tolerances fixed in the protocol; nothing was changed after the runs except the two amendments listed in the protocol (Optim timed once per size; ablations evaluated only at their final step). Round 1 of the review corrected the analysis of the D_L1 outliers (item 3 below) and added the `w < 0` count and the `|w|, |h|` fold to the tables; no number of the verdicts changed except the cell count of T1c.
Data (JSON, one file per table): `docs/experiments/data/pt_reproduction/`. Figures: `docs/media/pt_repro_fig{5,6,7}.jpg` (62 to 96 KB). The tables above regenerate with
`python -m scripts.exp_pt_table1 report` and `python -m scripts.exp_pt_table2 report` (no GPU needed; the measurements are the stages `strokes | real | optim` and `run | training`).

| item | verdict | evidence |
|---|---|---|
| **T1a** real images, PT better than Optim | **not reproduced (expected)**: our Optim is stronger | L_pixel 0.0726 (our PT) vs 0.0496 (Optim); L_pcpt 16.5 vs 8.1; LPIPS-Alex 0.521 vs 0.352. The paper has PT ahead of Optim on L_pixel and L_pcpt. Ours vs the official model: +8 % L_pixel, +9 % L_pcpt, +5 % LPIPS (seed spread 2 to 3 %: ours is consistently a little worse than the released weights). The stretched recipe leaves real-image L_pixel unchanged (0.0735) but improves LPIPS-Alex (0.502). |
| **T1b** stroke metrics | **partly reproduced** | Official recipe (`\|h\|`): L_pixel 0.0461 (+10 % vs paper, pass), D_L1 0.1013 (+22 %, pass), **D_W 0.0237 (+32 %, fail)**; PT beats Optim on D_L1 (0.101 vs 0.129) and D_W (0.024 vs 0.059) but **not on L_pixel** (0.0461 vs 0.0438 for our Optim, far better than the paper's 0.073). Stretched recipe: 0.0421 (+0.3 %), 0.0971 (+17 %), 0.0223 (+24 %), PT ahead of Optim on all three: **passes every criterion**. The released weights themselves score D_W 0.0240 (+33 %): the official-recipe D_W gap is shared with the reference, so it points to the evaluation protocol rather than to our training. |
| **T1c** ablations | **mostly not reproduced** (7 of 20 cells have the paper's direction) | see the ratio table. Reproduced: removing `L_pixel` degrades the pixel metrics (real 1.21, strokes 1.44; paper 1.45 / 1.62), removing `D_W` raises D_W (1.26; paper 1.89), removing `D_L1` raises D_L1 (1.90; paper 3.22), but this rise comes from one seed and is at least partly a parameter-convention difference (item 3), so it is a weak reproduction at best. The `w/o D_bce` D_L1 cell (1.46; paper 1.12) is **an artefact, not a reproduction**: its outlier seed predicts `w < 0` everywhere, and with `|w|, |h|` folded it is 0.101 like the full model. Not reproduced: removing `D_L1`, `D_W` or `D_bce` does **not** degrade the real-image metrics (ratios 0.95 to 1.00 against 1.15 to 1.32), and removing `L_pixel` does not raise D_L1 / D_W (0.97 / 0.94 against 2.90 / 1.33). |
| **T2a** inference time | **partly reproduced** | Optim / PT ratios 979 / 773 / 2545 vs 1391 / 1303 / 1715 (factors 0.70 / 0.59 / 1.48, all within 2: pass); **PT growth 128 -> 512 is 1.5x vs 5.5x (fail)**, while the official script on the same GPU grows 4.7x like the paper. Our PT is slower than the official code at 128 and 256 px (0.153 vs 0.100 s, 0.201 vs 0.166 s) and 2x faster at 512 px (0.231 vs 0.467 s). |
| **T2b** training time | **reproduced (better)** | 94.3 ms per step alone on the GPU, 1.48 h for 56,320 steps, against 3.78 h (T-010 benchmark) and 8.68 h (T-014 retrain) for the official code on the same GPU, and 3.79 h in the paper. |
| **Fig. 5** | reproduced | more layers = finer painting: L1 to the target 0.086 / 0.063 / 0.050 / 0.043 for K = 1..4 with 112 / 363 / 1,247 / 4,611 strokes. At K = 1 and 2 black patches remain at the borders (never covered by a stroke). The L1 is taken against the 512 px target after a nearest upsampling of the 64 / 128 px painting, so it includes the upsampling error at small K. |
| **Fig. 6** | partly reproduced | only the `L_pixel` ablation is visibly (and numerically, L1 0.051 vs 0.045) worse on Mona Lisa; the other three ablated models look like the full one (L1 0.045 / 0.042 / 0.045), unlike the paper where each ablation is visibly worse. One image, one seed. |
| **Fig. 7** | reproduced | the network trained with the brush images paints with a rectangle (L1 0.0442) and a disc (0.0422) as well as with its own brushes (0.0428): the predicted strokes are interpreted through the renderer, not tied to the brush texture. |

## Gap analysis

Starting from the discrepancy list of section 9 of the study note. Each cause is **measured here**, **measured earlier (cited)** or a **hypothesis** (marked as such, not tested).

1. **Real-image numbers (T1a).** (a) *Datasets*: five images instead of the paper's 3 x 100, so absolute numbers cannot be compared. (b) *L_pcpt scale*: our pinned definition (relu3_3, squared error averaged over features) gives values about 20 times the paper's 0.74 to 0.86; the paper's layer, normalization and reduction are unpublished (study note section 9), so only orderings and ratios are comparable. (c) *Our Optim is stronger than the paper's*: with the same number of strokes it beats PT on every real-image metric, the opposite of the paper's table. This negative result is kept as such: a tuned optimization painter is better than the feed-forward network on the fidelity metrics; PT's advantage is time (T2a: 770 to 2,500 times faster). Why the paper's Optim is weaker is a hypothesis (different renderer, no tuned perceptual term). (d) *Ours vs the released weights*: our models are 8 to 9 % behind on real-image L_pixel although their synthesized-stroke metrics equal or beat the released model's (stretched recipe L_pixel 0.0421 vs 0.0410, D_W 0.0223 vs 0.0240). The training distribution is the same, so this is **not explained**; hypotheses: the released checkpoint's training budget (paper: batch 128 x 30,000 iterations, code: 64 x 56,320), a favourable seed, the train / inference window mismatch of section 9. None isolated.
2. **Stroke metrics (T1b).** The ordering "PT beats Optim" fails on L_pixel only: the random-init Optim (300 iterations) reaches L_pixel 0.0438 but its parameters stay far from the ground truth (D_L1 0.129, D_W 0.059, within 6 % and 3 % of the paper's Optim 0.137 / 0.057). D measures how identifiable the ground-truth stroke is, not painting quality: Optim finds other strokes that paint equally well. The released model's D_W is 33 % above the paper's PT value, so part of the gap comes from the unspecified synthesized-stroke protocol (hypothesis: a different stroke distribution in the paper's test set). Training longer closes part of the gap (stretched recipe vs official recipe: D_W -6 %, L_pixel -9 %, measured).
3. **Sign and convention of the stroke parameters.** Some runs paint as well as the full model with the stroke parameters in another convention (signs of `w` and `h`, offset of `theta`); only `D_L1` penalizes it. Seed 0 of the full model learns `h < 0` for every stroke (T-014): raw D_L1 is 0.148 ± 0.084 against 0.101 ± 0.004 with `|h|`. The same happens in 4 of the 12 ablation runs (`w/o L_pixel` seed 0, `w/o D_L1` seed 0, `w/o D_bce` seeds 1 and 2). After the `|h|` fold two outliers remain (`w/o D_L1` seed 2: 0.352, `w/o D_bce` seed 2: 0.247), and **they are also a different sign convention**: these two checkpoints are the only ones of the 21 that predict `w < 0` for 100 % of the strokes (see the per-seed table, column `w < 0`); the raw predictions of these two runs paint as well as the full model (stroke L_pixel 0.0441 and 0.0425, against 0.0447 to 0.0468 for the full model), and folding `|w|` too brings `w/o D_bce` seed 2 to 0.101 (full model: 0.101) and `w/o D_L1` seed 2 to 0.200; the rest of the latter, and the shifted `theta` means of `w/o D_L1` seeds 0 and 1 (reviewer's probe: 0.28 and 0.42 against 0.48 to 0.54 for the full models), are consistent with an orientation-convention offset (not measured in the tables). The `|w|`, `|h|` fold is a diagnostic, not an exact symmetry of the renderer: it changes the rendered strokes (stroke L_pixel 0.0441 -> 0.0559 and 0.0436 -> 0.0570 for these two runs, and 0.0468 -> 0.0479 when only `|h|` is folded for the full model seed 0); the exact equivalences were not derived. Consequences: the D_L1 ratios of `w/o D_bce` (1.46) is a convention artefact, and the `w/o D_L1` rise (1.90) is at least partly one. Hypothesis, untested: the paper's 3.22 for the same ablation may have the same mechanism, since without `D_L1` nothing fixes the convention. The paper does not mention this convention issue; the table of the ablations uses `|h|` only (the folds are shown in the per-seed table).
4. **Ablations (T1c).** The paper's effects of `D_L1`, `D_W`, `D_bce` on the pixel metrics (+15 to +32 %) are not observed (ratios 0.95 to 1.00, seed std 1 to 3 %). Candidate causes, **none tested**: (a) *loss weights*: we follow the code (`lambda_pixel = 10`, `lambda_decision = 10`, section 9), where the pixel loss dominates the gradient and a stroke term is a small share of it, whereas in the paper's weights (all 1, W 10) each stroke term weighs more; this fits the pattern that the pixel-term ablation reproduces and the stroke-term ablations do not. The "paper weights" preset exists (T-012), but running the ablations with it is 12 more trainings, not done here. (b) *budget*: the paper trains 30k x 128, we use 56k x 64. (c) *matching*: our ablations keep the code's `1 - logit` matching cost, so removing `D_bce` leaves the assignment unchanged, while the paper's matching cost contains the BCE term.
5. **Inference time (T2a).** The Optim / PT ratio is of the paper's order, but PT time grows only 1.5x from 128 to 512 px in our pipeline: it is launch- and synchronization-bound at small sizes (cProfile at 128 px: `paint_layer` self time 28 ms per call, with host synchronizations such as `bool(flat_decision.any())` and gather / scatter of small tensors), i.e. a per-layer cost independent of the image size; at 512 px our batching wins (0.231 s against 0.467 s). The official script grows 4.7x like the paper, so the paper's growth is a compute-bound GPU behaviour and ours is the pipeline overhead. Not a defect for painting at 512 px, but a 2 to 3x opportunity at 128 px. The Optim baseline takes 150 s at 128 px for 2,037 strokes and 587 s at 512 px: iterations, not pixels, dominate.
6. **Training time (T2b).** 94 ms per step alone, and about 88 ms per process when three trainings shared the GPU (T-014: 83 min for 56,320 steps), so one process is CPU / launch-bound. The 38 ms per step measured in T-013 was **not reproduced** today with an idle GPU (re-measured twice: by `profile_step` and by a 600-step run of `train_pt`, 94 ms in both); cause not isolated (hypothesis: CPU frequency / power state). The conclusion (faster than the official code: 1.5 h against 3.8 to 8.7 h) holds with either value. The 2.3x gap between the official code's retrain (8.68 h) and its benchmark (3.78 h) was noted in T-014 and is not investigated.
7. **Figures.** Fig. 6 is one image and one seed: a visual check cannot see the 1 to 5 % effects of the table. The black patches of Fig. 5 at K = 1, 2 come from the official procedure (non-final layers paint only the confident strokes on a black canvas).

### What this changes for the project
- The M2 network is usable as the stroke predictor: the stretched recipe equals the released weights on the stroke metrics and is 8 % behind them on real-image L_pixel.
- The paper's ablation conclusions about the stroke terms do not transfer to the code's loss weights; any ablation ticket (T-017) should state the weights it uses, take `|h|`, and use 3 seeds (single seeds are within the noise).
- A good optimization baseline beats PT on fidelity: the argument for PT is time.

### Proposed tickets
Speed of small-size inference (host synchronizations in `paint_layer`); ablations with the paper's loss weights and matching cost (3 seeds x 4); an option that fixes the (w, h, theta) convention of the stroke parameters (loss or reparameterization) and a symmetry-invariant D_L1; a check of the synthesized-stroke protocol against the paper's D_W; the 38 ms vs 94 ms training-step discrepancy.


## T-017: ordering of the loss terms and per-term verdicts

**Scope and data.** T-017 asks for the four ablation trainings and their evaluation with the T-016 protocol. They were already run in T-016 (4 terms x 3 seeds, official recipe, `outputs/pt_abl/`, evaluated with the protocol above, tables in the "T1c" block), together with the three full-model seeds of T-014 (`outputs/pt_full/`). T-017 therefore trains nothing new: it fixes the ordering analysis and the per-term verdict rules below, applies them to the stored measurements (`docs/experiments/data/pt_reproduction/`), and reports them. One command regenerates the table: `python -m scripts.exp_pt_ablation_verdict report`.

**Rules (written before the script that applies them; fixed knowing the T-016 ablation tables, so they are not blind to the data, only to the ordering statistics).**
- Five columns per term, as in T1c: real L_pixel, real L_pcpt, stroke L_pixel, D_L1, D_W. Primary numbers: the T-016 protocol (`|h|`). Sensitivity: D_L1 with `|w|`, `|h|` folded (the fold is a diagnostic of the parameter convention, T-016 gap analysis item 3).
- Effect of a term = ablated mean / full mean per column (> 1 = the term helps). A cell is **worse** if the ratio is > 1 and the Welch t statistic of (ablated - full) over the 3 + 3 seeds is >= 2.13 (one-sided 5 % level at about 4 degrees of freedom).
- **Per-term verdict:** the term **matters (reproduced)** if its ablation is worse on at least 3 of the 5 columns, **partly** on 1 or 2, **not reproduced** on 0. In the paper every term is worse on all 5 columns (there is no seed statistic), so the paper's verdict is "matters" for the four terms.
- **Importance of a term** = mean of its five ratios. **Ordering** = terms sorted by importance. Agreement with the paper = Kendall tau between the two importance vectors, and whether the most important term is the same. Ties (importances within 0.05 of each other) are reported as ties, not ranked.
- The paper's ordering is computed from its Table 1 with the same rule.
