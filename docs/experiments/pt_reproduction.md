# Paint Transformer: reproduction of the paper results and gap analysis (T-016)

**Hypothesis.** Our reimplementation (T-011 to T-015) reproduces the reproducible results of Liu et al. (ICCV 2021) listed in the feasibility table of
[`docs/design/paint_transformer.md`](../design/paint_transformer.md) section 7 (Table 1 stroke part and ablations, Table 2 times, Fig. 5 to 7) within the tolerances
fixed below; every gap is quantified and explained, starting from the paper-versus-code list of section 9.

This section (**Protocol**) was written and committed **before any run of this ticket**; the results sections are appended after. Where a choice had to be made because the paper does not specify it, the choice and its reason are stated
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

### Table 2
- **T2a, inference time at 128 / 256 / 512 px** (`K` = 2, 3, 4): PT through our pipeline, PT through the official script (time from the end of the model loading to the end of the painting), our optimization baseline (M1 default, 2,037 strokes, scaled to the image size by keeping the same levels). Median of 5 runs per image over the five images
  after 2 warm-up runs, GPU idle (no training running). RL: n/a. **Absolute times are not compared** (RTX 5070 against 2080 Ti): the comparison is the Optim / PT ratio and the growth of PT from 128 to 512.
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

### Commands (one per table; filled in below once the scripts exist)
`python -m scripts.exp_pt_table1 {run,report}`, `python -m scripts.exp_pt_table2 {run,report}`, `python -m scripts.exp_pt_figures`; the ablation trainings are `python -m scripts.train_pt seed=S loss.preset=custom loss.<term>=0 output_dir=outputs/pt_abl/<name>/seedS`.
