# Baseline (a): optimization-only progressive stroke decomposition (T-005)

Code: `brushstroke/baselines/optim/{config,init,losses,sinkhorn,optimize}.py`. Config: `configs/baseline_optim.yaml`
(typed `PaintOptimConfig`). Script: `python -m scripts.paint_optim --input img --out dir [key.sub=value ...]`.
Tests: `tests/test_optim_baseline.py`.

**Placement.** `brushstroke/baselines/optim/` (new subpackage): baselines (a) optimization, (b) Paint Transformer and
(c) the RL agent are siblings and are kept apart from `stroke_model/` (our learned predictor) and `contribution/`.

> **Update (T-009).** The defaults described below are those of T-005 (9 levels, 100 steps per level, frozen earlier strokes,
> LPIPS-VGG 0.44 at 1794 strokes on the Mona Lisa). T-009 found that under-optimization and the frozen schedule caused most of the
> quality gap and changed the defaults of `configs/baseline_optim.yaml` to: grids 1, 2, 4, 8, 16, 20 with 1, 1, 1, 3, 4, 2 strokes
> per patch (2037 strokes), steps per level 100, 100, 100, 300, 300, 150 (`iters_per_level`), `finetune_all: true` with the same learning
> rate for the earlier strokes. Mean over 5 images × 3 seeds: LPIPS-VGG 0.416 → 0.308, held-out LPIPS-Alex 0.465 → 0.359, 563 s
> per image. The T-005 configuration is preserved in `configs/experiments/baseline_t005.yaml`. New flags: `init_mode`,
> `init_size_mode`, `final_pass_iters`, `iters_per_level`. See [../experiments/optim_sweep.md](../experiments/optim_sweep.md).

## Method
1. Canvas = flat background colour (white).
2. Levels `g` in `levels` (grid `g x g`, default 1, 2, 3, 4, 6, 8, 12, 16, 20). At each level, `strokes_per_patch[level]`
   new strokes are created **per patch** (`g^2 * s` strokes) and **optimized jointly** for `iters` steps against the loss, rendered on
   top of the (frozen) canvas of the earlier levels with the T-003 renderer (chunked; gradient checkpointing switches on
   automatically when the estimated activation memory `strokes * H * W * 44 B` exceeds `checkpoint_bytes`).
   `finetune_all: true` also optimizes the earlier strokes (lr x `finetune_lr_factor`).
3. Strokes are appended level by level, patch-major (`scale_level` = index in `levels`, `patch_id` = patch inside the level,
   `source = "optim_baseline"`), giving a coarse-to-fine `StrokeSequence` whose `meta` has config hash, seed, per-step loss
   curves, per-level final losses and timings, runtime and peak VRAM.

**Initialization** (`init.py`, after SNP's `stroke_sampler`): the centre of each new stroke is sampled inside its patch with
probability proportional to `blur(sum_c |target - canvas|) ** init_error_power` (sigma = `init_blur_frac` x patch size;
uniform fallback if the patch is already perfect), the colour is read from the lightly blurred target at that position, the
length is `init_size_scale * U(0.6, 1.2) / g`, thickness = length * `U(init_aspect)`, rotation uniform, opacity
`init_opacity`. All randomness comes from one CPU `torch.Generator(seed)`; brush variants per stroke are drawn from it too and
stored explicitly in `brush_index`.

**Parameters** are optimized in logit space (`params_from_unconstrained`, sigmoid) so they always stay in range; the learning
rate decays with a cosine schedule to `lr_final_frac * lr` within each level. Optional dead-stroke resampling
(`resample_every > 0`): every k steps, strokes with opacity < `dead_opacity` or area < `dead_area` are re-drawn from the current
residual and their optimizer statistics reset.

**Losses** (`losses.py`, weights in `loss:`; 0 disables): pixel L1; LPIPS-VGG (Zhang et al., CVPR 2018, `lpips` package, frozen);
Sinkhorn divergence (`sinkhorn.py`): each colour channel is average-pooled to `ot_size^2`, turned into a distribution over
grid cells and compared with the debiased entropic OT `OT(a,b) - OT(a,a)/2 - OT(b,b)/2` (Cuturi, NIPS 2013; log-domain
iterations, `ot_eps`, `ot_iters`; squared grid distance). Tested: identical -> 0, symmetric, decreases as distributions
approach, matches the squared cell distance for Dirac-like inputs, differentiable.

## Where we differ from Stylized Neural Painting (verified against `painter.py` / `demo.py` of the official repo)
| SNP (official code) | ours | why |
|---------------------|------|-----|
| neural renderer (`net_G`) approximating strokes | analytic textured renderer (T-003), exact and differentiable | no renderer training, exact gradients, any resolution |
| RMSprop on `x_ctt, x_color, x_alpha`, with hard clamps (`x_ctt` to [0.1, 0.9], colour/alpha to [0, 1]) after each step | Adam (default) or RMSprop on sigmoid-reparameterized logits | no clamping artefacts; ranges guaranteed by construction |
| strokes are added one at a time per block (`stroke_sampler(anchor_id)` then ~500/m optimization steps) | all new strokes of a level optimized jointly | far fewer sequential steps, one batched render per step |
| error-map sampling: blur ~`out_size/8`, power 4 | blur 0.25 x patch, power 2 (configurable) | same idea, tuned on our scale |
| stroke budget `max_m_strokes / total_blocks` per block | explicit `strokes_per_patch` per level | direct control of the 1000-2000 stroke budget |
| L1 + optional OT (`epsilon=0.01`, `niter=5`, weight 0.1) | L1 + optional LPIPS + optional debiased OT on pooled intensities | LPIPS added as perceptual term; debiasing makes OT 0 at the optimum |
| earlier strokes not explicitly frozen in `demo.py` | frozen by default (flag to fine-tune) | cheaper, stable coarse-to-fine |
Not reproduced: SNP's dilation/erosion of foreground/alpha and its dedicated neural renderer; its other brush types. The SNP
paper's exact grid/budget schedule was read only through the repo (per-level counts are not in the files above).

## Results of T-005 (512^2 public-domain image, RTX 5070; superseded by the T-009 defaults)
These are the T-005 numbers, kept as the record of the first run. The current defaults and their 5-image confirmation are in
[../experiments/optim_sweep.md](../experiments/optim_sweep.md).

Image: *Mona Lisa* (Leonardo da Vinci, c. 1503), Wikimedia Commons file
`Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg` (960 px thumbnail, public domain, 2-D reproduction of a work
long out of copyright), center-cropped to a square and resized to 512^2. Kept out of git (`data/raw/`).

T-005 configuration (9 levels up to 20 x 20, 1794 strokes, Adam lr 0.02, 100 steps/level, frozen earlier strokes,
L1 + 0.1 LPIPS; the frozen copy is `configs/experiments/baseline_t005.yaml`), seed 0:

| metric | value |
|--------|-------|
| L1 (mean abs, [0, 1]) | 0.0358 |
| PSNR | 26.1 dB |
| LPIPS-VGG | 0.441 |
| strokes | 1794 |
| runtime | 149 s |
| peak VRAM (allocated) | 6.7 GB (incl. LPIPS; 3.3 GB at 1160 strokes, L1 only) |

Outputs: `outputs/optim_final/{final.png, target.png, progress/level_XX.png, sequence.npz, loss_curve.json, metrics.json}`
(gitignored). The result reads as an oil painting (broad strokes for the background, visible bristle texture); the face keeps a
soft, rough rendering: uniform allocation spends strokes evenly, which is exactly what the later contribution (saliency-aware
allocation) targets. **The M1 success criterion "LPIPS < 0.25 within 2000 strokes" is NOT met** (0.44): with textured strokes of
bounded size, LPIPS-VGG at 512^2 stays far above 0.25 for this baseline; we report this honestly.

Loss curves: within every level the 10-step-averaged training loss never increases (0 of 9 intervals, all 9 levels); per-level
end losses (L1 canvas error) fall 0.375 -> 0.165 -> 0.093 -> 0.076 -> 0.057 -> 0.047 -> 0.040 -> 0.037 -> 0.036.

Ablations/tuning runs on the same image (single seed, indicative only):

| variant | strokes | L1 | LPIPS-VGG | time |
|---------|---------|-----|-----------|------|
| levels 1..12 (1160 strokes), RMSprop lr 0.05, L1 only | 1160 | 0.0341 | 0.592 | 80 s |
| + levels 16 (1702), L1 only | 1702 | 0.0336 | 0.583 | 115 s |
| same as first, `brush.kind=flat` | 1160 | 0.0336 | 0.655 | 79 s |
| same as first, 250 steps/level | 1160 | 0.0324 | 0.577 | 197 s |
| same as first, Adam lr 0.02 | 1160 | 0.0317 | 0.590 | 80 s |
| same as first, LPIPS weight 0.3 | 1160 | 0.0466 | 0.489 | 104 s |
| final config (levels to 20, Adam, LPIPS 0.1) | 1794 | 0.0358 | 0.443 | 149 s |
| final config + OT 0.1 | 1794 | 0.0356 | 0.441 | 182 s |
Observations: L1 alone plateaus around 0.032-0.034 and LPIPS ~0.58 whatever the budget/steps; the textured oil brush is better
perceptually than `flat` (0.59 vs 0.66); the LPIPS term is what moves LPIPS (at some L1 cost); the OT term has no measurable
effect at this weight.

## Determinism
Seed + `seed_everything(seed, deterministic=True)`: identical sequences on CPU (tested bitwise). On GPU the default sampler
was bitwise reproducible in the renderer tests, but the whole optimization is not guaranteed bit-identical (see `renderer.md`:
use `render.sampler: gather` with the deterministic flag for a guaranteed path); two GPU runs of the same configuration (tuning run and final run) gave L1 0.0357 and 0.0358.

Update (T-007): the GPU non-determinism of the optimization comes from cuDNN's convolution backward in the LPIPS-VGG loss, not
from `grid_sample`. The switch `cudnn_deterministic: true` (deterministic cuDNN only, full speed) makes runs bitwise identical:
two full pipeline runs of the three M1 demos gave identical sequences, GIFs and metrics. The full `deterministic` mode is about
3x slower and refuses the `grid_sample` backward, so the demos do not use it. Without the switch (the default), GPU runs are
still not guaranteed bit-identical.

## Limitations
- Cost is O(strokes * H * W) per step because each stroke is rasterized over the whole canvas (new strokes only touch one patch):
  cropping to the patch bounding box would cut the runtime a lot.
- No resampling by default; no LPIPS at coarse levels (it is evaluated at full 512^2 every step, a large part of the runtime).
- Hyperparameters (lr, steps, init) were tuned by hand on one image; no multi-seed study (that belongs to M3).

## References
- Zou, Shi, Guo, Ye. *Stylized Neural Painting*. CVPR 2021 (and github.com/jiupinjia/stylized-neural-painting).
- Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NIPS 2013.
- Peyre, Cuturi. *Computational Optimal Transport*. 2019 (log-domain Sinkhorn, debiased divergence; Feydy et al. AISTATS 2019).
- Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS)*. CVPR 2018.
- Kingma, Ba. *Adam*. ICLR 2015; Tieleman, Hinton. *RMSprop*. 2012.
