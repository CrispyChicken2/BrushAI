# Paint Transformer training runs (T-014)

**Question.** Does our reimplementation (T-011 to T-013), trained with the official recipe on the synthetic stroke generator, converge to a validation plateau
that is consistent across seeds, and does it match the official code? **Short answer:** yes, with one explained caveat (the sign of the stroke height, below).
Three seeds of the official 56,320-step recipe were not always flat enough before the learning-rate decay (pre-decay gain 0.5-1.05 %), so the recipe was stretched to
112,640 steps (decay at 92 % of the run, same shape): all three seeds then satisfy the plateau criterion. Our seed 0 and the *official code* trained with seed 0 end at the same
total (3.950 vs 3.944) and make the same sign choice, which is the strongest evidence that the reimplementation behaves like the original.

## Protocol
- Code: `python -m scripts.train_pt` (T-013), recipe of `configs/stroke_model/pt.yaml`: Adam lr 1e-4, betas (0.5, 0.999), batch 64, fp32, official loss preset, synthetic data generated on the fly, seeds 0/1/2 (the seed fixes the initialization and the training stream; the validation set is fixed by `val.seed`).
  *Official recipe*: `n_epochs 200 + 20 decay` (56,320 steps). *Stretched recipe*: `schedule.n_epochs=400 schedule.n_epochs_decay=40` (112,640 steps, 7.2 M samples), all other settings unchanged. Three runs shared the GPU at a time (about 88 ms per step each).
- During training: 512 fixed held-out patches every 1000 steps (eval mode). **Final evaluation** (`scripts.pt_training_report evaluate`): final checkpoint of every seed, the official pretrained weights (`model.pth`) and the official code retrained here, all on the **same 4096 held-out patches** (seed 999), same loss and matcher.
- **Plateau criterion** (fixed before looking at the stretched runs): relative improvement of the median training total between the windows [74 %, 83 %) and [83 %, 92 %) of the steps, i.e. the loss must be flat (< 1 %) *before* the learning-rate decay of the recipe is applied. Medians over 250-step block means. The validation gain is printed for information only (points every 1000 steps are too few and are polluted by the spikes explained below).
  The lr decay itself always gives a final drop of about 3 %, so a plain "last 10 % of the steps" window could never be flat; the "last-5 %" column shows that the end of the decay tail is stable (about 1 % or less).
- **Checkpoint-selection rule.** The final checkpoint of the run (end of the lr tail), not a best-on-validation one: the validation curve is too noisy for a best-of rule and the tail is the lowest-loss region anyway. Across seeds: the lowest total on the 4096-patch evaluation after the `|h|` canonicalization (see finding 2). Result: **`outputs/pt_long/seed2/latest.pt`** (total 3.547; seed 1 is within 0.003). Checkpoints stay in `outputs/` (never committed).
- Time and memory (**correction, T-016**: the 38 ms step of T-013 was re-measured at 94 ms alone on an idle GPU, so one process needs about 1.5 h for the official recipe and about 3 h for the stretched one; the 36 / 71 min below are the T-013 figures, kept as measured then): one process alone does a step in 38 ms (T-013), i.e. 36 min / 71 min for the two recipes; with three processes sharing the GPU each took 83 / 181 minutes (column "time" below). Peak VRAM 851 MiB per run. The official code retrained once took **8.68 h (0.555 s per step, 56,320 steps)**, 2.3 times slower than the 0.237 s per step benchmarked in T-010; I did not investigate the gap.

## Results
Regenerate everything below with `python -m scripts.pt_training_report report` (it reads `docs/experiments/data/pt_training/`; the full logs and checkpoints live in `outputs/`).

<!-- BEGIN GENERATED -->

### Official recipe (56,320 steps)

| seed | steps | time (min) | peak VRAM (MiB) | pre-decay gain, train (criterion) | pre-decay gain, val | last-5 % gain, val |
|---|---|---|---|---|---|---|
| seed0 | 56320 | 83 | 851 | 1.05 % (NOT met) | 3.22 % | 0.24 % |
| seed1 | 56320 | 83 | 851 | 0.65 % (met) | -2.07 % | 0.94 % |
| seed2 | 56320 | 83 | 851 | 0.50 % (met) | -0.03 % | 1.15 % |

Final checkpoints on one held-out synthetic set (4096 patches, identical for all rows); `h<0` = fraction of matched strokes with a negative predicted height, `abs(h)` = loss after replacing `h` by `|h|`:

| model | total | pixel | L1 | Wasserstein | BCE | count error | h<0 | total, abs(h) |
|---|---|---|---|---|---|---|---|---|
| seed0 | 3.950 | 0.468 | 0.244 | 0.245 | 2.993 | 0.745 | 1.000 | 3.823 |
| seed1 | 3.731 | 0.458 | 0.100 | 0.238 | 2.935 | 0.754 | 0.000 | 3.731 |
| seed2 | 3.760 | 0.447 | 0.099 | 0.227 | 2.988 | 0.754 | 0.000 | 3.760 |
| official pretrained weights | 3.864 | 0.410 | 0.098 | 0.240 | 3.117 | 0.793 | 0.000 | 3.864 |
| official code retrained here (seed 0) | 3.944 | 0.461 | 0.245 | 0.247 | 2.992 | 0.765 | 1.000 | 3.811 |
| **ours, mean ± std over seeds** | 3.814 ± 0.119 | 0.458 ± 0.011 | 0.148 ± 0.084 | 0.237 ± 0.009 | 2.972 ± 0.032 | 0.751 ± 0.005 | 0.333 ± 0.577 | 3.771 ± 0.047 |

![official_recipe](../media/pt_training_official_recipe.png)

### Stretched recipe (112,640 steps)

| seed | steps | time (min) | peak VRAM (MiB) | pre-decay gain, train (criterion) | pre-decay gain, val | last-5 % gain, val |
|---|---|---|---|---|---|---|
| seed0 | 112640 | 181 | 851 | 0.83 % (met) | 0.16 % | 0.09 % |
| seed1 | 112640 | 181 | 851 | 0.79 % (met) | -0.93 % | 0.02 % |
| seed2 | 112640 | 181 | 851 | 0.79 % (met) | 2.71 % | 0.08 % |

Final checkpoints on one held-out synthetic set (4096 patches, identical for all rows); `h<0` = fraction of matched strokes with a negative predicted height, `abs(h)` = loss after replacing `h` by `|h|`:

| model | total | pixel | L1 | Wasserstein | BCE | count error | h<0 | total, abs(h) |
|---|---|---|---|---|---|---|---|---|
| seed0 | 3.708 | 0.418 | 0.242 | 0.229 | 2.819 | 0.707 | 1.000 | 3.584 |
| seed1 | 3.550 | 0.419 | 0.095 | 0.221 | 2.814 | 0.712 | 0.000 | 3.550 |
| seed2 | 3.547 | 0.416 | 0.096 | 0.211 | 2.825 | 0.726 | 0.000 | 3.547 |
| official pretrained weights | 3.864 | 0.410 | 0.098 | 0.240 | 3.117 | 0.793 | 0.000 | 3.864 |
| official code retrained here (seed 0) | 3.944 | 0.461 | 0.245 | 0.247 | 2.992 | 0.765 | 1.000 | 3.811 |
| **ours, mean ± std over seeds** | 3.602 ± 0.092 | 0.417 ± 0.002 | 0.144 ± 0.085 | 0.220 ± 0.009 | 2.819 ± 0.005 | 0.715 ± 0.010 | 0.333 ± 0.577 | 3.560 ± 0.021 |

![long](../media/pt_training_long.png)

<!-- END GENERATED -->

## Findings
1. **The recipe is not fully converged after 56,320 steps; the stretched one is.** Under constant lr the training total still falls by 0.5-1 % per 10 % of the run before the decay (official recipe) and by 0.8 % with the stretched recipe (criterion < 1 %: met for the three seeds, but only just, the curve decays slowly like a log). The 2x schedule improves the held-out total by about 5 % (3.73-3.76 -> 3.55 for seeds 1 and 2) and is 8 % below the **official pretrained weights** (3.864; per term: our pixel 0.416 vs 0.410, our BCE 2.82 vs 3.12, equal L1 and Wasserstein). The official weights were trained with the authors' own settings (the paper says batch 128 and 30k iterations), which may explain the difference; the comparison shows our runs are at least as good, not that the recipe is better.
2. **Seed outlier explained: the sign of `h`.** Seed 0 ends with L1 0.242 vs 0.096 for seeds 1 and 2 because **100 % of its matched strokes have a negative predicted height `h`** (L1 of `h` = 0.81 = 2 x 0.4). The Wasserstein distance depends on `h^2` only, and the PT renderer accepts a negative `h` (the brush is just mirrored), so only the weak parameter-L1 term prefers `h > 0`; the choice is made in the first few hundred steps and is a deterministic function of the seed.
   Evidence that it is not a bug of ours: the **official code trained with seed 0 makes the same choice** (h<0 = 1.000) and reaches total 3.944 / L1 0.245 against our 3.950 / 0.244 (pixel, Wasserstein and BCE within 2 %), while the official pretrained weights and our seeds 1 and 2 have h<0 = 0. With `h` replaced by `|h|` seed 0 goes from 3.950 to 3.823 (stretched: 3.708 to 3.584), still above seeds 1 and 2 because its pixel term is slightly worse. Consequence for downstream tickets: **use `|h|` (or the h>0 seeds)**; a flagged ablation that fixes the sign (e.g. `|h|` in the matching and L1 terms) is proposed as a new ticket. The mean ± std row therefore includes the outlier; the `abs(h)` column (3.560 ± 0.021 for the stretched recipe) is the relevant spread.
3. **Validation spikes are an eval-mode BatchNorm artefact, not training instabilities.** The validation curve shows isolated points 0.3 to 1.2 above the plateau (figure, left), always in the BCE term, while the training curve is smooth. Diagnosis on a fresh 8000-step run (seed 0): every 100 steps the validation set was scored with the running BN statistics (what `train.py` does), with batch statistics and with statistics re-estimated on 20 fresh batches. Running statistics differ from the re-estimated ones by more than 0.3 in 16 of 81 evaluations (max 1.23), and the spread (std) of the validation total along the run is 0.33 with running statistics against 0.11 recalibrated. At the end of a run (tiny lr) the two agree within 0.003 for the three official-recipe checkpoints, so **the final metrics are unaffected**; intermediate validation values are read through medians (the plateau criterion uses the training loss for the same reason). `train.py` was left unchanged (T-013 scope); an optional BN recalibration before evaluation is a proposed ticket. The diagnostic scripts were throw-away (not committed); the numbers above are from their logs.
4. **No divergence** in any of the six runs: the loss falls from about 69 to 5 within 500 steps and decreases afterwards (2000-step means).
5. **Seed spread among the h>0 seeds (1 and 2):** total 3.550 vs 3.547 (stretched) and 3.731 vs 3.760 (official recipe), i.e. 0.1 % and 0.8 %: small against the loss scale, as required.

## Hugging Face upload (prepared, not run)
`python -m scripts.upload_pt_to_hub --checkpoint outputs/pt_long/seed2/latest.pt --metrics-json outputs/pt_long/final_eval.json --seed-key seed2 --repo-id <user>/brushstroke-painttransformer` builds the staging folder `outputs/hub/painttransformer/`
(weights-only `pytorch_model.pth`, resolved config, SHA-256 of that config, model card with licence and metrics) and stops there; `--push` is required to upload (needs `huggingface_hub` and `HF_TOKEN`, neither installed here). The upload is a later decision.
Licence stated in the card: MIT (this repository); the weights are trained from scratch on synthetic data and none of the original authors' weights is redistributed.

## Reproduction
```
python -m scripts.fetch_pt_brushes ; python -m scripts.pt_reference_study fetch      # assets + official code and weights (outside git)
python -m scripts.train_pt seed=0 output_dir=outputs/pt_full/seed0                     # seeds 0,1,2 (official recipe)
python -m scripts.train_pt seed=0 output_dir=outputs/pt_long/seed0 schedule.n_epochs=400 schedule.n_epochs_decay=40
python -m scripts.pt_official_train --seed 0                                           # reference run of the official code (8.7 h here)
python -m scripts.pt_training_report evaluate --runs outputs/pt_full                   # and --runs outputs/pt_long
python -m scripts.pt_training_report archive  --runs outputs/pt_full --tag official_recipe   # and: --runs outputs/pt_long --tag long
python -m scripts.pt_training_report report                                            # tables + figures from docs/experiments/data/
```

## T-019: fixing the `(w, h, theta)` convention and a convention-invariant `D_L1`

**Question.** Can the convention outliers of T-014 / T-016 (a seed predicts `h < 0` or `w < 0` for every stroke) be removed, or
neutralized in the metric, without losing quality? **Hypotheses.** H1: a convention fix (option A, B or C below) leaves 0 of
3 seeds with more than 50 % negative `w` or `h`, and the held-out total is not worse than the baseline recipe. H2: a
convention-invariant `D_L1` agrees across seeds. Derivation of the equivalence classes and of the options:
[docs/design/pt_convention.md](../design/pt_convention.md).

### Protocol (fixed before the runs)
- **Options** (flags, default = official behaviour, bitwise unchanged; checked by a forward and backward hash against `main`):
  **A** `model.convention=positive` (`w`, `h` = softplus, `theta` wrapped into [0, 1)); **B** `loss.l1_invariance=mirror` (parameter
  L1 of the loss and of the matching cost minimized over the mirror class of the prediction); **C** `loss.lambda_sign=10`
  (`+ 10 mean(relu(-w) + relu(-h))`, the weight of the other terms). Official recipe (56,320 steps), seeds 0, 1, 2, otherwise the
  T-014 settings. The stretched recipe (112,640 steps) only for the best option. Runs: `python -m scripts.exp_pt_convention train`.
- **Evaluation**: the T-014 held-out set (4,096 patches, seed 999, same loss and matcher) for every model, the new runs and
  the existing T-014 / T-016 checkpoints (`python -m scripts.exp_pt_convention evaluate`). Three totals of the official loss
  (weights 10 / 1 / 10 / 10): `raw`; `|h|` (T-014 protocol, `h` replaced by `|h|`); `invariant` (parameter L1 and matching over the
  mirror class). The invariant total is the **convention-neutral comparator**: it is the one used for H1, because options B and C
  may keep a negative sign that the raw total penalizes. Stroke metrics are the unweighted terms of T-016 plus `D_L1` per class
  level (`exact`, `mirror`, `rot180`, `swap`), all with the official matching.
- **H1 rule** (per option, 3 seeds): *outlier seed* = more than 50 % of the matched strokes with `w < 0` or `h < 0`; *worse* =
  mean invariant total above the baseline's and Welch t >= 2.13 (the T-017 rule); H1 is met when there is no outlier seed and the
  option is not worse. Baseline = the T-014 runs of the same length (official: 3.771 +- 0.047 after `|h|`; stretched: 3.560 +- 0.021).
  Option B keeps the sign free by design, so its outlier count is reported but it is judged on `D_L1` and on the total.
- **Best option** = the one that meets H1 with the lowest invariant total, else the fewest outlier seeds, then the lowest total.
- **H2 rule**: the coefficient of variation (std / mean over 3 seeds) of `D_L1` at a class level is at most 5 % and at least 3 times
  lower than the raw `D_L1`'s, for the two T-014 groups (official and stretched recipe); the four T-016 ablation groups are reported
  as supporting evidence. Levels: `exact` (strict: renders identically), `mirror` (primary), `rot180` and `swap` (diagnostics).

### Results
Regenerate with `python -m scripts.exp_pt_convention report` (reads `docs/experiments/data/pt_convention/`).

<!-- BEGIN CONVENTION -->

### Equivalence classes verified by rendering

Mean absolute alpha difference between the stroke and each parameter variant, 4000 random strokes of the training distribution (64 x 64 pixels; the mean alpha is 0.091); `error / area` is the share of the stroke that changes. Unrelated strokes of the same distribution differ by 0.1424 (1.57 of the area).

| variant `(sign w, sign h, theta shift, swap)` | class | alpha error (train) | error / area | foreground error (train) | alpha error (inference) |
|---|---|---|---|---|---|
| `(+1, +1, +1.0, -)` | rot180 (diagnostic) | 0.02753 | 0.303 | 0.03328 | 0.03325 |
| `(+1, -1, +0.0, -)` | mirror | 0.01310 | 0.144 | 0.01694 | 0.01191 |
| `(+1, -1, +1.0, -)` | rot180 (diagnostic) | 0.03047 | 0.336 | 0.03410 | 0.03638 |
| `(-1, +1, +0.0, -)` | rot180 (diagnostic) | 0.03047 | 0.336 | 0.03477 | 0.03643 |
| `(-1, +1, +1.0, -)` | mirror | 0.01310 | 0.144 | 0.01635 | 0.01196 |
| `(-1, -1, +0.0, -)` | rot180 (diagnostic) | 0.02753 | 0.303 | 0.03304 | 0.03325 |
| `(-1, -1, +1.0, -)` | exact | 0.00000 | 0.000 | 0.00306 | 0.00005 |
| `(+1, +1, +0.5, swap)` | swap (diagnostic) | 0.02840 | 0.313 | 0.03551 | 0.03261 |
| `(+1, +1, +1.5, swap)` | swap (diagnostic) | 0.02840 | 0.313 | 0.03539 | 0.03266 |
| `(+1, -1, +0.5, swap)` | swap (diagnostic) | 0.02776 | 0.306 | 0.03623 | 0.03136 |
| `(+1, -1, +1.5, swap)` | swap (diagnostic) | 0.02591 | 0.285 | 0.03192 | 0.03034 |
| `(-1, +1, +0.5, swap)` | swap (diagnostic) | 0.02591 | 0.285 | 0.03248 | 0.03039 |
| `(-1, +1, +1.5, swap)` | swap (diagnostic) | 0.02776 | 0.306 | 0.03575 | 0.03131 |
| `(-1, -1, +0.5, swap)` | swap (diagnostic) | 0.02840 | 0.313 | 0.03555 | 0.03266 |
| `(-1, -1, +1.5, swap)` | swap (diagnostic) | 0.02840 | 0.313 | 0.03571 | 0.03262 |

### H2: convention-invariant D_L1 on the existing checkpoints (T-014, T-016)

| model | L_pixel | D_L1 | D_L1 exact | D_L1 mirror | D_L1 rot180 | D_L1 swap | D_W | h < 0 | w < 0 | outlier seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| official weights | 0.0410 | 0.0978 | 0.0978 | 0.0977 | 0.0947 | 0.0806 | 0.0240 | 0.0 % | 0.0 % | 0 / 1 |
| ours, official recipe (T-014) | 0.0458 ± 0.0011 | 0.1476 ± 0.0836 | 0.1471 ± 0.0828 | 0.1015 ± 0.0039 | 0.0990 ± 0.0044 | 0.0842 ± 0.0028 | 0.0237 ± 0.0009 | 33.3 ± 57.7 % | 0.0 ± 0.0 % | 1 / 3 |
| ours, stretched recipe (T-014) | 0.0417 ± 0.0002 | 0.1445 ± 0.0848 | 0.1439 ± 0.0837 | 0.0974 ± 0.0034 | 0.0947 ± 0.0037 | 0.0774 ± 0.0019 | 0.0220 ± 0.0009 | 33.3 ± 57.7 % | 0.0 ± 0.0 % | 1 / 3 |
| ablation w/o pixel (T-016) | 0.0665 ± 0.0017 | 0.1465 ± 0.0856 | 0.1460 ± 0.0847 | 0.0980 ± 0.0020 | 0.0963 ± 0.0022 | 0.0857 ± 0.0023 | 0.0223 ± 0.0009 | 33.3 ± 57.7 % | 0.0 ± 0.0 % | 1 / 3 |
| ablation w/o dl1 (T-016) | 0.0450 ± 0.0008 | 0.2403 ± 0.1248 | 0.2066 ± 0.0876 | 0.1190 ± 0.0128 | 0.1078 ± 0.0077 | 0.0805 ± 0.0037 | 0.0195 ± 0.0040 | 33.3 ± 57.7 % | 33.3 ± 57.7 % | 2 / 3 |
| ablation w/o dw (T-016) | 0.0449 ± 0.0003 | 0.1098 ± 0.0019 | 0.1098 ± 0.0019 | 0.1098 ± 0.0019 | 0.1082 ± 0.0023 | 0.0925 ± 0.0005 | 0.0299 ± 0.0005 | 0.0 ± 0.0 % | 0.0 ± 0.0 % | 0 / 3 |
| ablation w/o dbce (T-016) | 0.0426 ± 0.0003 | 0.2445 ± 0.1506 | 0.1908 ± 0.0834 | 0.1362 ± 0.0658 | 0.0985 ± 0.0054 | 0.0815 ± 0.0033 | 0.0238 ± 0.0008 | 66.7 ± 57.7 % | 33.3 ± 57.7 % | 2 / 3 |

**Spread of `D_L1` across the 3 seeds (coefficient of variation = std / mean).**

| group | raw | `\|h\|` fold | exact | mirror | rot180 | swap |
|---|---|---|---|---|---|---|
| ours, official recipe (T-014) | 56.7 % | 3.4 % | 56.3 % | 3.9 % | 4.5 % | 3.3 % |
| ours, stretched recipe (T-014) | 58.7 % | 2.8 % | 58.2 % | 3.5 % | 3.9 % | 2.4 % |
| ablation w/o pixel (T-016) | 58.4 % | 1.9 % | 58.0 % | 2.1 % | 2.3 % | 2.7 % |
| ablation w/o dl1 (T-016) | 51.9 % | 71.9 % | 42.4 % | 10.8 % | 7.1 % | 4.6 % |
| ablation w/o dw (T-016) | 1.7 % | 1.7 % | 1.7 % | 1.7 % | 2.1 % | 0.6 % |
| ablation w/o dbce (T-016) | 61.6 % | 58.5 % | 43.7 % | 48.3 % | 5.4 % | 4.0 % |

Bias of the invariant metric on the 12 models with no negative `w` or `h` (mean decrease of `D_L1` relative to the raw value; a level that lowers clean models also lowers wrong ones for the wrong reason): `exact` 0.0 %, `mirror` 0.1 %, `rot180` 2.6 %, `swap` 16.8 %.

Rule (H2 holds at a level): coefficient of variation across seeds at most 5 % and at least 3 times lower than the raw `D_L1`'s.

- `exact`: T-014 groups not both within the rule (no, no); ablation groups 0 / 4.
- `mirror`: T-014 groups both within the rule (yes, yes); ablation groups 1 / 4.
- `rot180`: T-014 groups both within the rule (yes, yes); ablation groups 1 / 4.
- `swap`: T-014 groups both within the rule (yes, yes); ablation groups 3 / 4.

<!-- END CONVENTION -->
