# Evaluation protocol (T-022)

Status: **fixed before any M3 code or run.** Every number, rule and threshold below is a decision made here, with the reason; the
baselines (M3) and the contribution (M4) are judged by this document and by nothing chosen afterwards. Facts about external
resources were checked on 2026-10-25 and are marked *(checked)*; measurements made for this note are marked *(measured)* and come
from the probe described in section 10.

**Amendment rule.** The protocol can change only by a dated entry in "Amendments" (end of this note) that says what changed, why, and
which runs it affects. An amendment made after test-set numbers exist for the affected metric is labelled *post hoc* in every table
that uses it. Nothing is changed silently, and no metric is added to or removed from a headline table after the results are seen.

## 0. Decisions at a glance

| # | decision | section |
|---|---|---|
| D1 | 100 images: 40 paintings + 30 photos + 30 SDXL; **dev split 10** (tuning, dev-only target), **test 90** (all final tables) | 1 |
| D2 | Only CC0 / public-domain sources whose licence is machine-readable per file; Met Open Access is the main one; manifest with SHA-256 | 1 |
| D3 | 512 x 512, centre crop of images with aspect ratio in [3:4, 4:3], area-antialiased resize; no padding | 1 |
| D4 | Headline metric **LPIPS-Alex** (never in a loss); L1, SSIM, LPIPS-VGG, PSNR reported; DISTS optional/confirmatory; a metric in a method's loss is marked | 2 |
| D5 | Anytime curve on 15 fixed checkpoints, x normalized by the budget; **AUC@B for B in {250, 500, 1000, 2000}**, normalized per image by the blank-canvas LPIPS | 3 |
| D6 | Methods are compared by the **prefix** of their ordered stroke sequence; "cannot reach B" is reported, not imputed | 3 |
| D7 | Saliency-weighted LPIPS: spatial LPIPS-Alex weighted by the saliency of the **target**; evaluation saliency = InSPyReNet (Swin-B, DUTS-TR); M4 must not use it | 4 |
| D8 | Strokes-to-target with tau fixed on the **dev split** from the strongest baseline, and fixed in a committed file before test runs | 5 |
| D9 | Unit of analysis = image; replicates averaged per image; Wilcoxon signed-rank, rank-biserial effect size, bootstrap CI, Holm within pre-registered families | 6 |
| D10 | Producer API: progressive canvases + stroke counts at 512², no dependence on our renderer | 7 |
| D11 | Learning to Paint trained on a **Met Open Access CC0 subset** disjoint from dev/test by object and by artist; no published numeric table exists, so the check is relative to the released actor | 8 |

## 1. Evaluation set

### 1.1 Composition (100 images)

| category | n | strata | source (licence) |
|---|---|---|---|
| paintings | 40 | 24 European paintings (8 portraits / 8 landscapes and scenes / 8 other: still life, figures, animals); 8 Asian art (paintings, woodblock prints); 8 other (Modern and Contemporary Art, American paintings) | Met Open Access (CC0) |
| photographs | 30 | 10 portraits (faces), 20 other clear subjects (animals, objects, scenes); at least 20 in colour | CC0 / public-domain photographs with a per-file machine-readable statement (see 1.2) |
| SDXL | 30 | 10 portraits, 10 scenes, 10 objects and animals; in each 5 photographic and 5 painted prompts | generated here, prompts and seeds in 1.4 |

About 28 of the 100 images carry a face as the salient subject (10 SDXL + 10 photographs + about 8 painted portraits): enough for the
"faces, subjects" claim of the contribution to be testable per category (section 6, item 6), not enough to make faces the headline.

**Dev split (10)**: 4 paintings (2 European, 1 Asian, 1 other), 3 photographs, 3 SDXL. **Test split (90)**: the rest (36 / 27 / 27).
The dev split is used for everything that involves looking at results: debugging, method tuning, choice of the strokes-to-target
threshold (section 5), sanity checks of the saliency maps. **No dev image appears in a final table.** The split is drawn once, by a
seeded draw inside each stratum (`random.Random(22)` over the objects sorted by id), *before any method is run on the set*, and is
written in the manifest (`dev: true/false`). The three SDXL dev images are fixed by prompt index instead (section 1.4).

### 1.2 Licensing rules

1. Only images that are **CC0 or public domain, with a machine-readable statement for that file** (API field or licence tag), are
   accepted. The record stored in the manifest is: source, identifier, page URL, licence string, retrieval date, SHA-256 of the file used.
2. Met Open Access: the Met waived copyright over its dataset under CC0 and serves images of public-domain works *(checked:
   metmuseum.github.io)*. Only objects with `isPublicDomain = true` are used. The same page says works still under copyright carry a
   credit in `rightsAndReproduction`, so the filter is the boolean, never the department.
3. **Excluded**: Unsplash, Pexels, Pixabay and similar (custom licences, not CC0, and no per-file legal text that can be checked by
   a script); any image of an identifiable living person; any image with a visible watermark, text block or frame that covers more
   than 5 % of the area (decided at curation by a rule, not by looking at method results).
4. Photographs of people: sitters of historical public-domain photographs (Met Open Access photographs, Commons files under CC0 or a
   US-government public-domain tag) only. The faces in the SDXL images are not real persons; no prompt names a person or an artist who is alive.
5. Smithsonian Open Access (CC0) is a candidate extra source for photographs; its licence statement must be verified file by file at
   the start of T-024 before use (the page could not be fetched while writing this note).
6. Images are **fetched, never committed** (the repository holds the manifest only), like `docs/data_sources.md` does for the M1 images.
   The 30 SDXL images are our own outputs; they are dedicated to CC0 1.0 in the data card, and generated images need not be committed either.
7. Met catalogue facts *(checked, 2026-10-25, API v1.1; the old `search` endpoint was retired on 2026-10-01)*: 371,179 objects with
   `isPublicDomain` and an image; European Paintings 2,725; medium "Paintings" 14,417; Asian Art 34,229; Photographs 26,716; Modern and
   Contemporary Art 13,162; Drawings and Prints 100,388. The pool is large enough that the draw is a real draw (stratum sizes >> 8).

### 1.3 Preparation (resolution policy)

- **512 x 512** (the resolution of all M1/M2 numbers).
- A source image is eligible if its short side is at least 768 px (so the resize is a reduction) and its aspect ratio lies in [3:4, 4:3]
  (a centre crop then removes at most 25 % of the area). Other images are not padded (padding adds a blank region that every metric
  scores differently from content); they are either excluded or given a **hand-set crop box recorded in the manifest, set before
  any method run**.
- Centre crop to a square, then area-antialiased downscale to 512 (`brushstroke.utils.io.center_crop_resize` on a 960 px source
  was the M1 policy; the 768 px floor is new). The prepared tensor is float32 in [0, 1], three channels, saved as PNG next to the manifest hash.
- Greyscale sources are converted to three equal channels; at most 10 of the 30 photographs may be greyscale.

### 1.4 SDXL subset

Model: Stable Diffusion XL base 1.0 (Podell et al., 2023), licence CreativeML Open RAIL++-M *(checked: the licence says "Licensor claims
no rights in the Output You generate using the Model"; its use restrictions concern unlawful or harmful uses and do not restrict the
benign prompts below)*. Generation settings are the library defaults, **not tuned**: `StableDiffusionXLPipeline`, 1024 x 1024, fp16
weights with model CPU offload (one-off, about 8-10 GB), default scheduler, library default steps and guidance scale, no refiner, no
negative prompt, one image per prompt with seed = prompt index (0 to 29). Package versions are pinned in T-024 and written in the manifest. An image is regenerated with the next seed
only for a technical failure (black output, safety-checker blank), and the event is logged; never for aesthetic reasons. Images are downscaled from 1024 to 512 with the area filter.

| id | style | prompt |
|---|---|---|
| 00 | photo | a studio portrait photograph of an elderly fisherman with a white beard, soft window light, shallow depth of field |
| 01 | photo | a close-up photograph of a smiling young woman with freckles and curly red hair, outdoors in overcast light |
| 02 | photo | a portrait photograph of a child in a yellow raincoat looking at the camera, city street in the background |
| 03 | photo | a photograph of a middle-aged man wearing glasses and a green scarf, three-quarter view, warm evening light |
| 04 | photo | a candid photograph of two musicians laughing together on a street corner |
| 05 | paint | an oil painting of a young woman in a blue dress seated by a window, visible brush strokes, classical portrait |
| 06 | paint | an impressionist oil painting of an old sailor smoking a pipe, thick brush strokes |
| 07 | paint | a baroque oil painting portrait of a nobleman with a lace collar, dark background, dramatic lighting |
| 08 | paint | a watercolor and gouache portrait of a girl with a red ribbon in her hair |
| 09 | paint | an expressionist oil painting of a laughing woman with bold colors and a turquoise background |
| 10 | photo | a photograph of a mountain lake at sunrise with mist and pine trees |
| 11 | photo | a photograph of a narrow cobblestone street in an old European town after rain |
| 12 | photo | a photograph of a lighthouse on a rocky coast under a stormy sky |
| 13 | photo | a photograph of a wheat field with a lone tree at golden hour |
| 14 | photo | a photograph of a busy harbor with colorful fishing boats in the morning |
| 15 | paint | an impressionist oil painting of a river with water lilies and a small wooden bridge |
| 16 | paint | an oil painting of a snow covered village at dusk with lit windows, thick impasto |
| 17 | paint | a romantic landscape painting of a ship in a stormy sea with dramatic clouds |
| 18 | paint | a post-impressionist painting of a sunflower field under a swirling blue sky |
| 19 | paint | a Japanese woodblock print style image of large waves and a distant mountain |
| 20 | photo | a photograph of a red fox sitting in the snow looking at the camera |
| 21 | photo | a photograph of a bowl of ripe peaches on a wooden table, soft light |
| 22 | photo | a photograph of a vintage bicycle leaning against a blue wall |
| 23 | photo | a photograph of a ceramic teapot and two cups on a linen cloth |
| 24 | photo | a photograph of a horse standing in a misty meadow |
| 25 | paint | a still life oil painting of a vase of flowers, lemons and a glass of wine, Dutch golden age style |
| 26 | paint | an oil painting of a white cat sleeping on a velvet cushion |
| 27 | paint | a gouache painting of a rooster with colorful feathers |
| 28 | paint | an oil painting of a bowl of cherries and a silver spoon on a dark table |
| 29 | paint | a charcoal and oil sketch of a horse running in a field |

The dev images of this subset are prompts 04, 14, 25 (one per content type, fixed here: 04 is a photographic portrait, 14 a photographic scene,
25 a painted still life), so the dev split is not drawn from the generated outputs.

## 2. Metrics

All images are float tensors in [0, 1], shape `(3, 512, 512)`, compared with the target at the same resolution. Implementations live in
`brushstroke/eval/metrics.py` (T-026 extends them).

| metric | definition | role |
|---|---|---|
| **LPIPS-Alex** | LPIPS v0.1 (Zhang et al., 2018), AlexNet features, inputs scaled to [-1, 1], full 512² image, lower is better | **headline, held-out** |
| LPIPS-VGG | same with VGG16 | secondary; is a loss term of the optimization baseline (weight 0.1) |
| L1 | mean absolute RGB difference | secondary; a loss term of the optimization baseline |
| SSIM | Wang et al. (2004), Gaussian 11 x 11, sigma 1.5, C1 = 0.01², C2 = 0.03², mean over channels (the implementation already in the repo) | secondary |
| PSNR | 10 log10(1/MSE) | context only (it tracks L1) |
| DISTS | Ding et al. (2022), reference code under MIT *(checked)* | optional confirmation, never a headline |
| runtime, peak VRAM | section 2.3 | cost |

### 2.1 The held-out rule
A metric is **held-out for a method** if it appears in no loss, reward or selection criterion of that method. Every table marks, per
method, the metrics that are *not* held out (a dagger). Today: the optimization baseline optimizes L1 and LPIPS-VGG (and an optional
OT term), so both are daggered for it; the Paint Transformer trains on synthetic strokes with pixel and stroke-parameter terms, and
Learning to Paint uses a learned discriminator reward (Huang et al., 2019), so neither uses LPIPS. Claims "A beats B" are made on the
headline LPIPS-Alex and on held-out secondary metrics. For the contribution (M4): **its losses and its selection criteria may not
contain LPIPS-Alex or DISTS.** If M4 needs them, an amendment must name a new held-out metric before the M4 runs.

### 2.2 Reference canvases (controls printed in every results table)
- *blank*: a constant canvas (black and white, both reported).
- *blur*: the target with a Gaussian blur of sigma = 1 px, i.e. no painting at all. For the five M1 images it scores LPIPS-VGG 0.268 and
  LPIPS-Alex 0.318, while the optimization baseline at about 2000 strokes scores 0.308 and 0.359 (`docs/experiments/optim_sweep.md`):
  a stroke painting at this budget is *worse* than a blurred copy on LPIPS, so absolute LPIPS values must be read against this floor.
- *LPIPS range*: LPIPS-Alex is **not bounded by 1** *(measured)*: a blank white or black canvas scores 0.77 to 1.115 on the six images of
  the probe. This is why the AUC is normalized by the blank-canvas value (section 3.2).
- The sequence-render control: for Paint Transformer, the canvas of the official window-based pipeline and the generic re-render of its
  `StrokeSequence` differ by 0.5-0.9 % mean absolute value (M2 retrospective, limitation 9). The protocol evaluates PT **as its sequence**
  (the animation and the contribution act on sequences); the difference between the two renders is reported once as a control row.

### 2.3 Runtime and VRAM
- Runtime: wall-clock of the producer from receiving the target to returning the last checkpoint canvas, with `torch.cuda.synchronize()`
  before and after, excluding model loading and the first (warm-up) image, **with nothing else running on the GPU** (T-016 had a contended
  timing; the harness refuses to start a timed run when another process uses more than 1 GB of VRAM). Reported: median and inter-quartile
  range over images, GPU model and driver. Metric computation is outside the timed region.
- Peak VRAM: `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` per image, in GB; reserved memory is logged as well.
  For producers that use LPIPS inside their loop (the optimization baseline) that cost is part of the method and counted.
- Metric cost *(measured, RTX 5070)*: LPIPS-Alex on 15 canvases of 512²: 0.016 s, 1.06 GB peak; LPIPS-VGG on 15 canvases: 0.58 s, **9.96 GB** peak. The VGG metric
  must therefore be computed in chunks of at most 2 canvases (about 1.3 GB) so that it fits next to a running producer.
- Determinism: repeated LPIPS calls on the same tensors are bit-identical on this machine *(measured)*; GPU producers are bit-reproducible
  only on the same GPU, driver and torch build (M2 findings), so results record those.

## 3. Anytime quality

### 3.1 The curve
For a painting made of an ordered stroke sequence, `q(n)` is the LPIPS-Alex between the target and the canvas after the first `n` strokes (the
**prefix**: stroke `k` is painted over strokes `0..k-1`, nothing else changes). `n = 0` is the producer's declared blank canvas. Checkpoints (fixed):

`n in {0, 25, 50, 100, 150, 200, 250, 350, 500, 750, 1000, 1250, 1500, 1750, 2000}`

which contains the four reporting budgets 250, 500, 1000 and 2000. The grid is denser at small `n`, where the curves move fastest. The
x-axis is **normalized by the budget**: for a budget `B`, `u = n / B` on [0, 1] using the checkpoints `n <= B`. Between checkpoints the
curve is interpolated **linearly in (u, q)**; no smoothing, no fitting.

### 3.2 The score
`AUC@B = integral_0^1 q(Bu) du` by the trapezoid rule on the checkpoints up to `B`, divided by the image's **reference value**
`q_ref = max(LPIPS-Alex(black canvas, target), LPIPS-Alex(white canvas, target))`. `q_ref` depends on the image only, never on the method, so the
normalization cannot favour a method. Lower is better; 0 is a perfect painting from the first stroke, and a method that never improves on
a blank canvas scores about 1. The ratio is in [0, 1] in practice (the probe's `q_ref` ranges from 0.95 to 1.115, above every blank-canvas
value), and the harness **asserts** `0 <= AUC <= 1` per image and logs any violation instead of clipping.
The raw (un-normalized) integral is stored next to it. Headline anytime score: **AUC@2000**; AUC@250/500/1000 are the other budgets.
A log-spaced variant (x = log n over n in [25, B]) is stored as a secondary, because the linear AUC is dominated by the long plateau at the
end of the curve and the log variant weights early strokes, which is where allocation and ordering should help most.

Why this definition: the project's research question (does allocation and ordering improve quality "at every intermediate step, not only at the end")
is a statement about the whole curve. The idea follows anytime algorithms, whose quality profile is a function of the time or resource
spent (Dean and Boddy, 1988; Zilberstein, 1996), with the stroke count as the resource; the budget normalization and the per-image reference make
images of different difficulty comparable and the score bounded.

### 3.3 Comparing methods with different stroke counts
- Every method is evaluated on the **prefix of its ordered sequence** at the budgets 250/500/1000/2000. Totals differ (optimization baseline about 2000,
  Paint Transformer about 3.8-5.0 k at 512², Learning to Paint a multiple of its action bundle of 5): all are cut at the budget.
- A method whose total number of strokes is below a budget **cannot reach it**: `AUC@B` and the metrics at `B` are reported as "n/a (max n = ...)"
  and are not imputed with its final value. Statistical comparisons at that budget exclude the method.
- A stroke is one emitted primitive. Zero-opacity or fully covered strokes still count; a producer may not drop strokes after the fact to lower its count.
- The prefix of a sequence that was jointly optimized (the optimization baseline re-optimizes earlier strokes together with later ones) is *what the
  final painting looks like with only its first n strokes*, which is lower quality than a run with budget `n`. The prefix protocol is chosen because it is
  the setting of the research question and costs one run per image and seed; it is not the same as "fixed-budget runs". As a **side check** the
  optimization baseline is also run with its own 250/500/1000/2000-stroke configurations on 30 test images x 3 seeds (a subset, for cost: section 9)
  and the two protocols are reported together; if they rank the methods differently, the table says so.
- **Blank canvas**: the producers' native blank canvas is kept (optimization baseline white; Paint Transformer and Learning to Paint black, as they are trained), and recorded in the result.
  The two blank scores are printed (section 2.2). A claim that rests on `AUC@B` must also hold for `AUC` computed on `n >= 250` only (the part of
  the curve after the blank canvas has mostly been covered); a claim that does not survive is reported as not robust to the canvas colour.

## 4. Saliency-weighted LPIPS

### 4.1 Definition
Let `m(x)` be the **spatial LPIPS-Alex map** between canvas and target (the `lpips` package with `spatial=True`; at 512² it returns a 512 x 512 map whose mean equals
the scalar LPIPS to 1e-7 *(measured)*), and `s(x)` the evaluation saliency map of the **target**. Then

`SW-LPIPS = sum_x w(x) m(x)`, `w(x) = (1 - lambda) s(x) / sum_y s(y) + lambda / N`, `lambda = 0.1`, `N = 512 * 512`,

with `s` in [0, 1] as returned by the model (no re-scaling). The uniform component keeps the metric defined when the model finds almost nothing salient and keeps
a small cost on the background, so a method cannot score well by painting only the subject. `lambda = 0.1` is a decision made here, not tuned, and `lambda = 0`
and `lambda = 0.3` are stored as sensitivity columns. With uniform `s`, SW-LPIPS equals the scalar LPIPS (a unit test in T-026). SW-LPIPS is reported at the four budgets and as an AUC computed like section 3.
Treating a saliency map as a weighting distribution rather than as a set of fixation points follows the discussion of saliency map formats in Bylinskii et al. (2019).

### 4.2 The evaluation saliency model and the circularity rule
**Circularity rule.** *The evaluation saliency model must differ from every saliency or segmentation model used by the contribution (M4) for allocation or ordering:
different checkpoint, and not a fine-tune of, or distilled from, the same network. The contribution may use it for neither the loss nor the selection of strokes.*
Using the same model on both sides would measure how well the method agrees with that model, not whether the painting is better where it matters.

- **Evaluation model**: **InSPyReNet**, Swin-B, checkpoint trained on DUTS-TR only (`InSPyReNet_SwinB`), a salient-object detector (Kim et al., ACCV 2022).
  Code licence MIT *(checked, GitHub licence API)*. The model zoo page states no licence for the checkpoints *(checked)*; they are therefore treated like the Paint Transformer
  weights: **fetched, SHA-256 pinned, used as a measuring instrument, never redistributed**. The saliency maps of the dev and test images are derived data and stay in `outputs/`.
- **Contribution models allowed**: the Segment Anything family (Kirillov et al., 2023; Apache-2.0), a U²-Net-type network (Qin et al., 2020; code Apache-2.0), or a lightweight
  net trained by us; **InSPyReNet and its checkpoints are forbidden for M4**. An amendment is required to change either side.
- **Control (robustness)**: a model-free saliency, the spectral-residual map (Hou and Zhang, 2007), needs no weights and no licence. Every SW-LPIPS claim must keep its sign with
  this control; otherwise it is reported as depending on the evaluation model.
- Maps are computed once per image from the 512² target (T-025), stored with the model checkpoint hash, and **independent of the painting**: the weights of a metric
  never depend on the method being measured. The producer interface (section 7) never receives them.
- Why a salient-object model and not a fixation model: the contribution's claim is about subjects and faces getting more strokes, which is an object-level notion. DeepGaze IIE (Linardos et al.,
  2021) would be an independent fixation-based alternative but its repository carries no licence file *(checked: GitHub licence API 404)*, so it cannot pass the licence check.

## 5. Strokes-to-target

`n*(tau) = ` the smallest stroke count at which the LPIPS-Alex curve reaches `tau`, by linear interpolation between the two bracketing checkpoints (the curve is not assumed monotone:
the **first** crossing counts). If the curve never goes below `tau` within `n <= 2000`, the image is **right-censored** at 2000 (reported as "> 2000").
Per method: the fraction of images that reach `tau`, and the median of `n*` with censored values kept at 2000 (a conservative lower bound); the reaching-fraction curve over `tau` is a
Kaplan-Meier-style survival curve for censored data (Kaplan and Meier, 1958) and is plotted as the companion of the anytime figure; it is also an instance of a performance profile (Dolan and Moré, 2002).
Paired tests treat a censored image as a tie at 2000 (conservative).

**Choice of tau** (fixed here as a rule, evaluated on the dev split only): `tau_1000` = the median over the 10 dev images of the LPIPS-Alex that the **strongest baseline** reaches at
1000 strokes (the strongest is the baseline with the lowest dev mean LPIPS-Alex at 1000 strokes, expected to be the optimization baseline). Rationale: by construction the best baseline reaches the
target on about half of the images at that budget, so the metric discriminates between methods that are faster, equal and slower, instead of being trivially met or never met.
`tau_500` and `tau_2000` are defined the same way (budgets 500 and 2000) as secondary targets. The three values are written in `configs/eval/protocol.yaml` (T-026) and committed **before** any test run;
changing them requires an amendment.

## 6. Statistics

1. **Replicates.** Stochastic methods are run with seeds `{0, 1, 2}` at least. The producer declares what a replicate is (`replicate_kind`): the initialization seed for the
   optimization baseline; the **training seed** for learned models (the three Paint Transformer checkpoints of T-014, the final checkpoint of each, never chosen by test results).
   For Learning to Paint the rule is three independent training runs, *unless the cost measured in T-029 exceeds 48 GPU-hours per run* (the paper reports about 40 h on a 2080 Ti); then the
   protocol falls back to one training run, the table says "n = 1 training seed" and no between-training-seed uncertainty is claimed. Deterministic inference on a fixed checkpoint is one run per checkpoint.
2. **Unit of analysis = the image.** For each (method, metric, budget) the per-image value is the **mean over replicates**; tests and intervals are over the 90 test images. Seeds are not
   independent draws of images and are not pooled as if they were (Colas et al., 2018; Henderson et al., 2018; Agarwal et al., 2021 for the same issue in reinforcement learning).
3. **Descriptive**: mean +/- std over images, and separately the mean over images of the per-image std over replicates (the replicate noise), plus a 95 % bootstrap interval of the mean
   (10,000 resamples of images, percentile method, seed 22; Efron, 1979). Tables never show a seed std as if it were the uncertainty of the mean over images.
4. **"A beats B"**: a two-sided **Wilcoxon signed-rank test** (Wilcoxon, 1945) on the paired per-image differences over the 90 test images (the recommendation of Demsar, 2006, for comparing two
   methods over several data sets; Dror et al., 2018, for paired testing in general); zero differences dropped; `scipy.stats.wilcoxon` default method (exact for small n, normal approximation with continuity
   correction otherwise). Reported with the **effect size**: the matched-pairs rank-biserial correlation `r` (Kerby, 2014), the median paired difference in the metric's own unit, and the bootstrap 95 % interval of that median difference. A
   claim needs significance **and** an effect that is not negligible: an interval of the median difference that excludes 0, and `|r| >= 0.3`. A p-value alone is never a result.
5. **Multiple comparisons.** The *families* below are fixed now; Holm's step-down correction (Holm, 1979) is applied inside each family, and the corrected p-value is the one reported.
   - F1 (baselines): for each of the three headline endpoints P1 = LPIPS-Alex at 1000, P2 = AUC@2000, P3 = SW-LPIPS at 1000, all pairs of the baseline methods (3 pairs x 3 endpoints = 9 tests).
   - F2 (M4 main claims): the contribution against the best baseline on P1 to P3 (3 tests), plus each M4 ablation against the full contribution on P2 (defined in the M4 plan, before the M4 runs).
   - Everything else (other metrics, other budgets, per-category analyses, strokes-to-target, DISTS) is **secondary or exploratory**: reported with uncorrected p-values clearly labelled, never used for a headline claim.
   The caveat stays in every table: the families are a reasonable but arbitrary partition, each test is on 90 correlated images of three categories, and 90 images detect only moderate effects.
6. **Per-category** results (paintings / photographs / SDXL; faces vs the rest) are exploratory (n = 36, 27, 27 test images; far fewer faces), reported with the effect size and the interval, without a claim of significance.
7. **No tuning on test.** Method hyper-parameters (the optimization baseline's configuration, refinement steps, anything of M4) are tuned on the dev split only; the configuration hash is recorded before the
   test run. The test run is made once after a code freeze; a rerun for a bug is allowed if every run is logged with the reason and the choice between runs is never made by looking at the outcome.
8. Seeds and ordering: images are processed in sorted id order; every random source is seeded (`brushstroke.utils.seed`); the harness stores the git commit, package versions and GPU in each result file.

## 7. Baseline (producer) interface

The harness never renders a stroke itself for a non-sequence producer: a producer hands over **canvases**. This lets Learning to Paint (Bezier strokes, a neural renderer, 128² native) and any later
model be evaluated by the same code as ours, without our renderer. Sketch (T-026 implements it; names may change, the contract does not):

```python
@dataclass
class PaintResult:
    # stroke counts n at which canvases were taken (ascending, from the protocol grid)
    checkpoints: list[int]
    # (len(checkpoints), 3, 512, 512) float32 in [0, 1], canvas after exactly n strokes
    canvases: torch.Tensor
    # strokes emitted in all (at least max(checkpoints), or the method's own maximum)
    n_total: int
    background: tuple[float, float, float]
    # measured by the harness, not by the producer
    runtime_s: float
    peak_vram_bytes: int
    # config hash, native resolution, upsampled flag, replicate_kind, sequence path
    meta: dict[str, Any]


class Producer(Protocol):
    name: str
    stochastic: bool
    # "init seed" | "training seed" | "none"
    replicate_kind: str

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> PaintResult: ...
```

Rules:
1. **Input** is the 512² target and a seed, nothing else: no image id, category, dev flag or evaluation saliency map. A producer that wants a saliency map computes its own.
2. **Exact prefix**: the canvas at checkpoint `n` is the painting made of the first `n` strokes, produced by the method's own renderer. Nothing may be re-optimized or recomposed afterwards for a specific checkpoint.
3. **Stroke count**: one emitted primitive per stroke (section 3.3). A method that paints in patches or windows counts every stroke of every patch (Learning to Paint on a 512² image by 4 x 4 patches of 128² counts all strokes of all 16 patches).
4. **Resolution**: canvases are returned at 512². A producer working at a smaller native resolution renders its stroke parameters at 512² (Bezier parameters are resolution-free) or states `upsampled: true` in
   `meta`, and the report lists it. As a secondary evaluation **every** method's canvases are also scored at 128² (area downscale), which is Learning to Paint's native domain.
5. **Sequence adapter**: producers built on `StrokeSequence` (ours) use one adapter that calls `render_progressive`; other producers fill `canvases` themselves. A consistency test verifies that an adapter canvas at
   `n = len(sequence)` equals `sequence.render()`.
6. **Background**: declared and recorded (section 3.3).
7. Failures: an exception or a NaN canvas is recorded as a failure for that (image, seed) and shown in the table (count of failures); it is not skipped silently and not replaced.

## 8. Learning to Paint: training data and published-number check

### 8.1 What the paper did (checked on arXiv 1903.04411)
Huang, Heng and Zhou (ICCV 2019) trained a model-based DDPG agent with a learned WGAN-GP discriminator reward and a neural stroke renderer (quadratic Bezier stroke, 13 values), on MNIST, SVHN (200,000 sampled images),
CelebA (about 200,000 faces) and ImageNet (200,000 sampled images), all resized to 128 x 128; 5 strokes per step, 40 steps for training (the stroke budgets used per dataset are 5 for MNIST, 40 for SVHN, 200 for CelebA and 400 for ImageNet); about 40 h for CelebA/ImageNet; test sets of 10,000 (MNIST) or
2,000 held-out images. Their code and released weights are MIT-licensed code with weights on third-party hosting and no separate licence statement *(checked on the repository README)*.

### 8.2 Training data decision (the brief forbids copyrighted data)
CelebA and ImageNet are not CC0 (research-only licences over images of unclear rights), so neither is used. **Decision: train on a Met Open Access subset (CC0), disjoint from dev and test.**
- *Pool*: Met objects with `isPublicDomain = true` and an image, from the full object list (the paginated search is capped at 10,000 results per query; the object list endpoint has no such cap; at the documented limit of 80 requests per second the 371,179
  records take about 80 minutes to inspect).
- *Sample*: 100,000 images (the paper used 200,000 per dataset; 100,000 is half, chosen so that a download and a resize to 128² stays small, and recorded as a departure), stratified to resemble the evaluation mix: 40 % paintings (all painting
  departments and Asian art), 30 % photographs (Photographs department), 30 % drawings, prints and other flat works. The images are resized to 128 x 128 like the paper (centre crop for aspect).
- *Disjointness* (stronger than by id): remove the dev and test objects, all other images of those objects (`additionalImages`), **every work of an artist who has a dev or test painting**, and near-duplicates (a perceptual-hash match between a training and an evaluation image).
  The list of excluded ids is part of the T-029 data card. The SDXL subset and the non-Met photographs are disjoint from the Met by construction.
- *Alternatives rejected*: synthetic data (random shapes or generated textures) would avoid all licence questions but removes the natural-image statistics the actor relies on, and would make the baseline weaker for a reason unrelated to the algorithm; it is kept as a
  documented fallback only if the Met download turns out to be impractical. The Wikimedia Commons PD categories were rejected as the main source because their licence is per file and not machine-checkable at this scale.
- The neural renderer is trained on random strokes generated by a program, as in the paper; no external data.
- The released actor (CelebA) is never trained on our data and its weights are never redistributed; it is a reference.

### 8.3 Published-number check strategy
The paper contains **no numeric results table**: its quantitative results are curves of the test l2 loss (figures 8 and 9) and a few ratios in the text (l2 three times smaller than SPIRAL on CelebA with 20 strokes; five and twenty times smaller than two DDPG variants;
about 2.1 s for 200 strokes on a 2.2 GHz CPU and 9.5 times faster on a 2080 Ti). There is no published LPIPS, so **an "LPIPS reproduced from the paper" claim is impossible and is not made.** The check is instead:
1. *Renderer*: the trained neural renderer against an analytic rasterization of the same Bezier strokes on 1,000 random strokes (mean l2 reported); tolerance fixed in T-028.
2. *Agent against the released actor*: the same evaluation images and the same protocol for (i) our agent trained on the Met subset, (ii) the released CelebA actor. Pre-registered tolerance (to be confirmed in T-030 **before** the run): the l2 of our agent at 200 strokes is within 25 % of the released actor's, and both
   curves decrease monotonically with the stroke count at 100/200/400/1000 strokes as in the paper's figure 8c. Failing the tolerance is reported as a result, not hidden.
3. *Speed*: time for 200 strokes, to be put next to the paper's figure with the hardware difference stated.
4. The ratios against SPIRAL and DDPG variants are **not reproducible** (those systems are not part of the project) and are listed as such.

## 9. Compute and cost of the protocol (planning estimates, M3 plan)

| item | estimate | basis |
|---|---|---|
| optimization baseline, prefix protocol | 100 images x 3 seeds x 9.4 min = **47 h** (90 test: 42 h; 10 dev: 5 h) | 9.4 min per image at about 2000 strokes (`optim_sweep.md`) |
| optimization baseline, fixed-budget side check | 30 images x 3 seeds x 4 budgets x about 9 min = about 54 h | the 1000-stroke run takes 8.7 min, the 2000-stroke run 9.4 min (same document): a budgeted run is almost as expensive as the full one, so full fixed-budget evaluation (100 x 3 x 4 = 1,200 runs, about 180 h) is not affordable without T-027 |
| Paint Transformer | seconds in all | 0.23 s per 512² image per checkpoint (T-015) |
| Learning to Paint inference | negligible next to training | paper: 2.1 s per 200 strokes on a CPU |
| Learning to Paint training | measured in T-029 (paper: about 40 h on a 2080 Ti per run); 1 to 3 runs | decision rule in section 6.1 |
| metrics | about 1 s per curve with LPIPS-VGG in chunks of 2 | measured above; 15 canvases per curve |
| SDXL generation | 30 images, one-off, minutes | fp16 + model CPU offload |
| VRAM | metrics < 2 GB with chunking, saliency model estimated < 3 GB (to be measured in T-025), generation 8-10 GB | measured / estimated |

## 10. Probe measurements used in this note
One scratch probe (not part of the repository) on the five M1 images plus `kirkjufell` of `data/raw`, RTX 5070, torch 2.11.0+cu128, `lpips` 0.1.4:
LPIPS-Alex of a black/white canvas against the target 0.77 to 1.115 (all six maxima over the two canvases between 0.95 and 1.115); LPIPS-VGG 0.61 to 0.86; spatial LPIPS output `(1, 1, 512, 512)` whose mean equals the scalar to 1e-7;
15 canvases of 512²: Alex 0.016 s and 1.06 GB peak, VGG 0.58 s and 9.96 GB peak; repeated calls bit-identical. The numbers are inputs to decisions D5, section 2.3 and section 9; they are re-measured by the T-026 tests.

## 11. Limits of the protocol (stated, not fixed)
- LPIPS was trained on 64 x 64 patches and judged on natural-image distortions; applying it to a 512² painting made of strokes is common but unvalidated. It is the best held-out perceptual metric available, not ground truth; the planned blind preference study (M6) is the check on human preference.
- 100 images (90 test) detect moderate effects only; the categories are small. Per-category results are exploratory.
- The evaluation saliency (a DUTS-trained salient-object detector) can be wrong on paintings (domain shift), on abstract works and on SDXL images with several subjects; the control and the sign check of section 4.2 limit the damage but do not remove it.
- The prefix protocol disadvantages methods whose earlier strokes are re-optimized with the later ones; the fixed-budget side check measures the size of that effect on a subset.
- Blank canvas colour differs between producers; handled by controls, not removed.
- The first 100 images are one draw; with more time a second independent set would support a replication.

## References
- Zhang, R., Isola, P., Efros, A. A., Shechtman, E., Wang, O., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 (LPIPS; code BSD-2-Clause).
- Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P., *Image Quality Assessment: From Error Visibility to Structural Similarity*, IEEE TIP 13(4), 2004 (SSIM).
- Ding, K., Ma, K., Wang, S., Simoncelli, E. P., *Image Quality Assessment: Unifying Structure and Texture Similarity (DISTS)*, IEEE TPAMI 44(5), 2022 (code MIT).
- Dean, T., Boddy, M., *An Analysis of Time-Dependent Planning*, AAAI 1988; Zilberstein, S., *Using Anytime Algorithms in Intelligent Systems*, AI Magazine 17(3), 1996.
- Kim, T., Kim, K., Lee, J., Cha, D., Lee, J., Kim, D., *Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)*, ACCV 2022, arXiv 2209.09475 (code MIT).
- Kirillov, A. et al., *Segment Anything*, ICCV 2023; Qin, X. et al., *U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection*, Pattern Recognition 106, 2020; Hou, X., Zhang, L., *Saliency Detection: A Spectral Residual Approach*, CVPR 2007.
- Linardos, A., Kümmerer, M., Press, O., Bethge, M., *DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling*, ICCV 2021, arXiv 2105.12441 (considered, rejected on licence grounds).
- Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., Durand, F., *What Do Different Evaluation Metrics Tell Us About Saliency Models?*, IEEE TPAMI 41(3), 2019.
- Huang, Z., Heng, W., Zhou, S., *Learning to Paint With Model-based Deep Reinforcement Learning*, ICCV 2019, arXiv 1903.04411; Ganin, Y. et al., *Synthesizing Programs for Images using Reinforced Adversarial Learning (SPIRAL)*, ICML 2018.
- Liu, S. et al., *Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*, ICCV 2021, arXiv 2108.03798; Zou, Z. et al., *Stylized Neural Painting*, CVPR 2021, arXiv 2011.08114.
- Podell, D. et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, arXiv 2307.01952 (ICLR 2024); CreativeML Open RAIL++-M licence, 26 July 2023.
- Wilcoxon, F., *Individual Comparisons by Ranking Methods*, Biometrics Bulletin 1(6), 1945; Demsar, J., *Statistical Comparisons of Classifiers over Multiple Data Sets*, JMLR 7, 2006; Holm, S., *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of Statistics 6(2), 1979;
  Kerby, D. S., *The Simple Difference Formula*, Comprehensive Psychology 3, 2014; Efron, B., *Bootstrap Methods: Another Look at the Jackknife*, Annals of Statistics 7(1), 1979; Dror, R., Baumer, G., Shlomov, S., Reichart, R., *The Hitchhiker's Guide to Testing Statistical Significance in NLP*, ACL 2018;
  Kaplan, E. L., Meier, P., *Nonparametric Estimation from Incomplete Observations*, JASA 53(282), 1958; Dolan, E. D., Moré, J. J., *Benchmarking Optimization Software with Performance Profiles*, Mathematical Programming 91, 2002.
- Colas, C., Sigaud, O., Oudeyer, P.-Y., *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments*, arXiv 1806.08295, 2018; Henderson, P. et al., *Deep Reinforcement Learning That Matters*, AAAI 2018; Agarwal, R. et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021.
- The Metropolitan Museum of Art, *Open Access* and *Collection API* documentation, <https://metmuseum.github.io/> (CC0; accessed 2026-10-25).

## Amendments
### 2026-10-26 (T-024, before any method run): realization of section 1

No test-set number exists yet, so none of these is *post hoc*. What changed, and why:

1. **Photograph sources (1.1, 1.2.5).** 8 of the 10 portraits are Met Open Access photographs (CC0, nineteenth century, monochrome or sepia;
   they use 8 of the at most 10 greyscale photographs of 1.3), the 22 other photographs are Wikimedia Commons files with a public-domain tag from US
   federal collections (Library of Congress FSA/OWI Kodachrome slides, US Fish and Wildlife Service, one Prokudin-Gorsky plate). Reason: a
   colour portrait pool of enough clear faces was not available under a machine-readable licence. Sub-strata of the 20 other photographs: 7 animals, 6 objects, 7 scenes. Smithsonian Open Access is not used.
2. **Border trimming (1.3).** Uniform black film or mat borders are cut by a mechanical rule (near-black flat rows/columns, scanned inward over at most a
   quarter of the side; for scanned slides a row with 90 % of its pixels below 40), recorded as a `crop_box` in the manifest; the trimmed square must be at least 640 px (the 768 px floor stays the rule for
   the untrimmed source). Reason: the dark film-holder edges of the slides would otherwise count as a frame over 5 % of the area.
3. **Resize filter (1.3).** The box (area) filter of Pillow, not `center_crop_resize` (bicubic with antialiasing): it is the "area" filter
   of the protocol and does not depend on the torch version, which keeps the SHA-256 of the prepared pixels stable.
4. **Text rule (1.2.3).** Applied to captions, labels, watermarks and overlays. Inscriptions that are part of an East Asian painting or print are kept below a third of the area (one album leaf of calligraphy was rejected).
5. **Asian and "other" strata.** 4 paintings + 4 woodblock prints; 4 Modern Art + 4 American Wing paintings (the American Wing has an empty classification: `objectName = Painting`).
6. **Verification by pixels.** A source file whose bytes changed but whose decoded prepared pixels are identical passes (two Met files were re-encoded within a day); a pixel mismatch fails.

Affected runs: none (the set did not exist). Record of the check against the tuning images: `docs/experiments/data/eval_set/overlap_check.json`.

### 2026-10-26 (T-024, review round 1, still before any method run)

Defects found by the review of the first realization, corrected and redrawn (the manifest, the dev split and the records changed; no run existed):

1. **Dev to test leak.** Two views of one scene (same photographer, same shop, same car) were one dev and one test image. The curation now refuses a second
   view of a series (same normalized photographer and a shared place category or a similar title; not applied to agency wildlife photographs) and the final set is checked for it (`build_eval_set check`, 0 pairs).
2. **Frames, mats and calibration targets** over 5 % of the area were accepted for three Met images; rejected (the visual rejections list grew to 35 entries with reasons).
3. **"Clear subjects".** Animal photographs with no animal or an animal a few pixels tall were rejected. The animal stratum is drawn from the US Fish and Wildlife Service
   photographs, only if they carry a species category, name the animal in the title and mention no person, staff or event; images with a person who may be alive
   are rejected on the contact sheet.
4. **"At most 3 images per photographer"** now applies to normalized names (it was counted on the raw `Artist` string).
5. Smaller points: the hand-coloured Met print is `colour: true`; dates of Commons files are cleaned of their Wikidata residue; the data card lists the dev count once per stratum;
   `fetch_eval_set --sdxl` no longer hides a failure of the downloaded images behind the SDXL result.

Affected runs: none. The replay of the curation (`python -m scripts.build_eval_set curate`, same cached API data and live Commons) reproduces the manifest.

### 2026-10-26 (T-024, review round 2, still before any method run)

1. **Dark film borders (1.2.3).** The trimming and the check scanned from the outermost row and stopped at a 1-2 px light edge line, so three slides kept 6.4 % of near-black
   border while the record said 0.4 %. Both now skip up to three light edge lines (scaled to the source size) before the band; the three slides were redrawn or trimmed, the check
   finds at most 4.7 % on the FSA slides, and the measure is applied to the slides only (dark backgrounds of paintings and animals are content).
2. **Strata.** The keyword "mill" put a mountain scene in the object stratum: it is removed from the object words (a few object words added), and portraits, objects and scenes were re-verified on the contact sheets. Prokudin-Gorsky plates are dropped (sepia reproductions, not colour).
   The cap per photographer is 4 (it was 3): without the Prokudin plates the FSA pool could not fill the strata at 3.
3. **People in agency photographs.** The wording filter was not enough: the contact sheets still showed staff and visitors. A COCO person detector (torchvision Faster R-CNN MobileNetV3, score above 0.4, any size) now rejects any agency
   photograph with a person, in the draw and again in `build_eval_set check` (0 detections on the final set). The word list of the filter matches substrings (it over-excludes, never under-excludes), left as is.

Affected runs: none. The dev split was redrawn with the rest.

### 2026-10-26 (T-034, finish of the curation: review round 3 and a systematic audit, still before any method run)

Review round 3 found three more defects after two rounds of reactive fixes, so the set was audited systematically (`scripts/audit_eval_set.py`: every machine-checkable rule on every image, 0 violations; `docs/experiments/data/eval_set/visual_audit.md`: all 100 images looked at, one line each).

1. **The agency word filter was dead** (two backspace characters instead of the word boundaries): none of the draws had applied "no word about people, staff or events". Repaired, with a test that matches the words and never matches "manatee"/"Mandarin"/"Dayton", and a test that no text file of the repository holds a control character. Words for non-animal titles (nest, fire, smoke, habitat, tracks) were added after the first redraw showed titles that name an animal that is not in the picture.
2. **One border definition** for the trimming and the report: `flat` (near-black uniform lines) for paintings, prints and mats, `film` (mean luminance below 30, after a thin light edge line) for scanned slides; the report is computed on the cropped source at full resolution, and independently at 512 px for slides. A slide whose film bar sits next to a dark blue sky (`gus_worke`) cannot be told from its content by any rule and was rejected.
3. **Children and possibly living persons.** The three views of Shulman's Market (children on the step and in the street) were rejected; the rule is applied to every FSA or agency photograph: no child, no person identifiable who may be alive (adults of a 1940-43 slide cannot be alive; the Met sitters are long dead, including one girl).
4. **Identity of the files used.** Found by the audit and by the fetch from scratch: the Commons ids were the first 60 characters of the title, so four files of one series shared an id, and so a cache file; the manifest of one image held the hash, size and crop of a different file than its URL. Ids of colliding titles now carry a hash suffix; a cached file is only reused if its SHA-1 equals the one the API publishes (`audit_eval_set` checks it for every Commons image); the size in the manifest is that of the file used. Only that one image was affected (replaced).
5. **Composition** (counts fixed by the protocol are unchanged): the colour photographs are 14 FSA/OWI slides and 7 US Fish and Wildlife Service photographs; the monochrome Met portraits are 9 (the portrait stratum has 9 + 1), because the FSA pool holds one usable colour portrait. Colour photographs: 21 (minimum 20), greyscale 8 (maximum 10). The visual rejections list holds 52 entries with reasons.

Affected runs: none. The curation replay and the fetch from scratch reproduce the set.

### 2026-10-26 (T-025, evaluation saliency maps, before any method run)

Realization of section 4; nothing in the metric changes, the measured facts are recorded.

1. **Model and pins.** InSPyReNet SwinB trained on DUTS-TR, code at commit `692ad8a5` (MIT, 15 files pinned by SHA-256), checkpoint SHA-256 `6a352536...faee` (367,526,592 bytes, 90.7 M parameters), `configs/eval/saliency.yaml`. The checkpoint comes from the Google Drive file of the model zoo (the OneDrive mirror answers 403). **No licence is stated for the checkpoint** (README, model zoo, LICENSE checked on 2026-10-26): it is fetched, never committed or redistributed, and used only as a measuring instrument; this is the same status as the Paint Transformer weights.
2. **Maps.** Float32 `(512, 512)`, in [0, 1] as returned by the model, no re-scaling. A 512 px target is below the model's pyramid threshold (512), so the model sees the 384 px bilinear resize (ImageNet mean and std), returns its own min-max normalized map, which is resized to 512 with `align_corners=True` (the `run/Inference.py` path of the pinned commit); TF32 is off. The control, the spectral residual, is computed on the 64 x 64 greyscale image (3 x 3 local average of the log spectrum, Gaussian sigma 8, bilinear resize, min-max), and is zero on a flat image. Both are functions of the target only.
3. **Measured (RTX 5070, 100 images).** InSPyReNet: 95.5 ms per image, peak VRAM 692 MB allocated / 830 MB reserved (the estimate was under 3 GB); spectral residual: 6.6 ms per image on the CPU. The maps are **bitwise deterministic**: a recomputation reproduces the recorded SHA-256 of all 200 maps and two runs of the model give identical bytes. The mean value of the map is 0.30 / 0.25 / 0.29 for InSPyReNet on paintings / photographs / SDXL, 0.36 to 0.37 for the control; the two maps correlate at 0.32 to 0.35 on average, so the control is a genuinely different, much blurrier map: a SW-LPIPS claim that keeps its sign with both is robust, as section 4.2 requires.
4. **Circularity rule enforced.** `tests/test_circularity.py` fails if any method package (baselines, contribution, renderer, stroke_model, animation, generation, utils) or any script other than the two saliency scripts mentions InSPyReNet or `brushstroke.eval.saliency`. The M4 plan must name a different model (Segment Anything family, a U2-Net-type network, or a lightweight net trained by us).
5. **Known property of the control.** A spectral-residual map can put its peak on the border of the image (the transform is periodic) for a lone object on a flat background or next to the border; it is a control judged by sign agreement, not a second evaluator.

Affected runs: none.

