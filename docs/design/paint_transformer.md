# Paint Transformer: study note, reproduction targets (T-010)

Exact, cited specification of *Paint Transformer* (PT; Liu, Lin, He, Li, Deng, Li, Ding, Wang, ICCV 2021, arXiv:2108.03798) from the paper
and from the author's code, so that the M2 reimplementation and its gaps can be judged against facts.

**Sources and how they are cited**
- **[P §x]** = the paper (arXiv v1 HTML rendering, read in full; sections, equations and Tables 1-2 as numbered there).
- **[C file:function]** = <https://github.com/Huage001/PaintTransformer> at commit **`7960b8eef2f6af7171af01933a3d8dbc73190a8d`**
  (2021-10-20, "Update README.md"), files read in full: `inference/{inference,network,morphology}.py`, `train/train.py`,
  `train/train.sh`, `train/models/{painter_model,networks,base_model}.py`, `train/options/*.py`, `train/util/morphology.py`,
  `train/data/null_dataset.py`. Checksums are pinned in `scripts/pt_reference_study.py`.
- **[M]** = measured here by running that code/the released weights (`python -m scripts.pt_reference_study ...`; numbers in `outputs/pt_reference/`, not committed).

**Important caveat on "official".** The paper footnote says the released implementation is PaddlePaddle. The repository above is described in its
README as the *"officially unofficial PyTorch re-implementation"* by the first author (Songhua Liu); the Paddle repository
(`wzmsltw/PaintTransformer`) was **not** read. The paper's numbers come from the Paddle implementation, and the PyTorch code differs from the
paper in several places (section 9). Where paper and code disagree this note says so, and the reproduction must pick one explicitly.

## 1. Stroke parameterization and renderer

**Stroke** [P §3.2]: straight-line stroke `{x, y, h, w, theta, r, g, b}`, centre `(x, y)`, height `h`, width `w`, rotation `theta`, colour. Code order is
`x0, y0, w, h, theta, R, G, B` [C inference.py:param2stroke]: all normalized to the canvas the stroke is rendered on; `theta in [0, 1]` means `theta * pi`
radians [C inference.py:param2stroke: `sin_theta = torch.sin(acos(-1) * theta)`]. No opacity parameter.

**Rendering** [P §3.2 Eq. 3-4; C inference.py:param2stroke, train/models/painter_model.py:param2stroke]:
1. Two brush images (vertical, horizontal) stored as `meta_brushes = cat([brush_large_vertical, brush_large_horizontal])`; each is a 394 x 394 PNG read as
   grayscale `L` in [0, 1] [C inference.py:main; train PainterModel.__init__]. Brush **selection**: `index[h > w] = 0` (vertical), `index[h <= w] = 1`
   (horizontal) - no swapping of `w` and `h`, no rotation by 90 degrees [C param2stroke]. The `small` brush images exist in the repo but **no code path uses them**.
2. Affine warp with `F.affine_grid(warp, [b, 3, H, W], align_corners=False)` + `F.grid_sample` (bilinear, zeros padding), with the matrix
   `warp_00 = cos/w, warp_01 = sin H/(W w), warp_02 = (1-2x0) cos/w + (1-2y0) sin H/(W w), warp_10 = -sin W/(H h), warp_11 = cos/h,
   warp_12 = (1-2y0) cos/h - (1-2x0) sin W/(H h)` [C param2stroke] - **identical to ours** (`brushstroke/renderer/rasterize.py:stroke_affine`, derived
   from the same code in T-003).
3. **Alpha (binarized + eroded).** Inference: `brush = grid_sample(resize(meta_brush, (H, W)), grid)`, `alphas = (brush > 0).float()`, then
   `alphas = morphology.erosion(alphas)` [C inference.py:param2stroke; morphology.py:erosion]: a 3 x 3 min filter with border value `+1e9`.
   Training differs: `alphas = (brush > 0).float()` is computed **before** warping, the binary image is warped (so alpha is fractional at edges),
   and `Erosion2d(m=1)` is applied after [C train painter_model.py:param2stroke, forward]. The inference path also first downsizes the meta brush with
   `F.interpolate(meta_brushes, (H, W))` (default *nearest*) [C inference.py:param2stroke], the training path samples the 394 px image directly.
4. **Foreground (colour).** `foreground = brush_gray * colour` (the brush *intensity* modulates the colour, so bristle texture is a colour variation, not
   an opacity variation) followed by a 3 x 3 **dilation** (`max` filter, border `-1e9`) [C inference.py:param2stroke; train forward]. Training code also contains an
   unused colour gradient (`R0 -> R2` along the stroke); the generator sets `R2, G2, B2 = R0, G0, B0` so strokes are uniformly coloured
   [C train painter_model.py:set_input: `gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]`].
5. **Compositing** [P Eq. 4]: `canvas = foreground * alpha * decision + canvas * (1 - alpha * decision)` for strokes in order
   [C inference.py:param2img_parallel:partial_render; train forward]; `decision` is the binary confidence (0/1), so unconfident strokes are skipped.
   Initial canvas = **black zeros** [C inference.py:main `final_result = torch.zeros_like(...)`; train `old = torch.zeros(...)`].

**Brush images [M]:** after `convert('L')`, `brush_large_horizontal/vertical` (inference/ and train/ copies are different files but convert to nearly the same
image; 56.6 % vs 57.6 % of pixels > 0) cover about 57 % of the 394 x 394 square: a ragged blob spanning x 65-359, y 10-365 (about 75 % x 90 % of the
box), bristles along x (horizontal) or y (vertical). So a PT stroke's box `w x h` is *larger* than its visible footprint. Brush files are Apache-2.0 repository assets.

### Mapping PT -> our T-003 convention

| aspect | Paint Transformer | ours (T-003, `brushstroke/renderer`) | identical? / what to do for a PT-compatible mode (T-011) |
|--------|------------------|--------------------------------------|---------------------------------------------------------|
| parameters | `x, y, w, h, theta, r, g, b` | `x, y, w, h, theta, r, g, b, a` | same order and meaning; PT = ours with `a = 1`; **identical** warp matrix and normalization (fractions of canvas width / height) |
| theta | `theta * pi` | `theta * pi` | identical (periodic, `theta in [0, 1]`) |
| parameter ranges | network outputs are raw reals, interpreted as normalized values with **no clamp or sigmoid** (`x, y` about in [0, 1]) | `clamp_params` / sigmoid re-parameterization | keep raw outputs in the PT model; clamp only when exporting to a `StrokeSequence` |
| brush orientation | picks the *vertical* image when `h > w`, else the *horizontal* one; no swap | canonicalizes to `w W >= h H` (swap pixel sizes, theta + 0.5) and uses one horizontal mask | same box and rotation footprint; texture differs slightly (PT's vertical image is not the transpose of its horizontal one). **PT mode: two brush images + the `h > w` rule, no canonicalization** |
| brush alpha | **binarized** (`> 0`), then **3 x 3 erosion** | soft mask value = alpha (texture in opacity) | **PT mode: `alpha_mode`** = `pt_infer` (sample, `> 0`, erode) and `pt_train` (binarize, sample, erode); both exist in the official code |
| colour | `brush_gray * colour`, then 3 x 3 **dilation** | flat colour, alpha carries texture | **PT mode: foreground image = dilate(sampled gray * colour)**, composited with the binary alpha |
| opacity | none | `a` | `a = 1`, ignored in PT mode |
| initial canvas | black | white by default (`canvas` argument) | pass `canvas=[0, 0, 0]` (supported) |
| rendering extent | each stroke is rendered inside a patch window (32 x 32 in training, 64 x 64 windows in inference) and clipped to it | full canvas | equivalent except for clipping at the window border (strokes extending more than 16 px beyond their 32 px patch at layer resolution are cut in PT); window clipping can be added in the T-015 adapter |
| compositing order | per layer: 4 phases (even/even, odd/odd, odd/even, even/odd windows), within a phase strokes `0..7` in index order | index order of the `StrokeSequence` | the adapter orders strokes as (layer, phase, stroke index, patch raster order) so the full-canvas compositing equals PT's |
| brush texture resampling | meta brush nearest-resized to the window then bilinear warped (inference); direct bilinear from 394 px (training) | bank masks (64 x 128) warped bilinearly | small texture differences; irrelevant for training-parity if the brush images are loaded at 394 px |

**Patch coordinates to canvas coordinates (layer `l`, patch grid `n = 2^l`, patch `(i, j)`).** The network predicts `x_p, y_p, w_p, h_p` in units of the 32 px
patch; inference rewrites them as `x/2 + 0.25`, `w/2` [C inference.py:main] relative to a 64 px window that is the patch plus 16 px on each side, which is the same
stroke as `x = (j + x_p)/n`, `y = (i + y_p)/n`, `w = w_p/n`, `h = h_p/n` in canvas fractions (the layer canvas is `32 n` px). The final shifted layer uses patches offset by 16 px at
512 px resolution: `x = (32 j - 16 + 32 x_p)/512`, `j, i in 0..16`. Non-square images are zero-padded (black) and centered to a `32 * 2^K` square and cropped back [C inference.py:pad, crop].

## 2. Synthetic training data [C train/models/painter_model.py:set_input]

Everything is generated on the fly (no dataset); `NullDataset` only yields indices (`__len__ = max_dataset_size = 16384`) [C data/null_dataset.py; train.sh].
Per training step with batch size `B` (64 in `train.sh`):
- **Patch size** `P = 32` [P §4.1; C `self.patch_size = 32`]; **strokes per patch** `N = 8` [P §4.1; `--used_strokes 8`].
- **Background canvas** ("old"): `B/4` random canvases of 64 x 64, each with 8 strokes `x, y, w, h ~ U(0.2, 0.7)` (`rand * 0.5 + 0.2`), `theta ~ U(0, 1)`,
  colour `~ U(0, 1)^3`, all valid, drawn over a black canvas, then cut into 2 x 2 patches of 32 x 32 (so background strokes cross patch borders and `B` patches result).
- **Target** (`render`, "I_t"): the background patch plus `N = 8` foreground strokes sampled the same way, rendered **directly on the 32 x 32 patch** (strokes are clipped
  at the patch border). Strokes are generated in sequence and stroke `i` is marked **invalid** if, for some earlier *valid* stroke `j`, `sum(alpha_i * alpha_j) / sum(alpha_j) >= 0.75`
  (stroke `i` covers 75 % or more of stroke `j`'s eroded alpha); invalid strokes are not rendered in the target.
  [P §4.1 says the threshold is **60 %**; the code uses 0.75: deviation.]
- **Statistics [M]** (200 batches of 64 = 12 800 patches): valid target strokes per patch 6.92 +- 0.91 (histogram over 0..8: 0, 0, 1, 4, 107, 761, 2857, 5331, 3739);
  mean |target - background| per pixel 0.117; background mean intensity 0.18, target 0.26.
- Only the 5 shape parameters of valid strokes are supervised (colour is not predicted, section 3); no augmentation, no image dataset, no validation set in the code.

## 3. Model [P §3.3, §4.1; C inference/network.py:Painter, train/models/networks.py:Painter]

- **Feature extractors:** two independent CNNs (image, canvas), each `ReflectionPad(1) - Conv3x3(3, 32, s1) - BN - ReLU - ReflectionPad - Conv3x3(32, 64, s2) - BN - ReLU - ReflectionPad - Conv3x3(64, 128, s2) - BN - ReLU`:
  two stride-2 downsamplings, a 32 x 32 input becomes an 8 x 8 x 128 map [P §4.1: "three Conv-BatchNorm-ReLU blocks with two 1/2-scale down-sampling operations"].
- `conv 1x1` on the channel concatenation (256 channels) to the hidden size `256`; learned positional encoding `row_embed, col_embed in R^{8 x 128}` (initialized `torch.rand`)
  concatenated to 256 and **added** to the features (the paper says "concatenated"; the code adds) [C Painter.forward].
- **Transformer:** `torch.nn.Transformer(d_model=256, nhead=8, 3 encoder layers, 3 decoder layers)` with PyTorch defaults (`dim_feedforward=2048`, dropout 0.1, ReLU, post-LayerNorm); encoder input = 64 tokens (8 x 8),
  decoder input = **8 learnable query vectors** `query_pos in R^{8 x 256}` (`torch.rand`) repeated over the batch [P §3.3: "N learnable stroke query vectors"].
- **Heads:** `linear_param`: Linear-ReLU-Linear-ReLU-Linear(256 -> 5), `linear_decider`: Linear(256 -> 1). **Only 5 shape parameters are predicted.**
  The **colour is not predicted: it is read from the target image at the predicted centre** with `grid_sample(img, 2*(x, y) - 1)` [C networks.py:Painter.forward;
  inference.py:main], differentiably during training. (The paper lists `r, g, b` as stroke parameters and says the predictor outputs stroke parameters [P §3.2-3.3]: deviation.)
- **Confidence:** `Sign` with sigmoid gradient: forward `(x > 0)`, backward `sigmoid(x)(1 - sigmoid(x))` [P Eq. 6; C SignWithSigmoidGrad].
- **Initialization** [C networks.py:init_weights]: Conv and Linear weights `N(0, 0.02)`, biases 0; BatchNorm weight `N(1, 0.02)`, bias 0.
- **Parameter count [M]: 9,072,902 trainable parameters** (9,073,804 in the state dict with BatchNorm buffers): image CNN 93,696, canvas CNN 93,696, 1x1 conv 65,792,
  Transformer 8,682,496, parameter head 132,869, decision head 257, embeddings 2,048 + 1,024 + 1,024. (The M2 estimate of 10-15 M was high; it is ~9.1 M.)

## 4. Losses [P §3.4; C train/models/painter_model.py:optimize_parameters]

`rec` = the canvas after rendering the predicted strokes with binarized confidence; targets `render`.
- **Pixel loss:** `L1(rec, render)` (mean over pixels) x `lambda_pixel = 10` [C]; [P Eq. 7: plain L1, no weight given].
- **Hungarian matching** per patch between the 8 predictions and the *valid* ground-truth strokes (`scipy.optimize.linear_sum_assignment`, on CPU, per sample),
  cost = `cdist(pred5, gt5, p=1)` (L1 over the 5 shape parameters) `+ W-distance + (1 - decision_logit)` [C: `cost_matrix_decision = (1 - decision)` with `decision` the raw logit].
  [P Eq. 12: `g_v (D_L1 + D_W + D_bce)` with a BCE cost: deviation.] Invalid GT strokes are excluded from the matching (the same effect as `g_v = 0`).
- **Stroke L1** (`loss_gt`): `L1Loss` (mean over the matched pairs and the 5 parameters) x `lambda_gt = 1`.
- **Wasserstein stroke loss** (`loss_w`): `gaussian_w_distance` between the 2-D Gaussians of the strokes, `|mu_1 - mu_2|^2 + (w1^2 + h1^2)/4 + (w2^2 + h2^2)/4 - 2 Tr((S1^{1/2} S2 S1^{1/2})^{1/2})`
  [C gaussian_w_distance, get_sigma, get_sigma_sqrt; P Eq. 9-10, with `w/2, h/2` as in Eq. 9; the code's covariance entries use `w^2/4` and `h^2/4` for `S` and `w/2, h/2` for `S^{1/2}`],
  mean over matched pairs x `lambda_w = 10`.
- **Confidence loss** (`loss_decision`): `BCEWithLogitsLoss(pos_weight = lambda_recall = 10)` over **all** `B x 8` predictions, positives = the predictions matched to a valid GT stroke,
  x `lambda_decision = 10`.
- **Total:** `loss = 10 * L1_pixel + 1 * L1_stroke + 10 * W + 10 * BCE` [C defaults in `painter_model.py:modify_commandline_options`; `train.sh` does not override them].
  [P §4.1 gives `lambda_r = 8, lambda_L1 = 1, lambda_W = 10, lambda_bce = 1` and no pixel weight: deviation, section 9.]
- Stroke order is not supervised at stroke level [P §3.4]; the image-level pixel loss with sequential compositing is what regulates it.

## 5. Training hyperparameters

| item | paper | code (`train/train.sh`, `train_options.py`, `networks.py:get_scheduler`) |
|------|-------|------|
| optimizer | Adam, lr 1e-4 [P §4.1] | Adam, lr 1e-4, **betas (0.5, 0.999)** (`beta1` default 0.5) |
| batch size | 128 [P §4.1] | **64** (`--batch_size 64`) |
| length | 30,000 iterations [P §4.1] | `--n_epochs 200 --n_epochs_decay 20`, `--max_dataset_size 16384`: 220 epochs x 256 steps = **56,320 steps = 3.60 M samples** (paper: 30,000 x 128 = 3.84 M samples) |
| lr schedule | not described | custom `linear` policy: lr constant 1e-4 through epoch 204, then x0.3 every 5 epochs: 3e-5 (205-209), 9e-6 (210-214), 2.7e-6 (215-219), 8.1e-7 (220) (computed from `lambda_rule`: `0.3 ** max(0, (epoch + epoch_count - n_epochs) // 5)`) |
| precision | n/a | fp32, single GPU (DataParallel wrapper) |
| image / patch | `P = 32`, `N = 8` | same |
| augmentation | none | none (synthetic data) |
| hardware / time | one RTX 2080 Ti, "fewer than 4 hours" [P §4.1], 3.79 h for the stroke predictor [P Table 2] | **[M] 0.237 s/step on our RTX 5070 (fp32, batch 64, including data synthesis and CPU Hungarian matching) = 3.70 h for 56,320 steps; peak VRAM 3.6 GB** |
| validation | none reported | none in the code (the display shows `old / render / rec` images and the four loss terms) |
| seeds | one run | no seed set |

## 6. Inference [P §3.5, Algorithm 1; C inference/inference.py:main, param2img_parallel / param2img_serial]

1. `K = max(ceil(log2(max(H, W) / 32)), 0)` [P Eq. 14; C]; for 512 x 512, **K = 4** (the paper's default).
2. Image zero-padded (black) to `32 * 2^K` square. For layers `l = 0..K`: target and current canvas are resized to `32 * 2^l` with `F.interpolate` default mode (**nearest**),
   cut into non-overlapping 32 x 32 patches (`2^l x 2^l` of them), the model predicts 8 strokes per patch, colours are sampled from the target patch at the predicted centres,
   strokes with `Sign(logit) = 1` are rendered; the layer result stays at **its own resolution** and is nearest-upsampled by the next layer. So coarse strokes are painted at low
   resolution and nearest-upsampled (blocky edges at 512 px).
3. **Rendering windows:** stroke parameters are re-expressed in 64 px windows (patch + 16 px margin); windows are rendered in **four phases** (even-y/even-x, odd/odd, odd/even, even/odd window
   indices; windows of one phase do not overlap), inside a phase strokes `0..7` in order [C param2img_*].
4. **Extra final layer (shifted by half a patch):** after layer `K`, one more pass at 512 px with the image and canvas padded by 16 px so that patches straddle the previous patch borders (17 x 17 = 289
   patches, 2,312 strokes) [C main, after the loop; `has_border=True` in the serial mode]. [P Algorithm 1 does not describe it.] **Quirk:** in this pass the confidences are *not* binarized
   (`decision = stroke_decision.view(...).bool()` on raw logits) so **all 2,312 strokes are painted regardless of confidence**. [M] For the Mona Lisa only 1,502 of them have a positive logit.
5. Stroke counts [M] (official pretrained weights, 512 x 512, serial = False): layer 0: 8 (always all), layer 1: 28-32 of 32, layer 2: 98-128 of 128, layer 3: 320-511 of 512, layer 4: 1,021-2,000 of 2,048,
   final shifted layer 2,312 (forced). **Total 3,787-4,991 strokes (mean 4,336) of a maximum 5,040** on our five images; counting the final layer by confidence only, the Mona Lisa would have 3,566.
6. No explicit ordering besides (layer, phase, patch raster order within the phase, stroke index). `serial=True` renders the same strokes one index at a time to save frames for animation.
7. **Speed/VRAM [M]** (RTX 5070, parallel mode, includes model and brush loading): 0.5-0.9 s per 512 x 512 image, 2.3 GB peak (the paper reports 0.304 s at 512 on a 2080 Ti [P Table 2]).

## 7. Reported results to reproduce

Paper evaluation protocol [P §4.2]: images at 512 x 512, `K = 4`; real images: **100 landscapes (Kaggle "landscape-pictures"), 100 artworks (WikiArt), 100 portraits (FFHQ)**; metrics `L_pixel`
(L1, Eq. 7) and `L_pcpt` (VGG perceptual loss of Johnson et al.; layer and normalization not specified); for strokes, synthesized stroke images are fed to PT and Optim and the predicted strokes are compared
to the synthesized ones with `L_pixel`, `D_L1` and `D_W` (number of images and generator settings not specified). "Optim is applied with the same number of strokes as Ours" (Table 1 caption).

| # | result | numbers | reproducible without their data? | verdict |
|---|--------|---------|----------------------------------|---------|
| T1a | Real images, `L_pixel` / `L_pcpt` (lower better): RL [10] / Optim [39] / PT | 0.040 / 0.737; 0.059 / 0.856; **0.056 / 0.807** | datasets are Kaggle / WikiArt / FFHQ: licensing and access prevent committing them; partial replacement by our public-domain images (5 images in `docs/data_sources.md`) | **partially**: reproduce PT's `L_pixel` and `L_pcpt` on our PD set and compare *relative orderings* (PT vs our optimization baseline vs RL); absolute values are not comparable. [M] official pretrained model on our 5 images: L1 0.041-0.101 (mean 0.068), see below |
| T1b | Random strokes, `L_pixel`, `D_L1`, `D_W`: Optim / PT | 0.073, 0.137, 0.057; **0.042, 0.083, 0.018** (RL: 0.058, no D) | yes, with an assumed generator (their exact protocol is unspecified) | **feasible, with documented assumptions** (use the official `set_input` distribution) |
| T1c | Ablations (w/o `L_pixel`, `D_L1`, `D_W`, `D_bce`): `L_pixel`, `L_pcpt`, then stroke metrics | e.g. w/o pixel 0.081, 1.012, 0.068, 0.241, 0.024; w/o `D_L1` 0.074, 0.941, 0.077, 0.267, 0.019; w/o `D_W` 0.069, 0.947, 0.046, 0.113, 0.034; w/o `D_bce` 0.071, 0.928, 0.052, 0.093, 0.021 | yes in principle (4 extra trainings, ~3.7 h each at the official recipe) | **feasible**, needs four additional runs (T-014+ budget) |
| T2a | Inference time (s) at 128 / 256 / 512: PT / RL / Optim | 0.055 / 0.124 / 0.304; 0.242 / 0.266 / 0.322; 76.5 / 161.6 / 521.4 | yes, on our hardware | **feasible** for PT and for our own optimization baseline; absolute values differ (RTX 5070 vs 2080 Ti): compare ratios |
| T2b | Training time (h): PT stroke predictor / RL / Optim renderer | 3.79 / 40 (+ 5-15) / 10.16 | yes for PT | **feasible** (3.7 h measured for the official recipe on our GPU); RL cost is not needed |
| T2c | "Use external dataset": No / Yes / No | - | - | by construction |
| F5-8 | Qualitative figures: comparison with Optim/RL, effect of `K` (Fig. 5), loss ablations (Fig. 6), other brushes (Fig. 7), style transfer (Fig. 8) | qualitative | partly | Fig. 5 (`K`) and Fig. 7 (rectangle/circle brush swap) are cheap; Fig. 8 needs LapStyle/AdaAttN (out of scope) |
| - | User study | **none in this version of the paper** | - | n/a |

**Reference point [M]: the released pretrained model on our five evaluation images (512 x 512, same crops and metrics as M1, black initial canvas).** It is *not* a target, it anchors the comparison.

| image | strokes | L1 | PSNR | SSIM | LPIPS-VGG | LPIPS-Alex |
|-------|---------|----|------|------|-----------|------------|
| mona_lisa | 4,376 | 0.0436 | 24.3 | 0.449 | 0.653 | 0.621 |
| starry_night | 4,991 | 0.0824 | 18.8 | 0.240 | 0.617 | 0.497 |
| great_wave | 4,330 | 0.1005 | 15.5 | 0.421 | 0.570 | 0.508 |
| pearl_earring | 3,787 | 0.0407 | 22.7 | 0.541 | 0.567 | 0.476 |
| earth_apollo17 | 4,194 | 0.0705 | 18.2 | 0.407 | 0.505 | 0.369 |
| **mean** | 4,336 | 0.0675 | | | 0.582 | 0.494 |

For comparison, our optimization baseline (M1/T-009 default) reaches mean L1 0.051, LPIPS-VGG 0.308 and LPIPS-Alex 0.359 with **2,037** strokes on the same images (and the demos' L1 0.024-0.071).
PT's design goal is abstraction with a feed-forward model, not fidelity: it paints ~2 x more strokes with a worse LPIPS, which matches the paper's own statement that RL has the best content fidelity and PT/Optim are more abstract [P §4.2];
a reproduction should expect PT-like LPIPS (0.5-0.65 VGG), not ours.

Feasibility summary: **reproducible**: T1b, T1c, T2a, T2b, Fig. 5/6/7. **Partially / protocol-adapted**: T1a (datasets unavailable; use PD images + the official model as an anchor). **Not reproducible**: absolute T1a values, the RL numbers without the RL code (M3).

## 8. Licenses and reuse

- **Code (`Huage001/PaintTransformer`)**: Apache License 2.0 (`LICENSE` at the commit; SHA-256 `c71d239d...`). Reading, running outside git and citing are fine; copying code into this repository would require keeping the notice. We reimplement from the paper and this specification, and do not vendor files.
- **Brush images** (`inference/brush/*.png`, `train/brush/*.png`): part of the Apache-2.0 repository. Committing them is allowed with attribution but conflicts with our convention "own-generated or PD/CC0 only"; **recommendation: fetch them with checksums into `data/raw/` (not committed) for the PT-compatibility mode and compare against our own procedural brushes**; the choice is for the chef.
- **Pretrained weights**: the authors' Google Drive file (36.3 MB, SHA-256 `1fbd761ac908c1b53ebc7c9a28e52b8a8afb332f77819c718fe472b7bf6e9dde`) linked from the README. **No separate license is stated** for the weights; we treat them as research reference
  material: download to `data/raw/pt_official/` (gitignored) and never redistribute. Loaded with `torch.load(..., weights_only=True)` because it is an internet pickle. Running the official model as a reference point is therefore possible outside git (done here).
- The paper (arXiv) is cited, not copied. Datasets of the paper: Kaggle landscape-pictures, WikiArt, FFHQ: not used.

## 9. Paper vs official PyTorch code: discrepancy list (the reproduction must choose)

| item | paper | code (commit above) | recommendation for the reproduction |
|------|-------|----------------------|--------------------------------------|
| stroke predicts colour | stroke has `r, g, b`; predictor outputs stroke parameters | predictor outputs 5 shape parameters; colour sampled from the target at the predicted centre | **follow the code** (it is what the released weights do); keep colour prediction as a flagged ablation |
| invalid-label threshold | 60 % of a previous stroke's area | 75 % (`iou < 0.75`) | follow the code; flag |
| loss weights | `lambda_r = 8`, `L1 = 1`, `W = 10`, `bce = 1`, no pixel weight | pixel 10, gt 1, W 10, decision 10, recall (pos_weight) 10 | follow the code; the paper numbers cannot be matched to either exactly |
| matching cost | `D_L1 + D_W + D_bce` | `L1 + W + (1 - logit)` | implement both as a flag (default code) |
| batch size / length | 128 / 30,000 iterations | 64 / 56,320 steps (3.6 M samples) | equal sample budget: use the code's schedule |
| Adam betas | not stated | (0.5, 0.999) | follow the code |
| positional encoding | concatenated | added | follow the code |
| final shifted layer | not described | extra pass, all strokes painted (confidence ignored) | decide explicitly; report both variants |
| train vs inference alpha | n/a | train: binarize then warp then erode; inference: warp then binarize then erode | both modes in T-011 |
| train window vs inference window | n/a | train clips strokes at the 32 px patch, inference at 64 px windows | note as a train/test gap; optional experiment |
| small brushes | n/a | present, unused | ignore |

Open questions that cannot be settled from these sources: which of the two implementations (Paddle vs PyTorch) produced each paper number; the exact synthesized-stroke protocol of Table 1; the layer of the VGG perceptual loss.

## 10. How the numbers of this note were produced

`python -m scripts.pt_reference_study fetch|reference|datastats|trainspeed` (official files pinned by SHA-256, weights by SHA-256; outputs in `outputs/pt_reference/`). Environment: RTX 5070, PyTorch 2.11.0+cu128. The wrapper counts
`decision.sum()` per layer by wrapping `param2img_parallel`; metrics are the M1 `brushstroke.eval.metrics.image_metrics` on the same 512 x 512 center crops as the baseline.
