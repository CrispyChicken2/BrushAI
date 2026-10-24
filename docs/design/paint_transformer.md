# Paint Transformer: study note, reproduction targets (T-010)

Exact, cited specification of *Paint Transformer* (PT; Liu, Lin, He, Li, Deng, Li, Ding, Wang, ICCV 2021, arXiv:2108.03798) from the paper
and from the author's code, so that the M2 reimplementation and its gaps can be judged against facts.

**Sources and how they are cited**
- **[P §x]** = the paper (arXiv v2, read in full from its HTML rendering via ar5iv; the review cross-checked the text against `arxiv.org/html/2108.03798v2`; sections, equations and Tables 1-2 as numbered there).
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
| rendering extent | each stroke is rendered inside a patch window (32 x 32 in training; in inference twice the patch size on the full-resolution canvas, 64 x 64 px at the finest layers, see section 6) and clipped to it | full canvas | equivalent except for clipping at the window border (strokes extending more than 16 px beyond their 32 px patch at layer resolution are cut in PT); window clipping can be added in the T-015 adapter |
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
- **Pixel loss:** `L1(rec, render)` (mean over pixels) x `lambda_pixel = 10` [C]; paper: `L_pixel = ||I_r - I_t||_1` [P Eq. 7] and the total is `L = L_stroke + L_pixel` [P Eq. 2], i.e. **pixel weight 1**.
- **Hungarian matching** per patch between the 8 predictions and the *valid* ground-truth strokes (`scipy.optimize.linear_sum_assignment`, on CPU, per sample),
  cost = `cdist(pred5, gt5, p=1)` (L1 over the 5 shape parameters) `+ W-distance + (1 - decision_logit)` [C: `cost_matrix_decision = (1 - decision)` with `decision` the raw logit].
  [P Eq. 12: `M_{u,v} = g_v (D_L1 + D_W + D_bce)` with `D_bce` of Eq. 11 (the lambda_r-weighted BCE): deviation.] Invalid GT strokes are excluded from the matching (the same effect as `g_v = 0`).
- **Stroke L1** (`loss_gt`): `L1Loss` (mean over the matched pairs and the 5 parameters) x `lambda_gt = 1`.
- **Wasserstein stroke loss** (`loss_w`): `gaussian_w_distance` between the 2-D Gaussians of the strokes, `|mu_1 - mu_2|^2 + (w1^2 + h1^2)/4 + (w2^2 + h2^2)/4 - 2 Tr((S1^{1/2} S2 S1^{1/2})^{1/2})`
  [C gaussian_w_distance, get_sigma, get_sigma_sqrt; P Eq. 9-10, with `w/2, h/2` as in Eq. 9; the code's covariance entries use `w^2/4` and `h^2/4` for `S` and `w/2, h/2` for `S^{1/2}`],
  mean over matched pairs x `lambda_w = 10`.
- **Confidence loss** (`loss_decision`): `BCEWithLogitsLoss(pos_weight = lambda_recall = 10)` over **all** `B x 8` predictions, positives = the predictions matched to a valid GT stroke,
  x `lambda_decision = 10`. Paper: `D_bce = -lambda_r g_v log sigma(c_u) - (1 - g_v) log(1 - sigma(c_u))` [P Eq. 11], where **`lambda_r` is the recall weight on the positive term** (it equals the code's `pos_weight`), and the stroke loss
  `L_stroke = (1/n) sum_i g_{Y_i} (lambda_L1 D_L1 + lambda_W D_W + lambda_bce D_bce)` [P Eq. 13] weights the three terms by `lambda_L1, lambda_W, lambda_bce` (the code's `lambda_gt`, `lambda_w`, `lambda_decision`).
- **Total:** `loss = 10 * L1_pixel + 1 * L1_stroke + 10 * W + 10 * BCE` [C defaults in `painter_model.py:modify_commandline_options`; `train.sh` does not override them].
  [P §4.1 gives `lambda_r = 8, lambda_L1 = 1, lambda_W = 10, lambda_bce = 1`; with Eq. 2, 11, 13 the paper total is `1 * L_pixel + L_stroke` with recall weight 8, i.e. the mapping paper <-> code of section 9: deviation in the recall weight (8 vs 10), the BCE weight (1 vs 10) and the pixel weight (1 vs 10).]
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
2. Image zero-padded (black) to `32 * 2^K` square. The painting `final_result` lives at the **full padded resolution** (`32 * 2^K`, 512 px for K = 4) from the first layer on. For layers `l = 0..K` the target and the current canvas are resized to `32 * 2^l`
   with `F.interpolate` default mode (**nearest**) *only to feed the network*, cut into non-overlapping 32 x 32 patches (`2^l x 2^l` of them), the model predicts 8 strokes per patch, colours are sampled from the target patch at the predicted centres,
   strokes with `Sign(logit) = 1` are painted onto the full-resolution `final_result`. (An earlier version of this note said that each layer stays at its own resolution and is nearest-upsampled by the next one: wrong, corrected in T-015, section 14.)
3. **Rendering windows:** stroke parameters are re-expressed in windows of twice the patch size (`2 * 512 / 2^l` px on the full-resolution canvas: 1024 px at layer 0, 64 px at layer 4 and in the shifted layer; patch + half a patch of margin on each side); windows are rendered in **four phases** (even-y/even-x, odd/odd, odd/even, even/odd window
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
| recall weight in `D_bce` ([P Eq. 11], also inside the matching cost, Eq. 12) | `lambda_r = 8` | `lambda_recall` = 10 (`pos_weight`) | follow the code; flag `paper` = 8 |
| stroke L1 ([P Eq. 13]) | `lambda_L1 = 1` | `lambda_gt` = 1 | same |
| Wasserstein ([P Eq. 13]) | `lambda_W = 10` | `lambda_w` = 10 | same |
| confidence BCE ([P Eq. 13]) | `lambda_bce = 1` | `lambda_decision` = 10 | follow the code; flag `paper` = 1 |
| pixel loss ([P Eq. 2]: `L = L_stroke + L_pixel`) | weight 1 | `lambda_pixel` = 10 | follow the code; flag `paper` = 1 |
| matching cost ([P Eq. 12]) | `D_L1 + D_W + D_bce` (with the `lambda_r`-weighted BCE) | `L1 + W + (1 - logit)` | see the next row |
| **"paper loss weights" preset** | `pos_weight` 8, BCE weight 1, pixel weight 1, stroke L1 1, W 10, matching with the `lambda_r`-weighted BCE | `pos_weight` 10, 10, 10, 1, 10, matching `1 - logit` | T-012 should expose both presets; the paper preset is an unvalidated alternative (the released weights were trained with the code values) |
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

## 11. Implementation in this repository (T-011)

**Code:** `brushstroke/renderer/pt.py` (PT mode renderer, adapters), `brushstroke/stroke_model/synthetic.py` (data generator), `scripts/fetch_pt_brushes.py` +
`configs/pt/brushes.yaml` (assets, SHA-256, Apache-2.0). Tests: `tests/test_pt_mode.py`, `tests/test_synthetic.py` (they compare against the official code when
`data/raw/pt_official/` and `data/raw/pt_brush/` exist, otherwise skip with the command to run; the CI subset therefore skips them).

**Use.** `render.mode: pt` (config section `render:`; `pt_alpha: pt_infer | pt_train`, `pt_brush_dir`) switches `brushstroke.renderer.render` / `render_progressive` to the PT mode: the
`(B, N, 9)` strokes are read as PT strokes (`ours_to_pt`: first 8 parameters, the ninth `a` is the **0/1 decision**, `a > 0` = painted), the canvas is **black**, the `brush` argument is ignored
and the official brush PNGs are loaded (`pt_infer` -> the `inference/brush` pair, `pt_train` -> the `train/brush` pair). The default mode is untouched (all existing tests pass unchanged).
Direct API: `pt_param2stroke(param, H, W, brushes, mode)`, `render_pt`, `render_pt_progressive`, `pt_composite`, `pt_to_ours`, `ours_to_pt`, `patch_to_canvas` (patch units -> canvas fractions, with
`shifted=True` for the half-patch final layer).

**What the official code does that is easy to miss (found while matching it).**
- `train/util/morphology.py:Dilation2d/Erosion2d.forward` **write into their input in place** (`x[:, [i]] = channel`) and return it; `inference/morphology.py` does not. Equivalent results, but a test that reuses the input tensor sees it modified.
- The train and inference pipelines differ in the order binarize / warp (note section 1.3) and in the brush files; our two modes follow them separately.
- Morphology is implemented with `max_pool2d` (`-max_pool2d(-x)` for the erosion); it equals the official `unfold`-based one exactly (tested, bit-equal).

**Parity with the official renderer (random strokes, both brushes used, `h > w` and `h <= w`).** Maximum absolute difference of the foreground and alpha maps:
| mode | sizes | max abs diff |
|------|-------|--------------|
| `pt_infer` vs `inference.py:param2stroke` | 32², 64², 48 x 80 | **0.0** (bit-identical) |
| `pt_train` vs `PainterModel.param2stroke` + official `Dilation2d/Erosion2d` | 32², 64² | 1.8e-7 (we warp the gray brush and multiply by the colour afterwards instead of colourizing the 394 px image first) |
The acceptance target was <= 1/255 = 3.9e-3. The patch-to-canvas adapter was checked against the official four-phase window renderer on one stroke (centroid within 1 px, IoU of the painted footprints > 0.9): the
textures differ slightly because the official code nearest-resizes the brush to the 64 px window while we resize it to the layer canvas, the geometry is the same (note section 1).

**Synthetic generator** (`SyntheticStrokeGenerator.sample(batch, generator)`), same random draws as `set_input` (`rand(B/4, 8, 12)` then `rand(B, 8, 12)`):
- With the same CPU random stream the batches **equal the official ones**: ground-truth parameters and labels bit-identical, canvases and targets within 2.4e-7 (seeds 0, 1, 2, batch 64).
- Statistics over 12,800 patches (two GPU seeds): valid strokes per patch 6.937 / 6.940 (std 0.911 / 0.910) against the official 6.923 (0.911); histogram 0..8 `[0, 0, 0, 3, 116, 730, 2826, 5291, 3834]` and `[0, 0, 0, 7, 107, 738, 2777, 5336, 3835]` against the official `[0, 0, 1, 4, 107, 761, 2857, 5331, 3739]`.
  Test tolerances: mean within 0.06 (standard error of a difference of two means of 12,800 patches is 0.011, so >5 sigma), std within 0.04, each histogram bin within 5 sigma of the difference of two multinomial proportions + 0.001.
- Determinism: same `torch.Generator` seed -> bit-identical batches (CPU and GPU, tested); different seeds differ.
- Throughput on the RTX 5070 (`python -m scripts.pt_reference_study datagen`): **13.6 k patches/s at batch 64, 43 k at 256, 50 k at 1024**, against **1.07 k patches/s for the official `set_input` on the same GPU** (13x faster);
  at batch 64 that is 4.7 ms per batch against 237 ms for a whole official training step, so data synthesis is not a bottleneck (what dominates the rest of the official step was not profiled here; the per-sample CPU Hungarian matching is the likely suspect).
- Validity rule implemented as in the code (threshold 0.75, only earlier *valid* strokes count; unit-tested on constructed overlaps including the 0.75 boundary).

## 12. Model and losses in this repository (T-012)

**Code:** `brushstroke/stroke_model/model.py` (`PaintTransformer`, `PTModelConfig`, `SignWithSigmoidGrad`, `init_official_weights`), `brushstroke/stroke_model/losses.py`
(`PaintTransformerLoss`, `PTLossConfig`, `gaussian_w_distance`, matchers), configs `configs/pt/{model,loss_official,loss_paper}.yaml`, tests `tests/test_pt_model.py`, `tests/test_pt_losses.py`.

**Model.** Same module and parameter names as the official `Painter(5, 8, 256, 8, 3, 3)`: **9,072,902 trainable parameters** (tested), the released `state_dict` loads with `strict` matching and
reproduces the official outputs. On the weights `model.pth` (16 random input pairs, eval mode) the maximum absolute difference of the shape parameters and of the logits is **0.0** (bit-identical; the test tolerance is 1e-5). The colour is sampled from the target at the predicted centre (official code) by `PaintTransformer.sample_colour`; `colour_mode: predicted` builds an 8-output head
(paper description, ablation: not equivalent to the released weights).

**Losses** (`PaintTransformerLoss(brushes, PTLossConfig)(shape, colour, logits, batch)` -> `LossOutput(total, pixel, gt, w, decision, rec, assign)`), sources: DETR set loss (Carion et al., ECCV 2020), Gaussian
Wasserstein-2 distance between the rectangles' Gaussians (Yang et al., ICML 2021, cited by the paper):
- pixel: L1 between the canvas with the predicted strokes painted (PT train rendering, binary decision with the sigmoid straight-through gradient) and the target;
- Hungarian matching of the 8 predictions to the valid ground-truth strokes with cost L1 + Wasserstein + confidence cost, then parameter L1 and Wasserstein over matched pairs and BCE (with `pos_weight`) over all predictions.
- Presets: `official` (default, = the code: pos_weight 10, BCE 10, pixel 10, L1 1, W 10, matching confidence cost `1 - logit`) and `paper_weights` (pos_weight/lambda_r 8, BCE 1, pixel 1, L1 1, W 10, confidence cost `lambda_r * -log sigmoid(logit)`, which is
  the paper's lambda_r-weighted BCE). `preset: custom` keeps each field (`lambda_*`, `pos_weight`, `matching_cost`) as given. The colour discrepancy is the model flag `colour_mode` (above).

**Equivalence with the official code.** With the same weights and batch (eval mode), the four weighted loss terms equal those of `PainterModel.optimize_parameters` to float precision (seeds 0, 1; batch 32):
pixel 1.221690, gt 0.455586, w 5.370845, decision 42.711430 on both sides. `gaussian_w_distance` clamps the square-root arguments at 1e-20 only to keep gradients finite; with the first version (1e-12) the w term differed by 0.06 % at initialization because the
untrained predictions have sub-pixel widths, which shows the clamp must stay below the squared widths.

**Matching.** `scipy.optimize.linear_sum_assignment` (per sample, as DETR and the official code) is the reference; because it forces a device-to-host copy and a Python loop per sample, an exact batched alternative is provided:
exhaustive search over the 8! = 40,320 assignments on the GPU (`matcher: gpu`, N <= 8 only; ties may select another optimal assignment, same optimal total cost). Tested against brute force (N = 3, 4, 5, negative costs, partial validity) for both matchers.
Time for 64 patches x 8 strokes with ~7 valid on the RTX 5070: scipy 2.37 ms (including the transfer), GPU enumeration 0.59 ms; on CPU scipy is 0.70 ms and enumeration 19.7 ms, so `matcher: auto` uses the GPU enumeration on CUDA only.

**Wasserstein tests.** zero for identical strokes, symmetric, equal to the closed form for commuting covariances, `|dmu|^2 + (w1/2-w2/2)^2 + (h1/2-h2/2)^2` (also with a common rotation), and zero between a rectangle and the same rectangle rotated by a quarter turn
(the distance sees only the Gaussian, so theta has period 1/2 and the sign of `h` is free).

**Observations useful for the training ticket.**
- At initialization the predicted `w, h` are about 0.03 (sub-pixel): the pixel term has **exactly zero gradient** until the other terms grow the strokes (also in the official code); the gradient test therefore checks all four terms after 120 optimization steps (norms for the official preset: pixel 3.97, gt 0.38, w 4.38, decision 62.0).
- Because the Wasserstein term is invariant to the sign of `h` and to a half-turn of `theta`, the network can keep negative heights (observed on an overfitting run: predicted `h` about -0.4 versus ground truth +0.4) at zero Wasserstein cost; the parameter L1 term pulls it back only slowly. The pixel loss does not care either (the PT renderer uses `h > w` to pick the brush).
  Overfitting 4 patches with a 1-layer network: total loss 66.25 -> 1.13 in 700 steps (pixel 0.61, L1 0.14, W 0.30, BCE 0.077); the residual is dominated by the pixel term (strokes cannot reproduce the gradient-textured targets exactly) and this symmetry. A constant learning rate of 1e-3 without clipping was unstable on the full-size network.

## 13. Training pipeline (T-013)

**Code:** `brushstroke/stroke_model/train.py` (`TrainConfig`, `train`, `profile_step`), `scripts/train_pt.py`, `configs/stroke_model/pt.yaml`; tests `tests/test_pt_train.py`.
`python -m scripts.train_pt --config configs/stroke_model/pt.yaml [key=value ...]` (e.g. `schedule.max_steps=500 resume=auto precision=bf16 loss.preset=paper_weights`); `--profile` times the phases of a step.

**Recipe (all in YAML, defaults = the official `train.sh`).** Adam lr 1e-4, betas (0.5, 0.999), batch 64, 220 epochs x 256 steps = **56,320 steps = 3,604,480 samples**, lr x1 until epoch 204 then x0.3 every 5 epochs
(3e-5 for epochs 205-209, 9e-6, 2.7e-6, 8.1e-7 at epoch 220; tested at the epoch boundaries). Official init (`N(0, 0.02)`), data synthesized on the GPU at each step (T-011), no augmentation, no gradient clipping
(`optim.grad_clip` available, off). `schedule.max_steps` truncates a run without changing the lr curve.

**Validation.** `val.batches` (8 x 64 = 512) fixed held-out synthetic patches generated from `val.seed` only (independent of the training seed and stream; tested), evaluated in eval mode at step 0, every `val.every` steps and at the end.
Logged: the four weighted loss terms, their total, and `count_error` = mean |#strokes with logit > 0 - #valid GT strokes| per patch. Files in `output_dir` (never committed): `train.csv` (mean over the last `log_every` steps, lr, step ms, peak VRAM),
`val.csv`, `latest.pt`, `summary.json`, `config.resolved.yaml`; `tensorboard: true` also writes event files (optional package, not installed here).

**Checkpoint and exact resume.** `latest.pt` (atomic write every `checkpoint_every` steps and at the end) holds model, optimizer, scheduler, the step, the config and **all RNG states** (python, numpy, torch CPU/CUDA, and the generator that drives the synthetic data). `resume: auto|path`
restores them; CSV rows after the restored step are dropped. **Test: 6 steps straight == 3 steps + resume + 3 steps, bitwise** (model weights, Adam moments and step, scheduler, validation rows) on a CPU tiny config with dropout active.
On GPU the same holds up to non-deterministic kernels (`deterministic: true` forces deterministic cuDNN); not asserted bitwise.

**Step-time profile** (RTX 5070, batch 64, fp32, ms per step, warm; `train_pt --profile` and `pt_reference_study trainspeed`):
| phase | official code | ours, scipy matcher | ours, GPU matcher (`auto`) | ours, GPU matcher + bf16 |
|---|---|---|---|---|
| data synthesis | 59.7 | 5.0 | 5.4 | 4.9 |
| network forward (+ stroke rendering in the official forward) | 48.8 | 6.2 | 6.2 | 7.1 |
| loss incl. render of predictions | (in forward) | 6.5 | 6.6 | 6.3 |
| - of which Hungarian matching (cost matrix + assignment) | **95.3** | 2.4 | 2.3 | 2.2 |
| backward + loss terms + optimizer | 32.6 | 20.0 + 1.5 | 20.0 + 1.5 | 16.9 + 1.5 |
| **total per step** | **236.5** | 37.7 | 39.8 | 36.5 |
Full schedule: official 3.70 h; **ours about 36-40 min (56,320 x 38-40 ms)**. In the official code the suspected culprit is confirmed: the per-sample Python loop of `optimize_parameters` (`cdist`, Gaussian Wasserstein and a host copy for each of the 64 samples) costs 95 ms, 40 % of the step,
and the data synthesis another 25 %. Our batched cost matrix (one `(B, N, N)` tensor, equal to the official per-sample cost on the valid columns: tested) takes 2.3 ms, so after that change **matching is no longer a bottleneck** and the choice of assignment solver
(scipy per sample 2.4 ms including the transfer, GPU enumeration 2.3 ms within noise at this batch size) is immaterial for speed; `auto` keeps the GPU enumeration on CUDA (no host sync). The step is now dominated by the backward pass
(about 20 ms, many small kernels: 8 sequential compositing steps, a 394 px brush warp per stroke) and is launch-bound, which is why bf16 does not accelerate it. Equivalence of the new matching with the old one: the cost matrices (`test_batched_cost_matrix_equals_the_official_per_sample_costs`),
the optimum of both solvers against brute force (T-012) and the four loss terms against the official ones (T-012).

**bf16 comparison** (3000 steps, same seed, validation at the end, 512 patches): fp32 total 4.92 (pixel 0.673, L1 0.252, W 0.296, BCE 3.70, count error 0.947); bf16 total 4.87 (0.675, 0.252, 0.294, 3.65, 0.930): same curve within run-to-run noise. Speed 38.1 vs 38.2 ms per step;
peak VRAM 851 vs 651 MiB. **Default: fp32** (no speed gain, parity not proven over a full schedule); `precision: bf16` is available (network under autocast, losses and renderer in fp32).

**Short GPU run** (`schedule.max_steps=3000`, fp32): validation total 68.7 at step 0 -> 5.5 (500) -> 4.9 (3000); curve in `docs/media/pt_train_curve.png`. Peak VRAM **851 MiB** (allocated, ours) against 6.2 GB measured on the same GPU for the official code in `trainspeed`
(which keeps an unfold-based morphology and per-sample tensors; the 3.6 GB quoted in section 5 came from the earlier measurement, whose protocol differs). Test: `test_gpu_smoke_run_decreases_the_validation_loss` (300 steps, slow + gpu).
Note that at lr 1e-4 and 3000 steps the model is far from converged (the full schedule is 56,320 steps), and that the BCE term (weight 10, pos_weight 10) dominates the total at that stage.

## 14. Multi-scale inference (T-015)

**Code:** `brushstroke/stroke_model/inference.py` (`paint`, `paint_layer`, `PTInferenceConfig`, `PaintPTConfig`, `load_pt_model`), `brushstroke/stroke_model/pt_sequence.py` (`layers_to_sequence`, `replay_sequence`, `window_geometry`),
`brushstroke/stroke_model/refine.py` (`RefineConfig`, `refine_patch_strokes`); `scripts/paint_pt.py` + `configs/paint_pt.yaml`; tests `tests/test_pt_inference.py`; measurements `scripts/pt_inference_study.py` and `docs/experiments/pt_inference.md`.
`python -m scripts.paint_pt --config configs/paint_pt.yaml --input img --out dir --checkpoint outputs/pt_long/seed2/latest.pt [refine.enabled=true ...]` writes `sequence.npz`, `final.png`, `variants/final_{official,gated}.png`, `progress/layer_XX.png`, `target.png`,
`config.resolved.yaml` and `metrics.json` (like `paint_optim`). `--checkpoint` takes a training checkpoint or the released `model.pth`; both are read with `torch.load(weights_only=True)` (the numpy RNG globals of the training checkpoints are allowlisted, any other object is refused).

**Procedure** (section 6, now reproduced exactly): `K = max(ceil(log2(max(H, W) / 32)), 0)`; black centred padding to `32 * 2^K`; for each layer `l = 0..K` the target and the painting are nearest-resized to `32 * 2^l` for the network, cut into `2^l x 2^l` patches, 8 strokes
per patch, colour sampled at the predicted centre, strokes with `logit > 0` painted by `paint_layer` on the **full-resolution canvas** in four window phases (windows of `2 * S / 2^l` px, even/even, odd/odd, odd/even, even/odd, strokes `0..7` in order, clipped to the window,
the brush nearest-resized to the window side before the warp). Then the shifted final layer (patches offset by half a patch, 17 x 17 for K = 4, canvas and target padded by 16 px), whose strokes the official code paints **whatever their confidence** (`decision.bool()` on the raw logits).
Both variants are always computed (`PTResult.canvas` = `official`, `PTResult.canvas_gated` = only `logit > 0`; the final-layer predictions depend only on the canvas after layer K, so the gated painting is the official one minus the forced strokes);
`inference.final_layer` selects which one is exported to the `StrokeSequence`.

**Corrections to the T-010 note found while matching the official code.**
- The painting is **not** kept at the resolution of each layer: the strokes of every layer are painted on the full-resolution canvas (`final_result` starts as `zeros_like(padded image)`); the layer only sets the size of the windows (1024 px at layer 0 on a 512 px canvas, 64 px from layer 4 on)
  and of the brush image (nearest-resized to the window). Section 6 was corrected.
- For an odd grid (the 17 x 17 shifted layer) the official `param2img_parallel` re-assembles the canvas from window blocks plus margin strips copied from slightly misaligned slices; this only touches the 16 px border that is cropped afterwards (our painter equals the official one on the retained area, checked
  on the intermediate canvases). It is not reproduced, since it never reaches the output.
- The network must follow the official memory layout to be **bit-identical**: with the same `.contiguous()` calls as the official `forward` the attention kernels give exactly the official outputs (before, the outputs differed by one or two ulps on real patches, enough to flip a binary stroke decision and to move the
  final image by up to 0.3). `PaintTransformer.forward` was changed accordingly (the T-012 tests are unchanged).
- Brush choice: the official rule `h > w` compares fractions, which is the same as comparing pixels on a square canvas (all official windows are square). For a non-square image the same stroke would pick the other brush in an `H x W` render; `pt_param2stroke(select_by_pixels=True)` (used by `render_pt` and by the replay)
  compares pixel sizes. The direct function keeps the official rule by default (T-011 parity tests).

**Parity with the official code (the key test).** The official weights loaded in our network, our `paint` against the **float** output of the official `inference.py:main` (the script saves `(x * 255).astype(uint8)`; `scripts.pt_reference_study.official_paint` captures the float tensor):
maximum absolute difference **0.0** (bit-identical, same device) on the five evaluation images at 512 x 512 (target <= 1/255) and on Mona Lisa at 1024 x 1024 (K = 5, 18,835 strokes), with identical stroke counts in every layer. `tests/test_pt_inference.py` repeats it on a 256 x 256
and a non-square 160 x 224 image (padding). Table in `docs/experiments/pt_inference.md`.

**Adapter to `StrokeSequence`** (`layers_to_sequence`). Strokes are ordered by (layer, phase, stroke index, patch raster order inside the phase): the order of the official renderer (and of its `serial` mode). Parameters are in the renderer convention, as fractions of the **unpadded** image
(`x = (S x_c - pad_x) / W`, `w = S w_c / W`, ... where `x_c` is the position in the padded square from `patch_to_canvas`, `a = 1`), `scale_level` = layer (`K + 1` = shifted final layer), `patch_id` = `row * n + col` in the layer grid, `source` = `paint_transformer`, or `paint_transformer_forced`
for final-layer strokes painted although `logit <= 0` (filtering them gives the gated variant exactly). The background is black (`(0, 0, 0)`), render with `RenderConfig(mode="pt")`. Strokes whose bounding disc lies entirely outside the image (possible only with padding or in the shifted layer, whose centres can be up to 16 px outside) are dropped (`meta["dropped_invisible"]`);
`meta` also stores the padded size, the layer count, the patch size, the variant, `abs_h`, the refinement flag, the runtime and the VRAM.

**`|h|`.** The T-014 finding (the losses do not constrain the sign of `h`): with `abs_h` (default) the height is replaced by its absolute value right after the prediction, before the colour sampling, the painting and the export, so that the canvas the network sees at the next layer, the painted canvas and the
sequence are consistent. It is the identity for the official weights and for the T-014 seeds 1 and 2 (the statistic `negative_h_fraction` of `metrics.json` is 0 on all the images of the study). With `abs_h: false` negative heights are kept (and mirror the brush). Tests: a predictor that returns negative heights gives the same canvas and the
same parameters as one that returns their absolute values, and no negative height ever reaches the sequence.

**Re-rendering the sequence.** `StrokeSequence.render` (`RenderConfig(mode="pt")`) draws every stroke at the final resolution with the brush resized to the canvas and no window clipping: the same strokes but a different texture sampling, so it is close to the pipeline canvas (mean absolute difference about 0.004 to 0.011, i.e. 1 to 3 levels of 255,
see `docs/experiments/pt_inference.md`) but not equal. `replay_sequence(seq, brushes)` re-renders a sequence made by `paint` **as the pipeline painted it** (brush resized to the window of the stroke, strokes clipped to that window with the morphology applied inside the window, one extra pixel of canvas around the image so that the morphology of the border pixels sees the stroke
beyond it): it equals the pipeline canvas up to the float rounding of the parameter round trip (mean absolute difference about 1e-6; at most a few pixels of an image differ by more than 1/255, where an alpha threshold flips). It needs the `scale_level`/`patch_id` tags and the meta entries and is meant for checks and faithful animations of PT strokes;
generic consumers use `StrokeSequence.render`.

**Per-patch refinement** (`refine.enabled`, off by default). The strokes predicted for every patch of a layer are optimized for `iters` steps (Adam, lr 0.05, cosine decay to 10 %, on the sigmoid-reparameterized parameters of the M1 baseline, `brushstroke.baselines.optim`) on the pixel loss the network was trained on:
L1 between the target patch and the canvas patch with the 8 strokes painted over it, in the training rendering (`pt_train`: the binarized brush is warped bilinearly, so the stroke edges have a gradient with respect to the geometry; the inference alpha is binary and has none). The decisions are kept, the shape and the colour change; a patch keeps its refined strokes only if its patch loss
decreased (`keep_if_better`). All the patches of a layer are optimized in one batch (they are independent), after the layer's prediction and before painting, so that the next layers see the refined canvas. The proxy is not exactly the inference rendering (32 px patches, strokes clipped at the patch border, soft alpha, brush not resized to the window),
which is why the effect on the final image is measured (`docs/experiments/pt_inference.md`).

**Cost** (RTX 5070, 512 x 512, five images, median of 5 runs after warm-up): about 0.23 s per image and 1.05 GB of allocated memory at peak (the layer-0 window is 1024 x 1024 px) for both models; the refinement adds about 10 s (100 iterations per layer). The official code measured 0.5-0.9 s (loading included) and 2.3 GB in T-010.
