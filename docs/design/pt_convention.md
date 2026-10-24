# Paint Transformer: the `(w, h, theta)` convention and its equivalence classes (T-019)

**What.** The Paint Transformer network predicts `x, y, w, h, theta` of each stroke as unconstrained numbers. Several triples
`(w, h, theta)` paint the same (or nearly the same) stroke, so two networks can learn different conventions at the same
quality (T-014: seed 0 predicts `h < 0` for every stroke; T-016: `w < 0` and shifted `theta` in some ablation runs) and
only the parameter-L1 term, which compares raw numbers, notices. This note derives which triples are equivalent for the
renderer, how well, and what T-019 builds on it (`brushstroke/stroke_model/convention.py`, option flags, invariant `D_L1`).
Experiment and verdicts: `docs/experiments/pt_training.md`, section T-019.

## 1. The warp
`pt_warp_grid` (`brushstroke/renderer/pt.py`, the official `param2stroke`) samples the brush image `B` at
`s = M (p - c)`, `p` the output pixel in normalized coordinates, `c = (2x - 1, 2y - 1)` the stroke centre and, on a square
canvas,

    M = diag(1/w, 1/h) R(-pi theta),    R(-phi) = [[cos phi, sin phi], [-sin phi, cos phi]].

The alpha of the stroke is `B(s) > 0` (training renderer: the brush is binarized first), the foreground is `B(s)` times the
colour. The brush image is chosen by `h > w` (vertical image) or not (horizontal image); for the *training* images the two
masks are identical (bit for bit) and only the grey texture differs (0.023 mean absolute difference); for the *inference* images
the masks differ slightly.

## 2. Exact symmetries
* `theta` enters only through `cos(pi theta)`, `sin(pi theta)`: period **2**.
* `theta -> theta + 1` negates `R`, hence `M -> -M`. `(w, h) -> (-w, -h)` negates `diag(1/w, 1/h)`, hence also `M -> -M`.
  So **`(w, h, theta + 1)` and `(-w, -h, theta)` give the same sampling grid, for every pixel and for every canvas size**
  (the aspect factors `H/W` in `M` multiply the rotation, not the diagonal). The selection rule flips with the signs
  (`-h > -w` is `h < w`), so the other brush image is used: identical alpha for the training images, a few edge pixels
  for the inference images, and the foreground changes by the texture difference only.
* These two facts generate the **exact class**: `{(w, h, theta), (-w, -h, theta + 1)}` modulo `theta -> theta + 2`.
  Nothing else is exact: a stroke with `w, h > 0` has no other exact representative (its rotation by pi is *not* the
  same stroke because the brush mask is not point-symmetric: 26 % of its pixels differ).

## 3. Approximate symmetries (measured)
The brush mask is nearly mirror-symmetric about its horizontal axis (the pixels that change under `v -> -v`: 12.6 % of its
area; under `u -> -u`: 28 %; under `(u, v) -> (-u, -v)`: 26 %; for the rotation by pi/2 with `w <-> h`: 28 %). Therefore:

* **mirror class** (adds `(w, -h, theta)` and `(-w, h, theta + 1)`): the stroke changes on **14 %** of its area. This is
  what the `h < 0` networks use: the stroke they paint is the mirror image of the one the same parameters would paint
  with `h > 0`, and the rest of the network (position, `w`, `theta`, the colour) compensates.
* **rot180 class** (all sign pairs with `theta`, `theta + 1`) and **swap class** (adds `(+-h, +-w, theta +- 1/2)`): 28 to
  34 % of the area. Two *unrelated* strokes of the training distribution differ by 157 % of the area (they barely
  overlap), so these variants are "near" strokes, not equivalent ones: reported as diagnostics only.

Measured by `python -m scripts.exp_pt_convention symmetry` (4000 random strokes of the training distribution, 64 x 64
alpha maps; table in `docs/experiments/pt_training.md`): exact class 0.00000 (training renderer) / 0.00005 (inference
renderer); mirror class 0.0131 absolute (0.144 of the mean alpha 0.0908); rot180 and swap variants 0.026 to 0.031 (0.28 to
0.34); unrelated strokes 0.142. The unit tests assert the same ordering (`tests/test_pt_convention.py`).

## 4. Why the network can choose any convention
The Wasserstein term depends on the stroke only through its Gaussian `(mu, Sigma)`, `Sigma = R diag(w^2, h^2)/4 R^T`, which
is invariant under every `(+-w, +-h, theta + k)` and under `w <-> h` with `theta + 1/2`: the whole **swap class** is
invisible to `D_W`. The pixel term sees the renderer, which has the exact symmetry only, and the confidence term sees
neither. The parameter-L1 term is the only one that compares raw numbers, so it is the only one that prefers the
ground-truth convention (`w, h > 0`, `theta in [0, 1)`), and it is weak (weight 1, averaged over the five parameters).
The choice is made in the first hundreds of steps (T-014, finding 2) as a function of the seed.

## 5. What T-019 builds
* `convention.variant_specs / variants`: the variants of each class level (`none`, `exact`, `mirror`, `rot180`, `swap`;
  each closed under composition).
* `convention.stroke_l1_sum(pred, gt, level)`: the parameter L1 minimized over the class of the prediction, with the
  `theta` difference taken modulo its exact period 2 (the official distance for `level="none"`).
* **Invariant `D_L1`**: `eval.pt_strokes.invariant_l1` / the `d_l1_<level>` keys of the stroke metrics. The primary one is
  `mirror` (the smallest class that contains every observed `h < 0` outlier); `exact` is the strict one; `rot180` and
  `swap` are diagnostics that also absorb the shifted-`theta` runs.
* Options (flags; defaults reproduce the official behaviour bit for bit):
  **A** `model.convention: positive` (`w = softplus(raw)`, `h = softplus(raw)`, `theta = raw - floor(raw)`: the
  predictions live in the ground-truth convention by construction); **B** `loss.l1_invariance: mirror` (parameter L1 in the
  loss *and* in the matching cost is the class minimum: the convention is no longer penalized, and not removed);
  **C** `loss.lambda_sign: 10` (`+ lambda * mean(relu(-w) + relu(-h))`).

## References
Liu, Y. et al., *Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*, ICCV 2021 (parameters, losses);
official code `Huage001/PaintTransformer` commit 7960b8e (`inference/inference.py:param2stroke`, `train/models/painter_model.py`);
Yang, X. et al., *Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss*, ICML 2021 (the Gaussian
distance, whose symmetry group is the swap class); the symmetry analysis itself is ours.
