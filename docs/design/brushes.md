# Procedural brush masks and canvas texture (T-002)

Code: `brushstroke/renderer/brushes.py`, `brushstroke/renderer/canvas.py`. Config sections `brush:` and
`canvas:` in `configs/default.yaml` (typed in `brushstroke/config.py`). Preview: `python -m scripts.preview_brushes
output_dir=outputs/brush_preview` writes `outputs/brush_preview/brush_contact_sheet.png` (and `canvas_shaded.png`); without
the `output_dir` override it writes to `outputs/default/` (the default `output_dir`). Outputs are not committed.

## Why procedural
Brush images from other projects are under various licenses (or none). Generating the masks from a seed
keeps the repo license-clean (MIT, no third-party assets), makes them deterministic and lets the ablations
change the brush with one config flag (`brush.kind`).

## Convention
A mask is `(1, h, w)` float32 in [0, 1] (1 = opaque paint). **The main stroke axis is the local x axis
(`w`)**, y (`h`) is the thickness. The renderer (T-003) will warp the mask with an affine transform
(scale to stroke width/height, rotation, translation), the same idea as the reference implementations (see
below). Default mask size 64 x 128 (`brush.mask_height` x `brush.mask_width`).

### Compatibility with the reference implementations (verified in their code)
- Paint Transformer (`github.com/Huage001/PaintTransformer`, `inference/inference.py`): `main()` loads
  `brush/brush_large_vertical.png` and `brush/brush_large_horizontal.png`; in `param2stroke`, `index[h > w] = 0`
  selects the **vertical** image when the stroke is taller than wide, the horizontal one otherwise. The alpha is
  binarized (`alphas = (brush > 0).float()`) and eroded.
- Stylized Neural Painting (`github.com/jiupinjia/stylized-neural-painting`, `renderer.py`, `_draw_oilpaintbrush`):
  four images, large vs small by `w * h / CANVAS_WIDTH**2 > 0.1`, then `h > w` -> vertical, else horizontal.

In both, the bristle direction follows the **longer side** of the stroke box. We have a single mask with bristles
along local x, so **the renderer (T-003) must canonicalize every stroke to `w >= h`**: when `h > w`, swap `w` and
`h` and add 90 degrees to the rotation. This is equivalent to the references' vertical/horizontal selection (up to
the large/small distinction of SNP). Without it a tall stroke would get bristles across its long side.

## Kinds
| kind | models | construction |
|------|--------|--------------|
| `flat` | ablation baseline: a flat rectangle | rectangle inset by 1.5 px, 1 px antialiased border, empty ring outside; seed-independent |
| `oil` | loaded bristle brush | cover = end taper x lateral soft edge; alpha = cover * (0.9 + contrast * streaks) |
| `round` | soft round dab | elliptical smoothstep falloff x mild isotropic noise (0.97 + 0.08 n) |

`oil` details: *streaks* = white noise blurred anisotropically (sigma_y ~ 0.7 px, sigma_x = w/6), normalized to
unit std, so bristle marks run along x. *Ragged ends*: one random value per row (bristle length) perturbs the
taper position (`|u| * (1 + 0.12 row)`, smoothstep 0.55 -> 1). *Ragged sides*: one value per column perturbs the
lateral edge (`|v| + 0.07 col`, smoothstep 0.65 -> 1). *Dry brush*: streak contrast grows from 0.3 to 0.9 where
cover drops, so ends/sides break into bristle gaps while the center stays mostly solid.

All noise comes from a CPU `torch.Generator` seeded per mask, so a mask is bitwise reproducible for a given
`(kind, size, seed)`.

## BrushBank
`BrushBank.build(kind, K, size, seed)` stacks K variants (variant i uses seed `seed + i`) into `masks:
(K, 1, h, w)`; `.to(device, dtype)` returns a moved copy; `get_brush_bank` memoizes the build (lru_cache) and
`BrushBank.from_config` returns a clone so callers cannot corrupt the cache. Cost (CPU, measured): all three
kinds x 8 variants at 64 x 128 = ~45 ms (< 1 s required).

## Canvas
`make_canvas_texture(H, W, seed, strength, weave_period, low_freq_amount, low_freq_scale)` returns `(1, H, W)` in
[0, 1], neutral 0.5. It is a plain weave: cells of `weave_period` px alternate warp/weft (a sine bump across the
thread width), the coordinates get a smooth ~0.35 px jitter (irregular threads) and low-frequency Gaussian noise
adds uneven priming. The field is normalized to [-1, 1] and mapped to `0.5 + 0.5 * strength * field`
(`strength` = contrast of the texture itself). `apply_canvas(img, tex, strength)` multiplies the image by
`1 + strength * (2 tex - 1)` (default config `canvas.strength = 0.06`), which is neutral where the texture is
0.5. Spectrum (measured, 192 x 192, period 6, no low-frequency noise): the dominant peaks are at (+-1, 0)/p and
(0, +-1)/p (about half of the power (47-56% depending on the normalization) in the radial band 0.9-1.1/p; ~1% in 0.4-0.6/p, so there is no 1/(2p) peak); the next
peaks, from the products of the 2p checkerboard with the thread profile, are at (+-0.5, +-1.5)/p and
(+-1.5, +-0.5)/p. The FFT test checks the 1/p band.
512 x 512 takes ~18 ms.

## Limitations
No paint thickness / lighting model (no impasto relief), brush kinds are hand-designed rather than fitted to real
brush scans, and watercolor/marker-pen styles of Stylized Neural Painting are not modeled.

## References
- Zou, Shi, Guo, Ye. *Stylized Neural Painting*. CVPR 2021 (textured brush types: oil-paint, marker-pen, watercolor).
- Liu, Lu, Yang, Zhao, Gao. *Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*. ICCV 2021
  (vertical + horizontal brush images, selected by `h > w`, affine-warped; binarized and eroded alpha).
- Huang, Heng, Zhou. *Learning to Paint with Model-based Deep Reinforcement Learning*. ICCV 2019 (quadratic
  Bezier strokes, untextured; for contrast).

Note: the orientation facts above come from the reference code (files and functions cited in the compatibility
section). Paint Transformer's binarized/eroded alpha is not reproduced by our soft masks; it matters for the M2
reproduction (see the hand-off notes for T-003).
