# Differentiable batched stroke renderer (T-003)

Code: `brushstroke/renderer/{params,rasterize,compositing,pipeline}.py`. Config section `render:`
(`RenderConfig`). Benchmark: `python -m scripts.bench_renderer`. Tests: `tests/test_renderer.py`.

## Parameter convention
Strokes are a float tensor `(B, N, 9)`: B paintings, N strokes in painting order (stroke k is composited over
strokes 0..k-1). Last dimension:

| idx | name | range | meaning |
|-----|------|-------|---------|
| 0, 1 | `x`, `y` | [0, 1] | stroke centre as a fraction of canvas width / height (origin top-left, y down) |
| 2, 3 | `w`, `h` | (0, 1] | box size along the stroke's local x / y axis, as a fraction of canvas **width / height** |
| 4 | `theta` | [0, 1] | rotation, mapped to [0, pi) radians; periodic (0 and 1 identical) |
| 5-7 | `r, g, b` | [0, 1] | colour |
| 8 | `a` | [0, 1] | opacity (extra vs. Paint Transformer; PT strokes have a = 1) |

This is Paint Transformer's `param2stroke` layout (`x0, y0, w, h, theta, R, G, B`) plus opacity.
Helpers: `clamp_params` (hard clamp, `w, h >= min_size`), `params_from_unconstrained` (sigmoid, `w, h` rescaled to
`(min_size, 1)`) and its inverse `unconstrained_from_params` (logit), for optimization (T-005).

## Rasterization
For each stroke, the brush mask `(1, h_m, w_m)` (main axis = local x, see `brushes.md`) is warped onto the canvas
with `F.affine_grid` + `F.grid_sample` (bilinear, zero padding, `align_corners=False`). With normalized output
coordinates `X, Y` in [-1, 1], centre `(2x - 1, 2y - 1)` and `t = pi * theta`:

```
dx = (X - cx) W / 2,  dy = (Y - cy) H / 2                       # pixel offsets
u =  cos t dx + sin t dy,  v = -sin t dx + cos t dy             # stroke-local pixels
brush_x = u / (w W / 2),   brush_y = v / (h H / 2)              # in [-1, 1] inside the box
```
These are exactly the entries of the PT warp matrix (`warp_00 = cos t / w`, `warp_01 = sin t H / (W w)`,
`warp_02 = (1 - 2 x0) cos t / w + (1 - 2 y0) sin t H / (W w)`, ...), so a PT stroke renders at the same place.
Rotation is isotropic in pixel space on non-square canvases. The stroke alpha is `coverage * a`.

**Canonicalization (texture orientation).** In both references the texture direction follows the longer side of the
box (PT: `brush_large_vertical` if `h > w` else horizontal; SNP likewise plus large/small by area). Our masks have
bristles along local x, so before warping `canonicalize` makes every stroke at least as long as thick *in pixels*
(`w W >= h H`): otherwise it swaps the pixel sizes (`w' = h H / W`, `h' = w W / H`) and adds 0.5 to `theta`. The
footprint is identical, the texture runs along the long side (tests: footprint equality incl. a 512 x 128 canvas;
oil bristle direction for tall vs wide strokes). Compared with PT: on square canvases it is the same `h > w`
rule; PT swaps *fractions* but only ever renders square patches, where the two coincide. Deviation from PT to
note for M2: PT binarizes (`alpha > 0`) and erodes the brush alpha and uses two real brush images; we use soft
procedural masks (a `pt_brush` option can be added for the reproduction).

The brush variant of each stroke is `brush_index` (B, N) or, by default, a seeded random draw from the bank
(`brush_seed`). **One rule** (`default_brush_index`, shared with `StrokeSequence`): painting `b` is drawn from
its own generator seeded `seed * 1_000_003 + b`, so row `b` depends on `(seed, b)` only (not on the batch size B)
and is prefix-stable in N: rendering the first k strokes, or painting `b` alone, uses the same brushes.

Samplers (`render.sampler`): `grid_sample` (default, fast) and `gather` (a floor/gather bilinear sampler, identical
values, 1e-5) whose backward has no atomic adds. `grid_sample`'s CUDA backward raises under
`torch.use_deterministic_algorithms(True)`; in practice it was bitwise reproducible across runs in our tests
(brush masks do not require grad, so no atomic accumulation into the input is needed), but that is not guaranteed.
For a guaranteed-deterministic run use `sampler: gather` together with `seed_everything(..., deterministic=True)`.

## Compositing
Over operator in stroke order: `C_k = C_{k-1} (1 - a_k) + c_k a_k` (`C_{-1}` = initial canvas, default white; a colour,
an image or a batch can be given). `composite_sequential` is the naive reference loop. `composite_parallel` uses the
closed form with one reverse cumulative product per chunk:

```
S_k = prod_{j >= k} (1 - a_j),   T_k = S_{k+1} (T_{n-1} = 1),   C_n = C_{-1} S_0 + sum_k c_k a_k T_k
```
Chunks of `chunk_size` strokes are chained (`C_{-1}` of the next chunk = the result of the previous one), so only
`chunk_size` alpha maps exist at a time. `torch.cumprod` handles exact zeros (a = 1) in its backward; this is tested
against the sequential gradient. With `use_checkpoint`, each chunk is recomputed in the backward pass
(`torch.utils.checkpoint`, non-reentrant), so the graph keeps one `(B, 3, H, W)` canvas per chunk boundary.

`render_progressive(params, brush, H, W, checkpoints)` renders once and records the running canvas at each requested
stroke count (equal to rendering the prefixes from scratch; tested). `RenderConfig.dtype = bfloat16` composites in
bf16 (alphas are still sampled in fp32).

## Memory analysis
Without checkpointing, forward+backward stores per stroke pixel: sampling grid (2 floats), coverage, alpha,
`1 - alpha`, the flipped cumprod and its flipped copy, `behind`, the colour-weighted product, ... measured at about
**44 bytes per stroke pixel** (`0.76 GB` for 256 strokes at 256^2, i.e. `256 * 65536 * 44`). Hence 1024 strokes at
512^2 needs ~11 GB without checkpointing, and N*H*W grows linearly in N. With `use_checkpoint=True` the peak is one chunk
(forward in the backward pass) plus one canvas per chunk boundary: ~1-4 GB for the largest cases below.
Forward-only (`torch.no_grad`) memory is one chunk: 0.1-0.65 GB.
The acceptance case (512^2, 1024 strokes, forward+backward) therefore uses `use_checkpoint: true`: 0.65 GB peak
(see table). Without it, the same case needs ~11 GB, which fits but is not "well under 12 GB".

## Benchmark
RTX 5070 (12 GB), torch 2.11.0+cu128, fp32, 3 repeats after a warm-up, bank of 8 `oil` brushes (64 x 128),
random strokes of size 0.03-0.23. Each cell: time / peak allocated VRAM. "chunk / checkpoint" = `chunk_size` and
`use_checkpoint`.

| N | resolution | chunk / checkpoint | forward (time / peak VRAM) | forward+backward |
|---|-----------|--------------------|----------------------------|------------------|
| 256 | 256^2 | 32 / no | 12 ms / 0.09 GB | 42 ms / 0.76 GB |
| 1024 | 256^2 | 32 / no | 40 ms / 0.10 GB | 159 ms / 2.87 GB |
| 4096 | 256^2 | 32 / yes | 159 ms / 0.10 GB | 877 ms / 0.25 GB |
| 256 | 512^2 | 32 / no | 43 ms / 0.33 GB | 149 ms / 2.99 GB |
| 1024 | 512^2 | 32 / yes | 168 ms / 0.33 GB | 763 ms / 0.65 GB |
| 4096 | 512^2 | 32 / yes | 667 ms / 0.33 GB | 3083 ms / 0.93 GB |
| 256 | 1024^2 | 16 / yes | 175 ms / 0.65 GB | 996 ms / 1.29 GB |
| 1024 | 1024^2 | 16 / yes | 692 ms / 0.65 GB | 3911 ms / 1.85 GB |
| 4096 | 1024^2 | 16 / yes | 2683 ms / 0.65 GB | 15490 ms / 4.10 GB |

bf16 compositing, N=512, 512^2: max |err| = 0.0157, mean |err| = 0.00176; fwd 76 ms vs fp32 86 ms

bf16 vs fp32: compositing in bf16 changes pixels by at most 0.016 (mean 0.0018) and is only ~12% faster (the cost
is dominated by sampling, done in fp32), so fp32 stays the default.

Without checkpointing, 1024 strokes at 512^2 forward+backward measured 11.3 GB peak (first benchmark run, before
the table above; timings in that regime were dominated by driver paging and are not reported). With checkpointing it
is 0.65 GB and 0.76 s.

## Bounding-box path (T-027)
`render.bbox: true` rasterizes and composites each chunk only inside the **union pixel box** of its strokes, then pastes the
result back into the canvas (`rasterize.stroke_boxes`, `rasterize_window`, `pipeline._composite_chunk_bbox`).
- **Box.** Per stroke, the axis-aligned extent of the rotated (canonicalized) rectangle, grown by the half-texel of bilinear
  sampling (`1 + 1/w_mask`, `1 + 1/h_mask` in brush coordinates) and one pixel of margin, clipped to the canvas. Tested: no pixel
  outside the box has non-zero alpha, for every rotation, strokes crossing borders and strokes wholly outside (empty box, skipped).
- **Same semantics.** Outside the box every alpha is exactly 0, so the canvas is unchanged there (`C (1 - 0) + c * 0`). Inside, the
  window grid is the slice of the full pixel grid (same `stroke_affine`, same sampler, `grid_sample` or `gather`); the composite is
  the same closed form on the crop. Gradients flow through the crop and the paste (`use_checkpoint` works unchanged).
- **Tolerance.** The window grid comes from a matrix product instead of `F.affine_grid`, so sampling coordinates differ by float32
  rounding: image deviation <= 2e-4 (measured 1.8e-7 to 8e-5), gradient deviation <= 2e-3 of the largest entry (measured <= 4e-6
  on the real sequence). Not bitwise equal; each path is deterministic (same bits on rerun).
- **Chunking.** `bbox_chunk` (0 = `chunk_size`) sets the chunk the box is a union over; a chunk whose box covers more than
  `bbox_max_fraction` (0.75) of the canvas takes the full path, so large strokes cost nothing extra. Smaller chunks give tighter
  boxes but more kernel launches and canvas pastes: measured slower below 64 (table), so 64 = `chunk_size` is the setting.
- **Default: off** (`bbox: false`), so every existing sequence, hash and metric is unchanged; the optimization baseline's config
  hash ignores the three bbox fields while they hold their default (test: equals the hash `main` computed before T-027).

Benchmark (RTX 5070, idle, real 2037-stroke sequence of the M1 landscape demo, 512^2, forward+backward of the first N strokes,
checkpointing as the baseline chooses; `python -m scripts.bench_bbox`, full table `docs/experiments/renderer_bbox_bench.md`):

| strokes | full | bbox (chunk 64) | speed-up | peak VRAM full / bbox |
|---|---|---|---|---|
| 64 | 39 ms | 39 ms | 1.0x | 1.15 / 1.16 GB |
| 256 | 152 ms | 138 ms | 1.1x | 3.18 / 2.70 GB |
| 1024 | 771 ms | 378 ms | 2.0x | 1.15 / 1.11 GB |
| 2037 | 1514 ms | 641 ms | 2.4x | 1.19 / 1.16 GB |

The coarse strokes of the first levels are large and gain nothing; the fine strokes dominate the cost of the baseline.
End to end (`python -m scripts.bench_bbox_e2e`, M1 landscape demo, seed 0, 512^2, 2037 strokes, `cudnn_deterministic`):
**589 s -> 302 s (1.95x)**, peak allocated VRAM 3.38 -> 2.70 GB. Flag off reproduces the committed sequence hash `ee4dd8ac823f`. With
the flag on the optimization trajectory diverges after the first float-rounding differences (the optimizer is chaotic), so the
sequence is different, with equivalent quality: L1 0.0712 -> 0.0708, PSNR 20.31 -> 20.35, LPIPS-VGG 0.358 -> 0.347, LPIPS-Alex
0.432 -> 0.417 (one image, one seed: a tolerance check, not a claim of improvement; `docs/experiments/data/renderer_bbox/`).

## Limitations
- Per-stroke memory is O(H W) regardless of stroke size on the default path (every stroke is rasterized over the whole
  canvas); the bounding-box path below removes this for chunks of small strokes.
- Timings above are measured on a Windows (WDDM) GPU: when the allocator approaches 12 GB the driver may page, which
  makes close-to-limit cases much slower; checkpointed cases stay well below.
- The brush index is not differentiable (nor the canonicalization switch at `w W = h H`, where texture direction jumps).

## References
- Liu, Lu, Yang, Zhao, Gao. *Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*. ICCV 2021
  (`inference/inference.py::param2stroke`: parameter layout, affine warp, `h > w` brush selection).
- Zou, Shi, Guo, Ye. *Stylized Neural Painting*. CVPR 2021 (`renderer.py`: stroke parameters, large/small and
  vertical/horizontal brush choice).
- Porter, Duff. *Compositing Digital Images*. SIGGRAPH 1984 (the over operator).
- Chen et al., *Training Deep Nets with Sublinear Memory Cost* (2016) / `torch.utils.checkpoint` (activation checkpointing).

## Paint Transformer compatible mode (T-011)
`render.mode: pt` renders the strokes exactly like the official Paint Transformer code (two brush images chosen by `h > w`, binarized then eroded alpha, dilated `gray * colour` foreground, black canvas, opacity replaced by a
0/1 decision). It is bit-identical to the official `param2stroke` in the inference variant (max abs diff 0.0) and within 1.8e-7 in the training variant. The default mode described above is unchanged.
See [paint_transformer.md](paint_transformer.md) section 11 and `brushstroke/renderer/pt.py`.
