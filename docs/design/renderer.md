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

## Limitations
- Per-stroke memory is O(H W) regardless of stroke size (every stroke is rasterized over the whole canvas); a
  bounding-box crop would speed up small strokes a lot (possible optimization ticket).
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
