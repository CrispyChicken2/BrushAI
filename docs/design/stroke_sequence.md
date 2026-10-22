# Stroke sequence data model (T-004)

Code: `brushstroke/strokes.py` (`StrokeSequence`, `BrushSpec`). Tests: `tests/test_strokes.py`.

**Why a top-level module** (not `renderer/sequence.py`): the sequence is shared by every producer (optimization
baseline, Paint Transformer, RL agent, contribution modules) and consumer (animation, evaluation); it only *calls*
the renderer, so it should not live inside it.

## What it is
A painting = N ordered strokes (stroke k is painted over strokes 0..k-1) plus the metadata to render and interpret
them. Parameters are normalized (renderer convention, `renderer.md`), so the same sequence renders at any resolution.

## Schema (version 1)
| field | type / shape | meaning |
|-------|--------------|---------|
| `params` | float32 `(N, 9)` | `x, y, w, h, theta, r, g, b, a` (see `renderer.md`); float64 input is converted |
| `canvas_size` | `(H, W)` ints | default render size |
| `brush` | `BrushSpec(kind, bank_size, mask_height, mask_width, seed, index_seed)` | brush bank the strokes were made for (masks are regenerated from the spec); `seed` = masks, `index_seed` = seed of the default variant draw (= `RenderConfig.brush_seed` of the producer) |
| `brush_index` | int64 `(N,)` | bank variant of each stroke; default = painting 0 of the renderer's default draw with `brush.index_seed` (see below) |
| `background` | `(r, g, b)` floats | initial canvas colour |
| `background_ref` | `str | None` | informational reference (name / path / hash) to a background image; not used for rendering |
| `scale_level` | int32 `(N,)` | producer's coarse-to-fine level, `-1` = unset |
| `patch_id` | int32 `(N,)` | patch the stroke was predicted for, `-1` = unset |
| `source` | list of N `str` | producer name per stroke, `""` = unset |
| `meta` | JSON dict | config hash, seed, producer version, timings, ... (normalized by a JSON round trip: tuples become lists and dict keys become strings) |
| `schema_version` | int | `SCHEMA_VERSION = 1` |

Operations: `len`, `seq[:k]` / `seq[[i, j]]` / `seq[bool_mask]` (`select`; keeps all tags and meta), `reorder(perm)`
(validated permutation; params and every tag permuted), `StrokeSequence.concat(parts)` (same canvas/brush/background
required; meta and `background_ref` from the first part), `to(device)`, `render(height=None, width=None, cfg=None,
canvas=None)` -> `(3, H, W)`, `render_progressive(checkpoints, ...)` -> `(len(checkpoints), 3, H, W)`, `from_batch`, `equals(other)`, `save` / `load`, `empty`.

## Producers: keeping render == what was optimized
The brush variant of each stroke is part of the painting. The renderer, when not given `brush_index`, draws it with
`default_brush_index(B, N, K, seed)`: painting `b` comes from its own generator seeded `seed * 1_000_003 + b`
(independent of the batch size B, prefix-stable in N). A producer that renders a batch with the renderer defaults
(`RenderConfig.brush_seed = s`) must wrap painting `b` with

```python
StrokeSequence.from_batch(params, b, BrushSpec(..., index_seed=s), canvas_size=(H, W))
```

which fills `brush_index` with exactly the draw the renderer used (tested bitwise for B = 1, 2, 3 and `s` = 0, 3).
The plain constructor's default (`brush_index=None`) is painting 0 with `index_seed`. A producer that passed explicit
`brush_index` to the renderer just passes the same tensor to the sequence. Since `render()` always uses the stored
`brush_index`, a saved sequence re-renders identically whatever the seed conventions of the renderer later become.

## File format
One `.npz` (`np.savez`, no pickle): arrays `params` (float32), `brush_index` (int64), `scale_level`, `patch_id` (int32)
and `header`, a UTF-8 JSON byte array holding `schema_version`, `canvas_size`, `brush`, `background`, `background_ref`,
`source`, `meta`. Round trip is exact (floats are float32 on both sides; JSON floats are repr-exact). Meta values
must be JSON-serializable (`TypeError` otherwise); NaN/inf are allowed in JSON but discouraged.

## Versioning policy
- `schema_version` is an integer bumped on any change that an old reader could misinterpret (renamed/removed/
  re-meaning of a field, a different param layout). Adding an optional field with a default does not need a bump.
- `load` raises a clear `ValueError` for a file **newer** than the code (never guesses). For an **older** file it
  applies the migrations registered in `_MIGRATIONS[v]` (v -> v+1, edit header/arrays in place) in sequence; a missing
  migration is an error. There are none yet (v1 is the first version).
- Any bump must come with: a migration, a test loading a stored old-format file, and an update of this table.

## Rendering at other resolutions
`render(H, W)` re-renders the normalized parameters. A 2x render box-downsampled to 1x differs from the 1x render by
small sampling/antialiasing differences that shrink with resolution; measured (40 random strokes, seeds 0-4): at
128x192 mean abs error 2.6e-3..3.4e-3, 99th percentile 0.03..0.05, max 0.13..0.28; at 256x384 mean ~8e-4; at 64x96 mean
~8e-3. Test tolerance (128x192): mean < 6e-3, 99th percentile < 8e-2.
