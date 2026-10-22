# Stroke-by-stroke animation (T-006)

Code: `brushstroke/animation/{config,timeline,frames,export}.py`, reveal geometry in
`brushstroke/renderer/rasterize.py` (`reveal_front`, `rasterize_flat(reveal=...)`). Config: `configs/animation.yaml`
(`AnimationConfig`). Script: `python -m scripts.animate --sequence x.npz --out dir [key=value ...]`. Tests:
`tests/test_animation.py`.

## Timeline (`timeline.py`)
`num_frames = round(fps * duration) + round(fps * hold)`: a painting phase of `duration` seconds followed by `hold`
seconds of the finished painting. The painting phase is mapped so that its last frame is at `t = duration` (everything
is finished there, even with `hold = 0`).

Every stroke gets a start time, a reveal duration and a direction (all deterministic):
- **Slots**: stroke k owns a time slot `T * w_k / sum(w)`; `w = 1` (`easing: uniform`) or grows linearly from 1 to
  `decel_ratio` (`easing: decelerate`: fast coarse/background strokes, slower details).
- **Reveal duration** = `overlap x slot` (overlap = average strokes in flight), but at least `min_reveal_frames`
  frames so that no stroke pops in; the maximum is the whole painting phase.
- **Starts** are spread linearly over `[0, T - max(duration)]` in index order, so starts are non-decreasing and every
  stroke has ended by `T`.
- **Progress** inside a stroke is `smoothstep` or linear (`stroke_ease`); progress is monotone in time.
- **Direction** of each stroke: forward (from local `x = -1`), reverse, or random with `seed` (prefix-stable in N).
Pop-in rule: a stroke only appears in fewer than 2 frames if its reveal lasts < 1 frame, which requires
`min_reveal_frames < 1` (tested both ways).

## Partial reveal (`rasterize.py`)
The stroke geometry is the renderer's affine grid (`stroke_affine`; its x channel is the stroke-local coordinate along
the main axis, in [-1, 1] inside the box). With `t = (x + 1) / 2` clamped to [0, 1] (mirrored for reverse strokes):
```
reveal(p) = clamp((p (1 + edge) - t) / edge, 0, 1),     alpha_visible = alpha * reveal(p)
```
`edge` (`reveal_edge`, fraction of the stroke length) is the width of the soft front. `reveal(0) = 0`, `reveal` is
non-decreasing in `p` (tested over a grid of p for random strokes), and it is forced to exactly 1 at `p = 1`, so a finished
stroke is bit-identical to the normally rendered one. Because the front is computed in the same call from the same grid, no
geometry is duplicated, and texture, rotation, canonicalization (`w W >= h H`, so the front runs along the long side) and
the sampler option all carry over. A reveal along the main axis shows the stroke "being laid down" with no brush or cursor.

## Frame rendering (`frames.py`)
Strokes are composited in index order. At frame f, with progress `p_k(f)`:
- the **finished prefix** (all strokes before the first unfinished one) lives in a running canvas, advanced once when
  new strokes finish;
- only the **window** `[prefix, started)` (strokes that have started and are not in the prefix) is rasterized with its
  current progress and composited over it with the closed-form compositing of T-003, in chunks of `chunk_size`.
This is exactly the brute-force composite of all strokes at their progress (tested for every frame of a sequence, atol 1e-5)
at a cost proportional to the number of strokes in flight. A subtle point the test caught: the prefix must be computed
with `p >= 1`, not `p != 0`. The canvas texture of T-002 modulates every frame (`apply_canvas`, `canvas_strength`), so the
last frame equals `seq.render()` + canvas modulation to within 1/255 (uint8 rounding; tested). Hold frames repeat the last
frame. `FrameRenderer.frames()` can be iterated repeatedly (state is reset); `render_frame(f)` expects increasing `f`.
Runs on the GPU when available (`device`), strokes' parameters/masks stay on the device.

## Export (`export.py`)
ffmpeg is called as a subprocess with raw `rgb24` frames on stdin (`find_ffmpeg` raises a clear `RuntimeError` with
install hints if it is missing).
- **MP4**: `libx264 -preset medium -crf mp4_crf -pix_fmt yuv420p -movflags +faststart`, dimensions made even by a `scale`
  filter (a 65 x 63 canvas yields 64 x 62).
- **WebM**: `libvpx-vp9 -crf webm_crf -b:v 0 -row-mt 1 -pix_fmt yuv420p`.
- **GIF**: two passes (`palettegen=stats_mode=diff`, then `paletteuse` with `dither` in {none, bayer, sierra2_4a,
  floyd_steinberg}, Lanczos down-scaling to `gif_width` (never upscaled) and `gif_fps`); the frames are spooled to a temp
  raw file so both passes can read them.
MP4/WebM are encoded in streaming mode from one frame iterator, so several formats cost one render. Tests check codec, pixel
format, fps, packet count and duration with ffprobe (skipped when ffmpeg/ffprobe are absent).

## Performance (RTX 5070, ffmpeg 9.0.1, 512 x 512)
| case | frames | render + encode | peak VRAM |
|------|--------|-----------------|-----------|
| first 1000 strokes of the T-005 painting, 10 s @ 30 fps + 1.5 s hold, MP4 + WebM, GPU | 345 | 2.5 s | 89 MB |
| same, MP4 only, CPU (`device=cpu`) | 345 | 4.5 s | - |
| full 1794 strokes, 8 s @ 30 fps, MP4 + GIF (400 px, 15 fps) | 285 | 2.5 s | - |
Output sizes for these runs: MP4 0.5 MB, WebM 0.2 MB, GIF (400 px) 0.65 MB.

## Limitations
- A single global timeline (uniform or linearly decelerating slots); content-aware timing (e.g. slower on the face) is
  left to the contribution milestone.
- A long-lived early stroke keeps all later strokes in the window (they are re-rasterized each frame until it finishes);
  harmless with the default settings (the window is a few strokes).
- No paint-thickness or lighting; no brush or hand is drawn by design.
