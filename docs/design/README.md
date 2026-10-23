# docs/design

Short design notes for non-trivial components: what, why, references, conventions, measured cost.

| Note | Component |
|------|-----------|
| [brushes.md](brushes.md) | Procedural brush masks (`flat`, `oil`, `round`), `BrushBank`, canvas texture (T-002) |
| [renderer.md](renderer.md) | Differentiable batched stroke renderer: parameters, rasterization, compositing, memory, benchmark (T-003); Paint Transformer compatible mode `render.mode: pt` (T-011) |
| [stroke_sequence.md](stroke_sequence.md) | `StrokeSequence` data model, `.npz` format, versioning policy (T-004) |
| [optim_baseline.md](optim_baseline.md) | Baseline (a): progressive optimization-only decomposition, Sinkhorn OT, differences from SNP, results (T-005) |
| [animation.md](animation.md) | Stroke-by-stroke animation: timeline, partial reveal, frame renderer, ffmpeg export (T-006) |
| [paint_transformer.md](paint_transformer.md) | Paint Transformer study note (T-010): renderer, data, model, losses, training, inference, reproduction targets, licences, discrepancies; section 11: the PT mode and generator (T-011) |
