# docs/design

Short design notes for non-trivial components: what, why, references, conventions, measured cost.

| Note | Component |
|------|-----------|
| [brushes.md](brushes.md) | Procedural brush masks (`flat`, `oil`, `round`), `BrushBank`, canvas texture (T-002) |
| [renderer.md](renderer.md) | Differentiable batched stroke renderer: parameters, rasterization, compositing, memory, benchmark (T-003); Paint Transformer compatible mode `render.mode: pt` (T-011) |
| [stroke_sequence.md](stroke_sequence.md) | `StrokeSequence` data model, `.npz` format, versioning policy (T-004) |
| [optim_baseline.md](optim_baseline.md) | Baseline (a): progressive optimization-only decomposition, Sinkhorn OT, differences from SNP, results (T-005) |
| [animation.md](animation.md) | Stroke-by-stroke animation: timeline, partial reveal, frame renderer, ffmpeg export (T-006) |
| [evaluation.md](evaluation.md) | Evaluation protocol of M3 (T-022), decided before any M3 run: eval set and dev/test split, metrics, anytime AUC, saliency-weighted LPIPS and the circularity rule, strokes-to-target, statistics, baseline interface, Learning to Paint data policy |
| [paint_transformer.md](paint_transformer.md) | Paint Transformer study note (T-010): renderer, data, model, losses, training, inference, reproduction targets, licences, discrepancies; section 11: the PT mode and generator (T-011); section 12: the model and Hungarian-matched losses (T-012); section 13: the training pipeline, step profile and bf16 comparison (T-013); section 14: multi-scale inference, the sequence adapter and per-patch refinement (T-015) |
| [pt_bn_validation.md](pt_bn_validation.md) | BatchNorm statistics of the PT validation (T-020): `val.bn` modes (running, recalibrate, batch), training untouched, measured default |
| [pt_convention.md](pt_convention.md) | Paint Transformer `(w, h, theta)` convention: exact and approximate equivalence classes of the renderer, the invariant `D_L1`, the three convention options (T-019) |
