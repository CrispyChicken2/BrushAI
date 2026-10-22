# Documentation index

| Document | Content |
|----------|---------|
| [plan/M1.md](plan/M1.md) | Milestone M1 (renderer, optimization-only decomposition, animation export): tickets, VRAM estimate, success criteria |
| [design/optim_baseline.md](design/optim_baseline.md) | Baseline (a), optimization-only progressive decomposition (T-005): method, Sinkhorn OT, differences from SNP, results and tuning table, determinism |
| [design/animation.md](design/animation.md) | Stroke-by-stroke animation (T-006): timeline, partial reveal along the stroke axis, incremental frame renderer, ffmpeg export, performance |
| [design/stroke_sequence.md](design/stroke_sequence.md) | `StrokeSequence` data model (T-004): schema, producer convention (`from_batch`), `.npz` format, versioning policy, resolution behaviour |
| [design/renderer.md](design/renderer.md) | Differentiable batched stroke renderer (T-003): parameter convention, rasterization, compositing, memory analysis, benchmark table |
| [design/brushes.md](design/brushes.md) | Procedural brush masks and canvas texture (T-002): what each kind models, orientation convention, references |
| [design/](design/README.md) | Index of the design notes (folder README) |
| [media/](media/README.md) | Optimized demo GIFs referenced from the README (each <= 4 MB). None yet. |

Other entry points: [README.md](../README.md) (setup, layout, configuration, checks), [CHANGELOG.md](../CHANGELOG.md)
(changes per ticket), [configs/](../configs/README.md), [tests/](../tests/README.md), [scripts/](../scripts/README.md),
[report/](../report/README.md).
