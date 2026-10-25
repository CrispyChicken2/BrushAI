# configs

YAML configuration files. Every hyperparameter lives here and is loaded into typed config objects
(see `brushstroke/utils/config.py`). Override from the CLI with dot-notation, e.g. `optim.lr=0.01`.

Files: `default.yaml` (run settings, `brushstroke.config.RunConfig`) and `baseline_optim.yaml` (baseline a,
`brushstroke.baselines.optim.config.PaintOptimConfig`, used by `scripts/paint_optim.py`) and `animation.yaml`
(stroke-by-stroke animation, `brushstroke.animation.config.AnimationConfig`, used by `scripts/animate.py`).

`experiments/` holds the sweep inputs: `optim_sweep.yaml` (schedules, stages and variants), `images.yaml` (evaluation images with
checksums) and `baseline_t005.yaml` (frozen T-005 settings, so the sweep stays reproducible after the defaults changed).

`stroke_model/pt.yaml` is the training configuration of the Paint Transformer (`scripts/train_pt.py`): the official schedule,
the validation set, precision, resume and the loss preset.

The stroke-convention options of T-019 are off in the default configs (the official behaviour): `model.convention` (`raw` | `positive`),
`loss.l1_invariance` (`none` | `exact` | `mirror` | `rot180` | `swap`) and `loss.lambda_sign` (0 = off). They are set in
`configs/pt/model.yaml`, `configs/pt/loss_official.yaml` and `configs/stroke_model/pt.yaml`.

`pt/` holds the Paint Transformer configs: `model.yaml` (the architecture, `brushstroke.stroke_model.PTModelConfig`),
`loss_official.yaml` and `loss_paper.yaml` (the two loss presets, `PTLossConfig`), and `brushes.yaml`, the manifest of the
official brush images (URLs at a pinned commit, licence, SHA-256), read by `scripts/fetch_pt_brushes.py`. The PT-compatible rendering mode is switched in `default.yaml` (`render.mode: pt`).

`paint_pt.yaml` configures the multi-scale Paint Transformer inference (`scripts/paint_pt.py`): the checkpoint, the final-layer
variant, `|h|`, and the per-patch refinement (off by default).

`demos/` holds one partial YAML per M1 demo (`m1_portrait.yaml`, `m1_landscape.yaml`, `m1_photo.yaml`). Each inherits the
baseline and animation defaults and only sets the seed, the image, `cudnn_deterministic: true` and the animation of the demo.
