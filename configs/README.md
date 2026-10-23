# configs

YAML configuration files. Every hyperparameter lives here and is loaded into typed config objects
(see `brushstroke/utils/config.py`). Override from the CLI with dot-notation, e.g. `optim.lr=0.01`.

Files: `default.yaml` (run settings, `brushstroke.config.RunConfig`) and `baseline_optim.yaml` (baseline a,
`brushstroke.baselines.optim.config.PaintOptimConfig`, used by `scripts/paint_optim.py`) and `animation.yaml`
(stroke-by-stroke animation, `brushstroke.animation.config.AnimationConfig`, used by `scripts/animate.py`).

`experiments/` holds the sweep inputs: `optim_sweep.yaml` (schedules, stages and variants), `images.yaml` (evaluation images with
checksums) and `baseline_t005.yaml` (frozen T-005 settings, so the sweep stays reproducible after the defaults changed).

`demos/` holds one partial YAML per M1 demo (`m1_portrait.yaml`, `m1_landscape.yaml`, `m1_photo.yaml`). Each inherits the
baseline and animation defaults and only sets the seed, the image, `cudnn_deterministic: true` and the animation of the demo.
