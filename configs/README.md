# configs

YAML configuration files. Every hyperparameter lives here and is loaded into typed config objects
(see `brushstroke/utils/config.py`). Override from the CLI with dot-notation, e.g. `optim.lr=0.01`.

Files: `default.yaml` (run settings, `brushstroke.config.RunConfig`) and `baseline_optim.yaml` (baseline a,
`brushstroke.baselines.optim.config.PaintOptimConfig`, used by `scripts/paint_optim.py`) and `animation.yaml`
(stroke-by-stroke animation, `brushstroke.animation.config.AnimationConfig`, used by `scripts/animate.py`).
