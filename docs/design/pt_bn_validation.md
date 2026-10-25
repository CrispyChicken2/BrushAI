# BatchNorm in the validation of the PT training (T-020)

**Problem.** The validation curve written by `brushstroke/stroke_model/train.py` is scored in eval mode, i.e. with the *running*
BatchNorm statistics (exponential average, momentum 0.1, of the last ~10 training batches). Where the weights move fast those
statistics lag behind the weights, so isolated validation points sit 0.3 to 1.2 above the plateau, always in the BCE term
(T-014, `docs/experiments/pt_training.md`, finding 3: 16 of 81 evaluations differ by more than 0.3 from re-estimated statistics),
while the training curve is smooth. At the end of a run the learning rate is tiny and the two agree within 0.003, so final
metrics were never affected: the curve was just unusable for model selection during a run.

**What.** `val.bn` (config `configs/stroke_model/pt.yaml`, section `val`) selects the statistics used for the validation:

| `val.bn` | statistics | cost |
|---|---|---|
| `running` (the behaviour until T-020) | running averages of training | none |
| `recalibrate` (**default**) | average over `val.bn_batches` (20) fixed batches of the *training distribution*, estimated at each validation | `bn_batches` forward passes |
| `batch` | statistics of each validation batch (64 patches) | none |

**How it stays harmless to training** (`validation_network`, `make_calibration_set`):
* `recalibrate` and `batch` work on `copy.deepcopy(net)`: the running statistics, the weights and the optimizer of the training
  network are never read-modify-written. `recalibrate` resets the BatchNorm of the copy, sets `momentum=None` (cumulative
  average, i.e. the plain mean of the per-batch statistics over the `K` batches) and runs the forward passes with only the
  BatchNorm layers in training mode: **dropout stays off** (a plain `copy.train()` would switch the 0.1 dropout of the
  transformer on and add noise to the statistics). `batch` sets `track_running_stats=False` on the copy's BatchNorm layers
  (batch statistics, nothing stored).
* The calibration batches come from their own generator, seeded by `val.bn_seed`, and are produced once per run: neither the
  training data stream (`data_rng`) nor the global random states are consumed, so the training trajectory, resumption
  and the bitwise resume test (T-013) are unchanged. Tests: `tests/test_pt_bn_validation.py` (state and RNG untouched,
  bitwise identical weights for the three modes, resume with `recalibrate`, recalibrated statistics = mean/unbiased variance
  of the calibration batches).

**Choice of the default**: measured on the 8000-step diagnostic (`docs/experiments/pt_training.md`, section T-020):
`recalibrate` and `batch` are equally smooth (curve std 0.30 -> 0.20, spikes above 0.3: 10 -> 2 of 71) and agree within 0.041;
`recalibrate` is the default because it does not depend on the composition or size of the validation batch (a score per
network, like the final metrics, which are computed with running statistics that have converged) and costs only 20 forward
passes per validation (every 1000 steps). The number of calibration batches hardly matters (K = 5, 20, 50 agree within 0.064).
The two spikes that remain are present in every mode: they are real fluctuations of the weights. The mode is saved in the
resolved config of each run; runs made before T-020 (`running`) keep their meaning.

**Reproducibility note.** On the GPU, two runs of the identical configuration end with different weights even with
`deterministic: true` (control measured for T-020: two 400-step runs, different SHA-256), so "the validation does not change the
training" cannot be shown by comparing GPU runs; it is shown on the CPU (`tests/test_pt_bn_validation.py`, and 40 steps of the
full network with the three modes: identical state-dict hash, `docs/experiments/data/pt_bn/bitwise.json`). Finding the
non-deterministic GPU operation is outside T-020.

**Convention.** Training metrics (`train.csv`) are unaffected. `val.csv` of runs made before T-020 uses the running statistics.

## References
Ioffe, S., Szegedy, C., *Batch Normalization*, ICML 2015 (the inference statistics are an estimate of the population
statistics); Liu, Y. et al., *Paint Transformer*, ICCV 2021 (the network, official code `Huage001/PaintTransformer`,
BatchNorm in the image and canvas CNNs); PyTorch `torch.nn.modules.batchnorm._BatchNorm` (`momentum=None`: cumulative average).
