# Learning to Paint agent training (T-029)

Three DDPG training runs (training seeds 0, 1, 2) of the Learning to Paint agent on the Met Open Access CC0 subset (99,992 images at 128 px,
97,992 for training and 2,000 held out for the curves), with the official recipe (`brushstroke/baselines/l2p/ddpg.py`, `configs/l2p/train.yaml`):
50,000 episodes of 40 steps each, with 96 paintings in parallel at every step (the counter `env_steps` adds 40 per episode, so 2,000,000 at the end, as the official `train_times`), 10 updates per episode, WGAN-GP reward, our neural renderer
(`outputs/l2p_renderer/renderer.pt`, T-028). Commands:

```
PYTHONPATH=. .venv/Scripts/python -m scripts.train_l2p seed=K output_dir=outputs/l2p_train/seedK     # K = 0, 1, 2 (resumable)
PYTHONPATH=. .venv/Scripts/python -m scripts.l2p_training_report --runs outputs/l2p_train/seed0 outputs/l2p_train/seed1 outputs/l2p_train/seed2 \
    --cost outputs/l2p_train/cost_fp32 outputs/l2p_train/cost_bf16 --released data/raw/l2p_official/actor.pkl
```

The second command regenerates everything below: [`l2p_training_tables.md`](l2p_training_tables.md), [`l2p_training.png`](l2p_training.png),
`docs/experiments/data/l2p/training_summary.json` and a copy of each history. It reads the run directories in `outputs/l2p_train/` (`history.jsonl`,
`final_validation.json`, `actor.pt` for the 2,000-image figure) and the renderer weights, none of which are in git (weights, `last.pt`, and the
run directories stay outside git). **The committed histories in `docs/experiments/data/l2p/history_*.jsonl` are the stored logs, but without `outputs/`
the tables cannot be regenerated from the repository alone** (the same situation as the T-023 report).

## 48-hour rule (`evaluation.md` section 6)

One run was measured first (100 episodes on the real subset, warm page cache): **fp32 2.053 s per episode = 28.5 h per run, 4.23 GB peak VRAM;
bf16 autocast 1.691 s = 23.5 h, 3.67 GB.** Both are far below 48 GPU-hours per run, so the rule gives **3 training seeds**. All three ran in
fp32 (`amp: none`, the default: the 17 % saving did not justify an unvalidated precision change in a GAN-reward recipe). Measured full runs:
27.7 h, 28.7 h and 28.2 h on one RTX 5070 (12 GB), one after the other (about 84.6 h in total), peak VRAM 4.23 GB. `channels_last` was tried and was slower (0.226 vs 0.158 s per update; from the run notes, not stored in the logs).

## Result

Held-out Met validation images (2,000, never trained on, disjoint by object from the evaluation set and, except for 27 objects of evaluation artists in the training part, by artist: `evaluation.md`, amendment of 2026-11-16), 128 px, 40 steps of 5 strokes, our renderer, l2 = mean squared pixel error:

| actor | after 1 step | 5 | 20 | 40 steps | blank canvas |
|---|---|---|---|---|---|
| released (`actor.pkl`, ImageNet/CelebA-trained) | 0.0187 | 0.0090 | 0.0061 | 0.00527 | 0.4205 |
| ours, seed 0 | 0.0159 | 0.0085 | 0.0059 | 0.00522 | 0.4205 |
| ours, seed 1 | 0.0163 | 0.0085 | 0.0061 | 0.00550 | 0.4205 |
| ours, seed 2 | 0.0160 | 0.0086 | 0.0061 | 0.00556 | 0.4205 |

Our three agents reach the l2 of the released actor (0.0052 to 0.0056 against 0.0053 at 40 steps, spread between seeds 6 %); this is a check that the
recipe is reproduced, not a benchmark claim (different data, our renderer; no LPIPS is published for the released model). The full table, the
per-run time and the cost runs are in [`l2p_training_tables.md`](l2p_training_tables.md); curves in [`l2p_training.png`](l2p_training.png).

Training l2 falls from 0.23 (episode 50) to 0.015 (1,000), 0.008 (10,000) and 0.0055 to 0.0061 (30,000 to 50,000; seed 2 ends highest). **The validation result is the 2,000-image figure of the table above (0.0052 / 0.0055 / 0.0056).** The curve in the figure and the "val l2 (480)" column use the first 480 held-out images in file order, which is **not a random sample**: the file follows the fetch order, by stratum, so these 480 are 222 photographs, 201 drawings, 57 paintings and no object or print. They are easier (seed 0: 0.0039 on them against 0.0056 on the other 1,520), which is why 0.0039 to 0.0042 sits below the 2,000-image figure. Read the curve for its shape (the learning-rate drops at episodes 10,000 and 20,000 give the visible steps), not its level. Later runs take a seeded random subset (`validate_sample: random`). The debug run (9,000 images, 1,500 episodes) ended at validation l2 0.0242 (its last point; `history_dbg.jsonl`).

## Checkpoint rule

**The final checkpoint (episode 50,000) is the agent**, the rule decided in `evaluation.md` before any run. It is never chosen by test-set
results; the validation curve (held-out Met images only) is a diagnostic. On the 480-image curve (biased subset, see above) the best checkpoint of each run is the last or near-last
(seed 0: 50,000; seed 1: 49,600; seed 2: 48,000) and the differences are within the noise of single samples; the rule is not based on it. `actor.pt` is loadable by `load_actor`.

## Surprises, investigated

1. **Huge discriminator values at the start** (episode 50: `gan_fake` about -700 to -800, gradient penalty 145 to 178). Cause: the WGAN-GP critic is
   unbounded and the first 400 environment steps only fill the replay; the values fall to a few units by episode 1,000 (penalty 6 to 8, 3.5 at the end). No non-finite value in any run (0 of 3 x 1,000 log rows).
2. **Isolated spikes of the validation l2** (2 to 6 times the neighbouring value for one 100-episode sample, e.g. seed 2: episodes 1,400 (5.2 x), 2,500 (3.3 x), 38,600 (2.4 x), each against the median of the three samples on either side; 45,600 and 45,700 stayed 1.8 x up for two samples), always back to the trend within 100 to 200 episodes. Not NaNs; the training l2 does not show them. Probable
   cause (not isolated by an experiment): the batch-norm running statistics, which in our runs also absorb the target actor's forward passes (the sharing deviation below), and the non-stationary GAN reward make single policy
   snapshots worse in eval mode. Not fixed; this is why the final checkpoint rule matters: the 50,000 point is smooth for all three seeds.
3. **Q and critic loss**: Q rises to about 7 (episode 1,000) and settles at 3.4 to 3.7; critic loss settles at 0.006 to 0.013 (seed 2 is the noisiest). Consistent with the official logs' orders of magnitude.
4. **First cost measurement 3.8 s per episode**: cold page cache of the 4.9 GB image array (and a concurrent download); the warm figure is 2.05 s. Page-cache warm-up is an input of the cost, not of the algorithm.

## Deviations from the paper and the official run

Listed in `docs/design/learning_to_paint.md` section 7: training data (Met CC0, 98k instead of CelebA/ImageNet 200k), the composition of the Met
subset (not the 40/30/30 of `evaluation.md` 8.2: only 3,050 usable Met paintings), 27 training objects by evaluation artists, the aspect rule 1:2 to 2:1, **the target actor sharing the batch-norm running statistics of the online actor** (the official `DDPG.__init__` calls `hard_update`, which shares the buffers, and then `.to(device)`, which separates them on the GPU: officially the sharing is gone before training; ours stays, `share_bn_buffers: true`, `false` = official; the results match the released actor), our GPU replay layout (the replay is not
checkpointed and is refilled on resume; no run was resumed here), BGR training images as in the official code, and the use of our renderer. The three runs used `validate_sample: first` and `share_bn_buffers: true` (config keys added afterwards, same behaviour).
