# Learning to Paint: study note, Bezier rasterizer, neural renderer (T-028)

Huang, Heng, Zhou, *Learning to Paint With Model-based Deep Reinforcement Learning*, ICCV 2019, arXiv 1903.04411. Official code:
`megvii-research/ICCV2019-LearningToPaint`, pinned at commit `615c0707ce265706af41fd63a78d0c7824639f75` (called "official" below; paths relative to
the repository). The copy used here lives outside git in `data/raw/l2p_official/`. The protocol around it is `evaluation.md` sections 7-8.

## 1. Licences and what may be done with them

| Item | Licence | Verdict |
|------|---------|---------|
| Official code (`LICENSE`, MIT, 2019 Zhewei Huang) | MIT | The architectures were re-implemented from it with attribution in the docstrings (`brushstroke/baselines/l2p/`); no file is vendored in git. |
| Released weights `renderer.pkl`, `actor.pkl` (CelebA) | **No licence statement.** README links Google Drive and a Baidu mirror | **Local reference only**: fetched, pinned by SHA-256, never committed, never redistributed, never trained on or fine-tuned. Running the released actor as a reference for the agent check of T-030 is accepted on that basis (comparison, not derivative); the verdict is repeated in the T-029 data card. |
| Pins (SHA-256) | `renderer.pkl` `34639d4ab4f30807fb056bf4b71de0c3f434d6cbe8600cf97b6af3a855a5ca8e`; `actor.pkl` `75e908a42d4c90ad092892a9183bd467453696d8e2202b9640f9a5b4a488eab9` | Sources (official README): Google Drive id `1-7dVdjCIZIxh8hHJnGTK-RA1-jL1tor4` (renderer.pkl) and `1a3vpKgjCVXHON4P7wodqhCgCMPgg1KeR` (actor.pkl), Baidu mirror `https://pan.baidu.com/s/1GELBQCeYojPOBZIwGOKNmA` (code `aq8n`). |

## 2. Specification (paper and official code)

**Stroke.** Quadratic Bezier with 13 numbers in [0, 1]: `x0, y0, x1, y1, x2, y2` (`baseline/Renderer/stroke_gen.py::draw`: the control point is relative, `x1 = x0 + (x2 - x0) * x1`, same for y; reproduced verbatim in `bezier.draw_reference`),
`z0, z2` (radius at the ends, the code's "z"), `w0, w2` (transparency at the ends, the code's "w"), then RGB (3). 10 shape numbers go to the neural
renderer, the colour is applied by the environment. The paper's figure 3 draws 3 fully connected layers for the renderer, the code has 4.

**Neural renderer** (`baseline/Renderer/model.py::FCN`): 10 -> FC 512 -> 1024 -> 2048 -> 4096 (ReLU) -> `(16,16,16)` -> three
(conv, conv, PixelShuffle(2)) blocks -> sigmoid -> `1 - x`, output 128 x 128. Training (`baseline/Renderer/train_renderer.py`): 500,000 steps, batch 64,
Adam, lr 1e-4 then 1e-5 from 200k then 1e-6 from 400k, MSE against `draw` on uniform random strokes (no data set). The script resumes from an
existing `renderer.pkl`.

**Actor** (`baseline/DRL/actor.py`, `ddpg.py`): ResNet-18 with BatchNorm, input 9 channels (canvas 3, target 3, step / max_step 1, CoordConv 2), output 65 =
5 strokes x 13 (sigmoid). **Critic** (`ResNet_wobn(3 + 9, 18, 1)`, `ddpg.py` line 53): ResNet-18 without BatchNorm, TReLU activation, 12 input channels (the 9 of the actor plus the previous canvas "for better prediction"), one output. **Reward**: WGAN-GP discriminator (`wgan.py`, `update_gan`) on (canvas, target) pairs; the reward is the increase of the discriminator score between consecutive canvases. **DDPG**: action bundle k = 5, discount 0.95^5, 128 x 128 canvas, 40 steps per episode, model-based (the critic's gradient
passes through the renderer).

**Inference** (`baseline/test.py`, `--divide 4`, `--max_step 40`): global stage of `max_step` steps on the whole image resized to 128 x 128, then the image is cut
in `divide x divide` patches (resized to 128), each painted for `max_step` steps in a batch of 16 patches, patches are pasted back with a seam smoothing
(`smooth`), all in OpenCV BGR order. With `divide != 1` the step count is halved per stage (`test.py` lines 122-123). Two settings exist: the `test.py` default `--max_step 40` gives 20 global steps then 20 steps on each of the 16 patches, 5 strokes per step: 20 x 5 + 16 x 20 x 5 = **1,700 strokes** at 512 x 512; the command documented in the official README, `test.py --max_step=100 --actor=actor.pkl --renderer=renderer.pkl --img=image/test.png --divide=4`, gives 50 x 5 + 16 x 50 x 5 = **4,250 strokes**. **Decision (review round 1): the default of `RolloutConfig` is `max_step = 100`, the README command.** With 40 the protocol grid would stop at 1,500 and AUC@2000 (P2, the headline endpoint), the 1,750 and 2,000 checkpoints and tau_2000 would be unreachable ("n/a (max n = 1700)", `evaluation.md` 3.3 and 7); with 100 the budget 2,000 is reached and the producer is cut at the budget. Cost of one 512 px image at `max_step` 100: about 0.76 s and 0.21 GB peak VRAM in the reviewer's re-run after the canvases moved to the CPU (the first measurement, before that change, was 0.63 s and 1.16 GB; these are not regenerable from git, the probe is outside the repository). The 40-step variant stays available as `RolloutConfig(max_step=40)`. The paper does not describe the high-resolution procedure; it exists
only in the code.

## 3. Reported results and what is checkable

The paper has **no numeric table**: curves of test l2 (figs. 8-9) and ratios (l2 5x / 20x smaller than two DDPG variants, 3x smaller than SPIRAL at 20 strokes
on CelebA; 2.1 s per 200 strokes on a CPU, 9.5x faster on a 2080 Ti; about 40 h of training). No LPIPS is published, so no "LPIPS reproduced from the paper" claim is made
(`evaluation.md` 8.3). Checkable here: (i) the neural renderer against the analytic rasterization (section 5); (ii) our agent against the released actor (T-030).
**Renderer tolerance fixed here** (check 1 of 8.3, on 1,000 random strokes, seed 777): mean MSE <= 1.0e-3 *and* not above 1.25 x that of the official weights.

## 4. Our implementation (`brushstroke/baselines/l2p/`)

| File | Content |
|------|---------|
| `bezier.py` | `stroke_coverage`: batched GPU analytic rasterizer, equal to the official cv2 `draw` (max difference 1.2e-7, 16 ms per batch of 64); `draw_reference` is the official function verbatim, used by the test. |
| `renderer.py` | `FCN` (strict-loads `renderer.pkl`), `train_renderer` (official schedule, resumable `last.pt`, `history.jsonl`), `renderer_error`. |
| `actor.py` | `Actor` ResNet-18 (strict-loads `actor.pkl`), CoordConv channels. |
| `rollout.py` | Global + divided inference, seam smoothing, `Painting.canvas(k)` giving the exact canvas after k strokes, `progressive_canvases`. |
| `producer.py` | `L2PProducer`: harness adapter (canvases after exactly n strokes, black background, BGR flip, replicates = training seeds). Not registered in `configs/eval/protocol.yaml` (T-030 does it once an agent exists). |

Training the actor (DDPG + WGAN-GP critic) is T-029; the critic and discriminator are not implemented here.

## 5. Deviations

1. The renderer is trained **from a fresh initialization** (the official script resumes from `renderer.pkl`); the official weights are only a comparison.
2. The ground truth is a GPU rasterizer equal to the cv2 `draw` rather than cv2 itself (same schedule, same loss).
3. We follow the code, not the paper figure, for the number of FC layers (4) and the meaning of z (radius) and w (transparency).
4. **The official `test.py` itself was not executed** (running the downloaded script was refused by the sandbox classifier and not worked around). Parity is established
   by a line-by-line transcription of `test.py` instead: `tests/test_l2p.py::test_rollout_matches_a_transcription_of_the_official_test_script` (skipped without the official weights) shows that on CPU with float inputs every global-stage stroke canvas and every divided-step canvas equal the transcription (max difference below 1e-6; the reviewer's probe at divide 4, max_step 40 gave exactly 0.0). On CUDA the closed loop drifts after about 15 strokes through kernel-level float noise (final l2 4.98e-4 vs 5.12e-4), and ours is deterministic run to run.
5. `test.py` resizes the 8-bit image (quantized target) and saves 8-bit frames; ours resizes and keeps floats. With 8-bit inputs the trajectory changes from about stroke 5 (closed loop) and the final l2 is 4.95e-4 against 4.98e-4: no systematic effect, listed for completeness.
6. Stroke counting follows the protocol (every stroke of every patch), not the paper's per-image budgets.

## 6. Results

See `docs/experiments/l2p_renderer.md`: ours vs analytic 5.27e-4 MSE (official weights 6.09e-4), ours vs official 3.68e-4; tolerance met.

## 7. Agent training (T-029)

**Recipe** (`brushstroke/baselines/l2p/ddpg.py`, `configs/l2p/train.yaml`; official `train.py` defaults of commit 615c070): model-based DDPG, actor ResNet-18 (BN), critic
`ResNet_wobn(3+9, 18, 1)` (weight norm, TReLU, 12 input channels: canvas before and after, target, (t+1)/max_step, 2 coordinate channels), WGAN-GP discriminator on
(target, canvas) with lambda 10, reward = D(target, canvas after) - D(target, canvas before); discount 0.95^5, tau 0.001, 96 environments, batch 96, 40 steps per episode,
50,000 episodes of 40 steps (`env_steps` = 40 per episode, 2,000,000 in total, as the official `train_times`), 10 updates per episode, learning rates by stage (critic 3e-4 / 1e-4 / 3e-5, actor 1e-3 / 3e-4 / 1e-4 at episodes 10,000 and 20,000), Adam,
replay 800 x 40 transitions, 400 warm-up steps, no exploration noise, target networks share the batch-norm buffers of the online ones (a deviation from the official GPU run, item 7), canvas re-quantized to uint8 each step, horizontal-flip augmentation.
**Data**: the Met CC0 subset of `docs/data_sources.md`. **Checkpoint rule**: the final checkpoint. **Cost and results**: `docs/experiments/l2p_training.md` (3 seeds, 28 h each).

**Deviations from the official run:**
1. Training images are Met CC0 (97,992 at 128 px) instead of CelebA / ImageNet (licence), see `evaluation.md` 8.2 and its amendment.
2. The subset is built from the Met's Hugging Face export, not the Met API (the API blocks bursts with HTTP 403).
3. Aspect rule 1:2 to 2:1 for training (the evaluation set keeps 3:4 to 4:3): the narrow rule rejected two thirds of the paintings and photographs.
4. Composition: the Met has only 3,050 usable paintings after exclusions, so the 40/30/30 split is not reachable; the subset is dominated by objects and prints.
5. The replay buffer stores every canvas once on the GPU (ring of 333 steps x 96) with one target per episode; the previous canvas is the previous slot. Same transitions, 1.6 GB. It is not checkpointed: after a resume it is refilled with fresh episodes (no run needed a resume).
6. Images are fed as BGR, as in the official code (`actor.pt` and `actor.pkl` therefore share the channel convention; `L2PProducer` flips accordingly).
7. The renderer is ours (T-028), not `renderer.pkl`.
   **BatchNorm buffers**: the official `DDPG.__init__` runs `hard_update` (shares the buffer tensors of target and online actor) and then `choose_device()` (`.to(device)` per module), which re-assigns the buffers: on the GPU the target actor keeps its own statistics. Ours builds on the device first, so the sharing stays and the online running statistics also absorb the target's train-mode forward passes. The T-029 agents were trained that way (`share_bn_buffers: true`, still the default; `false` reproduces the official behaviour).
   **Artists**: 27 training objects are by evaluation artists (`configs/l2p/trainset_known_leaks.yaml`, `evaluation.md` amendment 2026-11-16); not retrained. **Validation curve**: first 480 held-out images, a stratum-biased subset; the validation figure is the 2,000-image one.
8. Optional bf16 autocast (`amp: bfloat16`) exists and was measured (23.5 h per run) but not used.
