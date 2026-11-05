# Learning to Paint neural renderer (T-028)

Training: `python -m scripts.train_l2p_renderer` (`configs/l2p/renderer.yaml`: official schedule, 500,000 steps, batch 64, Adam, lr 1e-4 / 1e-5 / 1e-6, fresh
initialization, MSE against the analytic rasterizer, 14,149 s on an RTX 5070). Report: `python -m scripts.l2p_renderer_report`, data in
`docs/experiments/data/l2p/renderer_report.json`. Test strokes: 1,000 uniform random strokes, seed 777 (no training batch uses it).

## Error against the analytic rasterizer (equal to the official cv2 `draw`)

| Renderer | mean MSE | mean abs error | largest pixel error | worst stroke MSE |
|----------|----------|----------------|---------------------|------------------|
| Ours (500k steps) | 5.27e-4 | 4.54e-3 | 0.983 | 3.20e-3 |
| Official `renderer.pkl` | 6.09e-4 | 4.90e-3 | 0.989 | 4.94e-3 |

Ours against the official renderer: MSE 3.68e-4, largest pixel difference 0.860.
**Tolerance of the protocol (fixed in `docs/design/learning_to_paint.md`): MSE <= 1.0e-3 and <= 1.25 x the official one: met** (0.87 x the official).
The largest pixel errors sit on stroke edges (thin high-contrast outlines); the errors are the same order for both networks.

Training curve (validation on 1,000 held-out strokes, `outputs/l2p_renderer/history.jsonl`): 9.9e-4 at 112k steps, 5.32e-4 at 500k (mae 4.51e-3); the lr drops at 200k and
400k flatten it. Train MSE 5.2e-4, so no overfitting (infinite synthetic data).

## Smoke run of the released actor (not a result)

`data/raw/l2p_official/image/lisa.png` (official sample, a 128 x 128 image upscaled to 512 x 512), `RolloutConfig()` defaults (`max_step` 100, 4,250 strokes). l2 to the target (mean squared pixel error on [0, 1]):

| Strokes | official renderer | our renderer |
|---------|-------------------|--------------|
| 0 | 0.1837 | 0.1837 |
| 5 | 0.02254 | 0.02311 |
| 25 | 0.00759 | 0.00709 |
| 100 | 0.00293 | 0.00302 |
| 500 | 0.001131 | 0.001179 |
| 1700 | 0.000579 | 0.000598 |
| 2000 | 0.000541 | 0.000548 |
| 4250 | 0.000404 | 0.000410 |

The two renderers give the same behaviour of the released actor within a few per cent, which supports using our renderer in T-029. One image only: a plumbing check of the
rollout and the producer skeleton, not a comparison of methods. Parity of the rollout with `test.py` is established on CPU by a transcription test (see the design note, section 5).
