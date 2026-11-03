"""The neural stroke renderer of Learning to Paint and its training on synthetic strokes (no dataset).

Architecture copied from ``baseline/Renderer/model.py`` (official code, commit 615c070, MIT): 10 stroke numbers ->
4 fully connected layers (512, 1024, 2048, 4096) -> ``(16, 16, 16)`` -> conv / PixelShuffle x3 -> sigmoid -> ``1 - x``,
``(B, 128, 128)``. The state-dict keys are those of the official ``renderer.pkl`` (checked by loading it strictly).
Training follows ``baseline/train_renderer.py``: batches of 64 uniform random strokes in ``[0, 1]^10``, ground truth
:func:`brushstroke.baselines.l2p.bezier.stroke_coverage` (the official ``draw`` on the GPU), MSE loss, Adam, learning rate 1e-4
for the first 200,000 steps, 1e-5 up to 400,000 and 1e-6 up to 500,000. Deviation: the official script *resumes* from an existing
``renderer.pkl``; here training starts from a fresh initialization (and the official weights are only a comparison).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from brushstroke.baselines.l2p.bezier import stroke_coverage


class FCN(nn.Module):
    """Official stroke renderer: ``(B, 10)`` -> ``(B, 128, 128)`` equal to ``1 - coverage``."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.reshape(-1, 128, 128)


def load_renderer(path: str | Path, device: torch.device | str = "cpu") -> FCN:
    """Load a renderer state dict (the official ``renderer.pkl`` or ours) strictly, in eval mode."""
    net = FCN()
    state = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(state.get("model", state))
    return net.to(device).eval()


def decode(
    strokes: torch.Tensor, canvas: torch.Tensor, renderer: nn.Module, bundle: int = 5
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Paint a bundle of strokes on ``canvas`` with the neural renderer (official ``decode``).

    Args:
        strokes: ``(B, bundle * 13)`` in [0, 1]: per stroke 10 shape numbers then RGB.
        canvas: ``(B, 3, 128, 128)``.

    Returns:
        The canvas after the bundle and the list of the ``bundle`` intermediate canvases (after each stroke).
    """
    x = strokes.reshape(-1, 13)
    cov = 1 - renderer(x[:, :10])  # (B * bundle, 128, 128) coverage
    color = cov[:, None] * x[:, 10:13].reshape(-1, 3, 1, 1)
    cov = cov.reshape(-1, bundle, 1, 128, 128)
    color = color.reshape(-1, bundle, 3, 128, 128)
    steps = []
    for i in range(bundle):
        canvas = canvas * (1 - cov[:, i]) + color[:, i]
        steps.append(canvas)
    return canvas, steps


@dataclass
class RendererTrainConfig:
    """``configs/l2p/renderer.yaml``: the official schedule."""

    seed: int = 0
    steps: int = 500_000
    batch_size: int = 64
    lr_stages: tuple[float, ...] = (1e-4, 1e-5, 1e-6)
    stage_steps: tuple[int, ...] = (200_000, 400_000)  # the learning rate changes after these steps
    log_every: int = 1000
    eval_strokes: int = 1000  # held-out random strokes of the periodic validation
    output_dir: str = "outputs/l2p_renderer"
    device: str = "auto"


def lr_at(step: int, cfg: RendererTrainConfig) -> float:
    for stage, limit in enumerate(cfg.stage_steps):
        if step < limit:
            return cfg.lr_stages[stage]
    return cfg.lr_stages[len(cfg.stage_steps)]


def random_strokes(n: int, gen: torch.Generator, device: torch.device | str) -> torch.Tensor:
    """``(n, 10)`` strokes uniform in ``[0, 1]^10`` (the official sampling)."""
    return torch.rand(n, 10, generator=gen).to(device)


@torch.no_grad()
def renderer_error(
    net: nn.Module, strokes: torch.Tensor, truth: torch.Tensor | None = None, block: int = 250
) -> dict[str, float]:
    """Error of a renderer against the analytic coverage of ``(n, 10)`` strokes.

    Returns the mean l2 (per-pixel MSE averaged over strokes), the mean absolute error, the largest absolute pixel error
    and the MSE of the worst stroke.
    """
    truth = stroke_coverage(strokes) if truth is None else truth
    mse, mae, mx = [], [], 0.0
    for a in range(0, strokes.shape[0], block):
        pred = 1 - net(strokes[a : a + block])
        err = pred - truth[a : a + block]
        mse.append(err.square().flatten(1).mean(1))
        mae.append(err.abs().flatten(1).mean(1))
        mx = max(mx, float(err.abs().max()))
    mse_t, mae_t = torch.cat(mse), torch.cat(mae)
    return {
        "mse": float(mse_t.mean()),
        "mae": float(mae_t.mean()),
        "max_abs": mx,
        "worst_stroke_mse": float(mse_t.max()),
    }


def train_renderer(cfg: RendererTrainConfig, log: bool = True) -> Path:
    """Train from scratch on synthetic strokes; resumable (``last.pt`` in ``output_dir``). Returns the final weights path."""
    from brushstroke.utils.device import get_device
    from brushstroke.utils.seed import seed_everything

    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    net = FCN().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr_stages[0])
    step = 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
    gen = torch.Generator().manual_seed(
        cfg.seed * 1_000_003 + step
    )  # a resumed run draws another seeded stream
    val = random_strokes(cfg.eval_strokes, torch.Generator().manual_seed(10_000_019), device)
    val_truth = stroke_coverage(val)
    history = out / "history.jsonl"
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    while step < cfg.steps:
        net.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        f = random_strokes(cfg.batch_size, gen, device)
        loss = F.mse_loss(net(f), 1 - stroke_coverage(f))
        opt.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        run_loss += float(loss.detach())
        run_n += 1
        step += 1
        if step % cfg.log_every == 0 or step == cfg.steps:
            net.eval()
            rec = {
                "step": step,
                "lr": lr_at(step - 1, cfg),
                "train_mse": run_loss / run_n,
                "seconds": time.time() - t0,
                **{f"val_{k}": v for k, v in renderer_error(net, val, val_truth).items()},
            }
            run_loss, run_n = 0.0, 0
            with history.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step}, last)
            if log:
                print(rec, flush=True)
    final = out / "renderer.pt"
    torch.save(net.state_dict(), final)
    (out / "config.json").write_text(json.dumps(asdict(cfg)), encoding="utf-8")
    return final
