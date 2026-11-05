"""Model-based DDPG with a WGAN-GP reward: the training of the Learning to Paint agent (T-029).

Reimplementation of the official recipe (``baseline/train.py``, ``baseline/DRL/{ddpg,critic,wgan,rpm}.py``, ``baseline/env.py``,
``baseline/DRL/multi.py``; commit 615c070, MIT) with the Met CC0 subset of :mod:`brushstroke.baselines.l2p.trainset` and our neural
renderer. What the official code does, and what is kept here:

* *Environment* (``env.py``): ``env_batch`` paintings in parallel, 40 steps of 5 strokes each (one actor output of 65 numbers), canvas and
  target stored as ``uint8`` in BGR order, the canvas re-quantized to ``uint8`` after every step (``(canvas * 255).byte()``).
* *Actor* ``ResNet(9, 18, 65)`` (BatchNorm), input ``(canvas, target, t / max_step, CoordConv)`` (:mod:`~.actor`); *critic*
  ``ResNet_wobn(12, 18, 1)`` with weight normalization and TReLU, input ``(canvas0, canvas1, target, (t + 1) / max_step, CoordConv)``
  (the canvas after the action is rendered by the neural renderer); *discriminator* of the WGAN-GP on ``(target, canvas)``.
* *Reward*: ``D'(target, canvas1) - D'(target, canvas0)`` with the target discriminator ``D'``; the L2 reward the environment also
  computes is not used by the update (as in the official ``update_policy``). The critic learns the remainder:
  ``Q(s, a) + r`` regresses ``r + gamma * (1 - done) * (Q'(s', a') + r')``, ``gamma = 0.95^5``.
* *Updates*: after every episode, ``episode_train_times`` (10) updates of batch ``batch_size`` (96): one discriminator step, one critic
  step, one actor step (loss ``-(Q + r)`` through the renderer), soft update of the three target networks (``tau = 0.001``). Adam, learning
  rates by episode stage. No exploration noise (``noise_factor = 0``): the official default.
* *Target networks* share the BatchNorm buffers of the online actor (``hard_update`` copies the dict of buffers, not the values) and
  stay in train mode during the updates: kept, so the target actor's forward also updates the shared running statistics.
* *Replay*: a ring of ``rmsize * max_step`` transitions (32,000), uniform sampling. The official buffer stores two 7-channel states per
  transition (7.3 GB of host memory); here it stores each canvas once on the GPU (the previous canvas of a transition is the canvas
  of the previous step, a blank canvas at the start of an episode) and the target once per episode (1.6 GB). Same content, same sampling.

Deviations (``docs/design/learning_to_paint.md``): RGB images are stored, the environment feeds BGR; the data set is the Met subset;
the replay buffer is not part of the checkpoint (after a resume it is refilled for ``warmup`` steps before the updates restart); optional
bfloat16 autocast of the actor and critic forward passes (``amp``), off by default.
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from brushstroke.baselines.l2p.actor import Actor, coord_channels
from brushstroke.baselines.l2p.renderer import decode, load_renderer

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.nn.utils import (
        weight_norm,
    )  # the official parametrization (keys weight_g / weight_v)

WIDTH = 128
STROKES = 5  # strokes per actor output
ACTION = STROKES * 13


class TReLU(nn.Module):
    """``relu(x - alpha) + alpha`` with a learned scalar ``alpha`` (official ``TReLU``)."""

    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x - self.alpha) + self.alpha


def _conv(cin: int, cout: int, k: int, stride: int, pad: int) -> nn.Module:
    return weight_norm(nn.Conv2d(cin, cout, k, stride, pad, bias=True))


class WNBlock(nn.Module):
    """Basic residual block of the critic: weight-normalized convolutions, TReLU, no BatchNorm."""

    def __init__(self, cin: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = _conv(cin, planes, 3, stride, 1)
        self.conv2 = _conv(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != planes:
            self.shortcut = nn.Sequential(_conv(cin, planes, 1, stride, 0))
        self.relu_1, self.relu_2 = TReLU(), TReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.relu_1(self.conv1(x)))
        return self.relu_2(out + self.shortcut(x))  # type: ignore[no-any-return]


class Critic(nn.Module):
    """Official ``ResNet_wobn(12, 18, 1)``: ``(B, 12, 128, 128)`` -> ``(B, 1)``."""

    def __init__(self, num_inputs: int = 12, num_outputs: int = 1) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = _conv(num_inputs, 64, 3, 2, 1)
        self.layer1 = self._layer(64, 2, 2)
        self.layer2 = self._layer(128, 2, 2)
        self.layer3 = self._layer(256, 2, 2)
        self.layer4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, num_outputs)
        self.relu_1 = TReLU()

    def _layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(WNBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu_1(self.conv1(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(F.avg_pool2d(x, 4).flatten(1))  # type: ignore[no-any-return]


class Discriminator(nn.Module):
    """Official WGAN-GP discriminator on ``(target, canvas)`` pairs: ``(B, 6, 128, 128)`` -> ``(B, 1)``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = _conv(6, 16, 5, 2, 2)
        self.conv1 = _conv(16, 32, 5, 2, 2)
        self.conv2 = _conv(32, 64, 5, 2, 2)
        self.conv3 = _conv(64, 128, 5, 2, 2)
        self.conv4 = _conv(128, 1, 5, 2, 2)
        self.relu0, self.relu1, self.relu2, self.relu3 = TReLU(), TReLU(), TReLU(), TReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu0(self.conv0(x))
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        return F.avg_pool2d(self.conv4(x), 4).view(-1, 1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """``target <- (1 - tau) target + tau source`` on the parameters (the buffers are not touched, as in the official code)."""
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters(), strict=True):
            t.mul_(1.0 - tau).add_(s, alpha=tau)


def share_buffers(target: nn.Module, source: nn.Module) -> None:
    """Official ``hard_update`` buffer part: the target uses the *same* buffer tensors as the source (BatchNorm statistics)."""
    for m1, m2 in zip(target.modules(), source.modules(), strict=True):
        m1._buffers = m2._buffers.copy()  # noqa: SLF001


def hard_update(target: nn.Module, source: nn.Module) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters(), strict=True):
            t.copy_(s)
    share_buffers(target, source)


@contextmanager
def frozen(net: nn.Module) -> Iterator[None]:
    """Stop computing weight gradients of ``net`` (the input gradient still flows)."""
    flags = [p.requires_grad for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, f in zip(net.parameters(), flags, strict=True):
            p.requires_grad_(f)


@dataclass
class DDPGConfig:
    """``configs/l2p/train.yaml``: the official ``train.py`` defaults unless noted."""

    seed: int = 1234
    episodes: int = 50_000  # train_times 2,000,000 env steps / max_step
    max_step: int = 40
    env_batch: int = 96
    batch_size: int = 96
    rmsize: int = 800  # replay of rmsize * max_step transitions
    warmup: int = 400  # env steps without update
    discount: float = 0.95**5
    tau: float = 0.001
    episode_train_times: int = 10
    stage_episodes: tuple[int, int] = (10_000, 20_000)  # lr stages by episode
    critic_lrs: tuple[float, float, float] = (3e-4, 1e-4, 3e-5)
    actor_lrs: tuple[float, float, float] = (1e-3, 3e-4, 1e-4)
    gan_lr: float = 3e-4
    gan_betas: tuple[float, float] = (0.5, 0.999)
    gan_lambda: float = 10.0
    gan_tau: float = 0.001
    amp: str = "none"  # "none" or "bfloat16": autocast of the actor and critic forward passes
    renderer: str = "outputs/l2p_renderer/renderer.pt"
    trainset: str = "data/raw/l2p_met"
    bgr: bool = (
        True  # the environment feeds BGR images (the released actor and our producer convention)
    )
    validate_every: int = 100  # episodes
    validate_images: int = 480  # held-out images (official: 5 episodes of 96)
    log_every: int = 50  # episodes
    checkpoint_every: int = 500  # episodes
    output_dir: str = "outputs/l2p_train/run0"
    device: str = "auto"
    max_episodes_this_run: int = (
        0  # 0 = until ``episodes``; a smaller value stops early (cost measurements)
    )


def stage(episode_end: int, cfg: DDPGConfig) -> int:
    """Learning-rate stage of the update after the episode that brings the count to ``episode_end``."""
    return (
        0
        if episode_end < cfg.stage_episodes[0]
        else (1 if episode_end < cfg.stage_episodes[1] else 2)
    )


class Replay:
    """Ring buffer of ``slots * env_batch`` transitions, canvases kept once (see the module docstring)."""

    def __init__(
        self, transitions: int, env_batch: int, max_step: int, device: torch.device
    ) -> None:
        self.slots = transitions // env_batch
        self.B, self.T, self.device = env_batch, max_step, device
        self.episodes = self.slots // max_step + 2
        self.canvas = torch.zeros(
            self.slots, env_batch, 3, WIDTH, WIDTH, dtype=torch.uint8, device=device
        )
        self.action = torch.zeros(self.slots, env_batch, ACTION, device=device)
        self.gt = torch.zeros(
            self.episodes, env_batch, 3, WIDTH, WIDTH, dtype=torch.uint8, device=device
        )
        self.g = 0  # env steps stored

    def add(self, canvas_next: torch.Tensor, action: torch.Tensor, gt: torch.Tensor) -> None:
        self.canvas[self.g % self.slots] = canvas_next
        self.action[self.g % self.slots] = action
        if self.g % self.T == 0:
            self.gt[(self.g // self.T) % self.episodes] = gt
        self.g += 1

    def __len__(self) -> int:
        return min(self.g, self.slots - 1) * self.B

    def sample(self, n: int, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
        """``(canvas0, canvas1, gt, t, action, done)``: uint8 canvases and target, ``t`` the step of the action, ``done`` at the last step."""
        lo = max(0, self.g - self.slots + 1)
        g = torch.randint(lo, self.g, (n,), generator=gen, device=self.device)
        b = torch.randint(0, self.B, (n,), generator=gen, device=self.device)
        t = g % self.T
        canvas1 = self.canvas[g % self.slots, b]
        canvas0 = self.canvas[(g - 1) % self.slots, b]
        canvas0 = torch.where((t == 0).view(-1, 1, 1, 1), torch.zeros_like(canvas0), canvas0)
        gt = self.gt[(g // self.T) % self.episodes, b]
        return canvas0, canvas1, gt, t, self.action[g % self.slots, b], t == self.T - 1


class PaintEnv:
    """``env_batch`` paintings in parallel (official ``Paint``): uint8 canvas and target, the canvas re-quantized after every step."""

    def __init__(
        self, images: np.ndarray, batch: int, max_step: int, device: torch.device, bgr: bool
    ) -> None:
        self.images, self.B, self.T, self.device, self.bgr = images, batch, max_step, device, bgr
        self.t = 0

    def load(self, ids: np.ndarray, flip: np.ndarray | None = None) -> torch.Tensor:
        """Target batch ``(B, 3, 128, 128)`` uint8 on the device from the image indices (sorted for the memory map)."""
        order = np.argsort(ids)
        arr = np.empty((len(ids), WIDTH, WIDTH, 3), np.uint8)
        arr[order] = self.images[np.asarray(ids)[order]]
        if flip is not None:
            arr[flip] = arr[flip][:, :, ::-1]
        if self.bgr:
            arr = arr[..., ::-1]
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(self.device)

    def reset(self, ids: np.ndarray, flip: np.ndarray | None = None) -> None:
        self.gt = self.load(ids, flip)
        self.canvas = torch.zeros_like(self.gt)
        self.t = 0

    @torch.no_grad()
    def step(self, action: torch.Tensor, renderer: nn.Module) -> None:
        canvas, _ = decode(action, self.canvas.float() / 255, renderer, STROKES)
        self.canvas = (canvas * 255).clamp(0, 255).byte()
        self.t += 1

    def dist(self) -> torch.Tensor:
        """Per-image mean squared error between canvas and target in [0, 1]: ``(B,)``."""
        return (((self.canvas.float() - self.gt.float()) / 255) ** 2).flatten(1).mean(1)


class DDPG:
    """Actor, critic, discriminator, their targets and the update of the official ``DDPG`` class."""

    def __init__(self, cfg: DDPGConfig, renderer: nn.Module, device: torch.device) -> None:
        self.cfg, self.renderer, self.device = cfg, renderer, device
        for p in renderer.parameters():
            p.requires_grad_(False)
        self.actor, self.actor_target = Actor(9, ACTION).to(device), Actor(9, ACTION).to(device)
        self.critic, self.critic_target = Critic().to(device), Critic().to(device)
        self.D, self.D_target = Discriminator().to(device), Discriminator().to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-2)
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=cfg.gan_lr, betas=cfg.gan_betas)
        self.targets()
        for p in self.D_target.parameters():
            p.requires_grad_(False)
        self.coord = coord_channels(WIDTH, device)
        self.amp = cfg.amp == "bfloat16" and device.type == "cuda"
        self.log = 0

    def targets(self) -> None:
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        hard_update(self.D_target, self.D)

    def autocast(self) -> Any:
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp)

    def modules(self) -> list[nn.Module]:
        return [self.actor, self.actor_target, self.critic, self.critic_target]

    def play(
        self, canvas: torch.Tensor, gt: torch.Tensor, t: torch.Tensor, target: bool = False
    ) -> torch.Tensor:
        """Actor output ``(B, 65)`` for uint8 canvas / target and step index ``t`` ``(B,)``."""
        x = torch.cat(
            [
                canvas.float() / 255,
                gt.float() / 255,
                (t.float() / self.cfg.max_step).view(-1, 1, 1, 1).expand(-1, 1, WIDTH, WIDTH),
                self.coord.expand(canvas.shape[0], 2, WIDTH, WIDTH),
            ],
            1,
        )
        with self.autocast():
            out = (self.actor_target if target else self.actor)(x)
        return out.float()  # type: ignore[no-any-return]

    @torch.no_grad()
    def act(self, canvas: torch.Tensor, gt: torch.Tensor, t: int) -> torch.Tensor:
        """Deterministic action in evaluation mode (running BatchNorm statistics), no exploration noise."""
        self.actor.eval()
        out = self.play(canvas, gt, torch.full((canvas.shape[0],), t, device=self.device))
        self.actor.train()
        return out

    def evaluate(
        self,
        canvas0: torch.Tensor,
        gt: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(Q + r, r)`` for the action taken at step ``t``: ``r = D'(gt, canvas1) - D'(gt, canvas0)``, canvas1 painted by the renderer."""
        gt_f, c0 = gt.float() / 255, canvas0.float() / 255
        c1, _ = decode(action, c0, self.renderer, STROKES)
        reward = self.D_target(torch.cat([gt_f, c1], 1)) - self.D_target(torch.cat([gt_f, c0], 1))
        merged = torch.cat(
            [
                c0,
                c1,
                gt_f,
                ((t + 1).float() / self.cfg.max_step).view(-1, 1, 1, 1).expand(-1, 1, WIDTH, WIDTH),
                self.coord.expand(c0.shape[0], 2, WIDTH, WIDTH),
            ],
            1,
        )
        with self.autocast():
            q = (self.critic_target if target else self.critic)(merged)
        return q.float() + reward, reward

    def gan_step(self, canvas: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
        """One WGAN-GP discriminator step on ``(gt, canvas)`` (fake) against ``(gt, gt)`` (real), then the soft target update."""
        fake_c, real_c = canvas.float() / 255, gt.float() / 255
        fake, real = torch.cat([real_c, fake_c], 1), torch.cat([real_c, real_c], 1)
        d_real, d_fake = self.D(real), self.D(fake)
        alpha = torch.rand(real.shape[0], 1, 1, 1, device=self.device)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        grad = autograd.grad(self.D(inter).sum(), inter, create_graph=True)[0]
        penalty = ((grad.flatten(1).norm(2, dim=1) - 1) ** 2).mean() * self.cfg.gan_lambda
        cost = d_fake.mean() - d_real.mean() + penalty
        self.D_opt.zero_grad(set_to_none=True)
        cost.backward()
        self.D_opt.step()
        soft_update(self.D_target, self.D, self.cfg.gan_tau)
        return {
            "gan_fake": float(d_fake.mean()),
            "gan_real": float(d_real.mean()),
            "gan_penalty": float(penalty),
        }

    def update(
        self, replay: Replay, gen: torch.Generator, lr_critic: float, lr_actor: float
    ) -> dict[str, float]:
        """One update of the official ``update_policy``."""
        for g in self.critic_opt.param_groups:
            g["lr"] = lr_critic
        for g in self.actor_opt.param_groups:
            g["lr"] = lr_actor
        c0, c1, gt, t, action, done = replay.sample(self.cfg.batch_size, gen)
        stats = self.gan_step(c1, gt)
        with torch.no_grad():
            next_action = self.play(c1, gt, t + 1, target=True)
            target_q, _ = self.evaluate(c1, gt, t + 1, next_action, target=True)
            target_q = self.cfg.discount * (1 - done.float()).view(-1, 1) * target_q
        cur_q, step_reward = self.evaluate(c0, gt, t, action)
        target_q = target_q + step_reward.detach()
        value_loss = F.mse_loss(cur_q, target_q)
        self.critic.zero_grad(set_to_none=True)
        value_loss.backward()  # type: ignore[no-untyped-call]
        self.critic_opt.step()
        new_action = self.play(c0, gt, t)
        with frozen(self.critic):
            pre_q, _ = self.evaluate(c0, gt, t, new_action)
        policy_loss = -pre_q.mean()
        self.actor.zero_grad(set_to_none=True)
        policy_loss.backward()  # type: ignore[no-untyped-call]
        self.actor_opt.step()
        for target, source in ((self.actor_target, self.actor), (self.critic_target, self.critic)):
            soft_update(target, source, self.cfg.tau)
        self.log += 1
        return {
            "q": float(-policy_loss),
            "value_loss": float(value_loss),
            "reward": float(step_reward.mean()),
            **stats,
        }

    def state(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "D": self.D.state_dict(),
            "D_target": self.D_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "D_opt": self.D_opt.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        for name in ("actor", "actor_target", "critic", "critic_target", "D", "D_target"):
            getattr(self, name).load_state_dict(state[name])
        for name in ("actor_opt", "critic_opt", "D_opt"):
            getattr(self, name).load_state_dict(state[name])
        share_buffers(
            self.actor_target, self.actor
        )  # loading breaks the sharing of the BatchNorm buffers


@torch.no_grad()
def validate(agent: DDPG, env: PaintEnv, ids: np.ndarray, cfg: DDPGConfig) -> dict[str, float]:
    """Mean l2 of the final canvas (and of the blank canvas) over held-out images, batches of ``env_batch``, no flip."""
    final, blank = [], []
    for a in range(0, len(ids), cfg.env_batch):
        chunk = ids[a : a + cfg.env_batch]
        env.reset(chunk)
        blank.append(env.dist())
        for t in range(cfg.max_step):
            env.step(agent.act(env.canvas, env.gt, t), agent.renderer)
        final.append(env.dist())
    return {
        "val_l2": float(torch.cat(final).mean()),
        "val_l2_blank": float(torch.cat(blank).mean()),
    }


@torch.no_grad()
def evaluate_actor(
    actor: nn.Module,
    renderer: nn.Module,
    images: np.ndarray,
    ids: np.ndarray,
    max_step: int = 40,
    batch: int = 96,
    bgr: bool = True,
) -> dict[str, float]:
    """Mean l2 (per-pixel MSE in [0, 1]) of the canvas after ``max_step`` steps (5 strokes each) of an eval-mode ``actor`` on 128 x 128 images.

    The canvas is re-quantized to 8 bits after each step as in the training environment; ``bgr`` is the channel order the actor expects
    (the released actor and ours are trained on BGR). Returns the mean l2 after 1, 5, 20 steps and at the end, and of the blank canvas.
    """
    dev = next(actor.parameters()).device
    env = PaintEnv(images, batch, max_step, dev, bgr)
    coord = coord_channels(WIDTH, dev)
    marks = {1, 5, 20, max_step}
    sums = {f"l2_step{m}": 0.0 for m in sorted(marks)}
    blank = 0.0
    for a in range(0, len(ids), batch):
        env.reset(ids[a : a + batch])
        n = env.gt.shape[0]
        blank += float(env.dist().sum())
        for t in range(max_step):
            x = torch.cat(
                [
                    env.canvas.float() / 255,
                    env.gt.float() / 255,
                    torch.full((n, 1, WIDTH, WIDTH), t / max_step, device=dev),
                    coord.expand(n, 2, WIDTH, WIDTH),
                ],
                1,
            )
            env.step(actor(x), renderer)
            if t + 1 in marks:
                sums[f"l2_step{t + 1}"] += float(env.dist().sum())
    out = {k: v / len(ids) for k, v in sums.items()}
    out["l2_blank"] = blank / len(ids)
    return out


def save_actor(agent: DDPG, path: Path) -> None:
    """The actor state dict in the format of the released ``actor.pkl`` (loadable by :func:`~.actor.load_actor`)."""
    torch.save({k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}, path)


def train_agent(cfg: DDPGConfig, log: bool = True) -> Path:
    """Train (or resume) the agent; returns the path of the final actor weights."""
    from brushstroke.utils.device import get_device
    from brushstroke.utils.seed import seed_everything

    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = get_device(cfg.device)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg.trainset)
    images = np.load(root / "images.npy", mmap_mode="r")
    is_val = np.load(root / "is_val.npy")
    train_ids, val_ids = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    renderer = load_renderer(cfg.renderer, device)
    agent = DDPG(cfg, renderer, device)
    env = PaintEnv(images, cfg.env_batch, cfg.max_step, device, cfg.bgr)
    val_env = PaintEnv(images, cfg.env_batch, cfg.max_step, device, cfg.bgr)
    replay = Replay(cfg.rmsize * cfg.max_step, cfg.env_batch, cfg.max_step, device)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    episode, refill_until = 0, 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=True)
        agent.load_state(ck["agent"])
        episode = int(ck["episode"])
        rng.bit_generator.state = ck["numpy_rng"]
        gen.manual_seed(cfg.seed * 7919 + episode)
        refill_until = (
            episode + -(-cfg.warmup // cfg.max_step) + 1
        )  # the replay is not saved: refilled before updating
    (out / "config.json").write_text(json.dumps(asdict(cfg), default=str), encoding="utf-8")
    history = out / "history.jsonl"
    t0 = time.time()
    acc: dict[str, float] = {}
    n_acc = 0
    stop = (
        cfg.episodes
        if not cfg.max_episodes_this_run
        else min(cfg.episodes, episode + cfg.max_episodes_this_run)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    while episode < stop:
        ids = rng.choice(train_ids, cfg.env_batch)
        env.reset(ids, rng.random(cfg.env_batch) < 0.5)
        for t in range(cfg.max_step):
            action = agent.act(env.canvas, env.gt, t)
            env.step(action, renderer)
            replay.add(env.canvas, action, env.gt)
        episode += 1
        train_l2 = float(env.dist().mean())
        if replay.g > cfg.warmup and episode > refill_until:
            s = stage(episode, cfg)
            for _ in range(cfg.episode_train_times):
                r = agent.update(replay, gen, cfg.critic_lrs[s], cfg.actor_lrs[s])
                for k, v in r.items():
                    acc[k] = acc.get(k, 0.0) + v
                n_acc += 1
        if episode % cfg.log_every == 0 or episode == stop:
            rec: dict[str, Any] = {
                "episode": episode,
                "env_steps": episode * cfg.max_step,
                "seconds": time.time() - t0,
                "train_l2": train_l2,
                "lr_critic": cfg.critic_lrs[stage(episode, cfg)],
                "lr_actor": cfg.actor_lrs[stage(episode, cfg)],
                **{k: v / max(n_acc, 1) for k, v in acc.items()},
            }
            if device.type == "cuda":
                rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
            if episode % cfg.validate_every == 0 or episode == stop:
                agent.actor.eval()
                rec.update(validate(agent, val_env, val_ids[: cfg.validate_images], cfg))
                agent.actor.train()
            acc, n_acc = {}, 0
            with history.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if log:
                print(rec, flush=True)
        if episode % cfg.checkpoint_every == 0 or episode == stop:
            torch.save(
                {"agent": agent.state(), "episode": episode, "numpy_rng": rng.bit_generator.state},
                last,
            )
            save_actor(agent, out / "actor.pt")
    if episode >= cfg.episodes:
        agent.actor.eval()
        final = validate(agent, val_env, val_ids, cfg)
        (out / "final_validation.json").write_text(
            json.dumps({"episode": episode, **final, "images": len(val_ids)}), encoding="utf-8"
        )
        agent.actor.train()
    return out / "actor.pt"
