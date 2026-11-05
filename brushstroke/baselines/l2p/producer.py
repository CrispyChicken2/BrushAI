"""Evaluation-harness adapter of Learning to Paint (skeleton of T-028; the trained agent comes in T-029/T-030).

The producer hands the harness canvases after exactly ``n`` strokes (``docs/design/evaluation.md`` section 7): the strokes are
13-number Bezier strokes painted by the neural renderer, so the canvases come from :func:`~.rollout.progressive_canvases`, not from
a :class:`~brushstroke.strokes.StrokeSequence`. The blank canvas is black (the model is trained that way) and the stroke count is
that of every stroke of every patch (section 3.3). Replicates are training seeds (one actor checkpoint per seed). The producer is
not registered in ``configs/eval/protocol.yaml``: T-030 does that once an agent exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from brushstroke.baselines.l2p.actor import load_actor
from brushstroke.baselines.l2p.renderer import load_renderer
from brushstroke.baselines.l2p.rollout import RolloutConfig, progressive_canvases, rollout
from brushstroke.eval.producers import PaintResult


class L2PProducer:
    """Learning to Paint inference on an actor checkpoint (one per training seed) and a neural renderer."""

    name = "l2p"
    stochastic = False  # the actor is run in eval mode without noise
    replicate_kind = "training seed"
    dagger_metrics: tuple[
        str, ...
    ] = ()  # the reward is a learned discriminator, no metric of the protocol

    def __init__(
        self,
        actors: dict[int, str],
        renderer: str | Path,
        cfg: RolloutConfig | None = None,
        device: str | torch.device = "cuda",
        bgr: bool = True,
    ) -> None:
        self.actors = {int(k): str(v) for k, v in actors.items()}
        self.renderer_path = str(renderer)
        self.cfg = cfg or RolloutConfig()
        self.device = torch.device(device)
        self.bgr = bgr  # the official code feeds OpenCV (BGR) images to the actor
        self._actor: Any = None
        self._renderer: Any = None
        self._loaded: int | None = None
        self._strokes: torch.Tensor | None = None

    def load(self, seed: int) -> None:
        if seed not in self.actors:
            raise KeyError(f"no Learning to Paint actor for the training seed {seed}")
        if self._loaded != seed:
            self._actor = load_actor(self.actors[seed], self.device)
            self._renderer = load_renderer(self.renderer_path, self.device)
            self._loaded = seed

    def paint(self, target: torch.Tensor, checkpoints: Sequence[int], seed: int) -> PaintResult:
        x = target.flip(0) if self.bgr else target
        painting = rollout(self._actor, self._renderer, x, self.cfg)
        cps, canvases = progressive_canvases(painting, checkpoints)
        if self.bgr:
            canvases = canvases.flip(1)
        self._strokes = painting.strokes.cpu()
        return PaintResult(
            checkpoints=cps,
            canvases=canvases.to(target.device),
            n_total=painting.n_total,
            background=(0.0, 0.0, 0.0),
            meta={
                "actor": self.actors[seed],
                "renderer": self.renderer_path,
                "seed": seed,
                "replicate_kind": self.replicate_kind,
                "native_resolution": [128, 128],
                "upsampled": True,  # a 128 x 128 renderer: divided into patches, then resized (section 7)
                "divide": self.cfg.divide,
                "max_step": self.cfg.max_step,
                "channel_order": "bgr" if self.bgr else "rgb",
            },
        )

    def control(self) -> dict[str, Any]:
        return {}
