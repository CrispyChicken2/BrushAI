"""Paint Transformer stroke predictor (Liu et al., ICCV 2021), architecture of the official code.

Reimplemented from the specification in ``docs/design/paint_transformer.md`` section 3 (paper
Section 3.3 and the
official ``inference/network.py:Painter`` / ``train/models/networks.py:Painter`` at commit
7960b8e). The module and
parameter names are those of the official network, so that the released ``state_dict`` loads
directly.

Input: a target patch and the current canvas patch, both ``(B, 3, 32, 32)`` in [0, 1]. Output: for
each of the
``N = 8`` stroke queries, 5 raw shape parameters ``(x, y, w, h, theta)`` (no clamping or sigmoid:
they are used as
normalized values, as in the official code) and one confidence logit. The colour of a stroke is
**read from the
target** at the predicted centre (:meth:`PaintTransformer.sample_colour`), as in the official code,
unless the
network is built with ``colour_mode="predicted"`` (an ablation of the paper description), in which
case the head
outputs 8 values and the last three are the colour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

COLOUR_MODES = ("sampled", "predicted")
CONVENTIONS = ("raw", "positive")


@dataclass
class PTModelConfig:
    """``model:`` config section (defaults = the official ``Painter(5, 8, 256, 8, 3, 3)``)."""

    strokes: int = 8  # N stroke queries per patch
    hidden_dim: int = 256
    heads: int = 8
    enc_layers: int = 3
    dec_layers: int = 3
    dropout: float = 0.1  # torch.nn.Transformer default, as in the official code
    patch_size: int = 32  # P (input size); the feature map is P / 4
    colour_mode: str = (
        "sampled"  # sampled (official code) | predicted (paper description, +3 outputs)
    )
    # raw (official: w, h, theta as predicted) | positive (T-019: w, h > 0, theta in [0, 1))
    convention: str = "raw"


class SignWithSigmoidGrad(torch.autograd.Function):
    """Binary neuron of the paper (Eq. 6): forward ``x > 0``, backward the derivative of the
    sigmoid."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(torch.sigmoid(x))
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (sig,) = ctx.saved_tensors
        return torch.as_tensor(grad_output * sig * (1 - sig))


def _block(c_in: int, c_out: int, stride: int) -> list[nn.Module]:
    return [
        nn.ReflectionPad2d(1),
        nn.Conv2d(c_in, c_out, 3, stride),
        nn.BatchNorm2d(c_out),
        nn.ReLU(True),
    ]


def _encoder() -> nn.Sequential:
    """Three Conv-BatchNorm-ReLU blocks with two stride-2 downsamplings: (3, P, P) -> (128, P/4,
    P/4)."""
    return nn.Sequential(*_block(3, 32, 1), *_block(32, 64, 2), *_block(64, 128, 2))


class PaintTransformer(nn.Module):
    """Two CNN encoders + ``nn.Transformer`` (DETR-style set prediction) + parameter and confidence
    heads."""

    def __init__(self, cfg: PTModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or PTModelConfig()
        if cfg.colour_mode not in COLOUR_MODES:
            raise ValueError(f"colour_mode must be one of {COLOUR_MODES}, got {cfg.colour_mode!r}")
        if cfg.convention not in CONVENTIONS:
            raise ValueError(f"convention must be one of {CONVENTIONS}, got {cfg.convention!r}")
        if cfg.patch_size % 4 != 0:
            raise ValueError("patch_size must be a multiple of 4")
        self.cfg = cfg
        hidden = cfg.hidden_dim
        feat = cfg.patch_size // 4
        params_per_stroke = 5 if cfg.colour_mode == "sampled" else 8
        self.enc_img = _encoder()
        self.enc_canvas = _encoder()
        self.conv = nn.Conv2d(128 * 2, hidden, 1)
        self.transformer = nn.Transformer(
            hidden, cfg.heads, cfg.enc_layers, cfg.dec_layers, dropout=cfg.dropout
        )
        self.linear_param = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, params_per_stroke),
        )
        self.linear_decider = nn.Linear(hidden, 1)
        self.query_pos = nn.Parameter(torch.rand(cfg.strokes, hidden))
        self.row_embed = nn.Parameter(torch.rand(feat, hidden // 2))
        self.col_embed = nn.Parameter(torch.rand(feat, hidden // 2))

    def forward(self, img: torch.Tensor, canvas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict strokes for a batch of patches.

        Args:
            img: ``(B, 3, P, P)`` target patches.
            canvas: ``(B, 3, P, P)`` current canvas patches.

        Returns:
            ``param`` ``(B, N, 5)`` (or ``(B, N, 8)`` with ``colour_mode="predicted"``) raw stroke
            parameters and
            ``decision`` ``(B, N, 1)`` confidence logits.
        """
        b = img.shape[0]
        feat = torch.cat([self.enc_img(img), self.enc_canvas(canvas)], dim=1)
        feat_conv = self.conv(feat)
        h, w = feat_conv.shape[-2:]
        pos_embed = torch.cat(
            [
                self.col_embed[:w].unsqueeze(0).repeat(h, 1, 1),
                self.row_embed[:h].unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        )
        # contiguous() calls as in the official forward: the memory layout changes the reduction
        # order of the attention kernels, hence the last bits of the outputs (T-015 parity)
        tokens = (
            pos_embed.flatten(0, 1).unsqueeze(1)
            + feat_conv.flatten(2).permute(2, 0, 1).contiguous()
        )
        queries = self.query_pos.unsqueeze(1).contiguous().repeat(1, b, 1)
        hidden = self.transformer(tokens, queries).permute(1, 0, 2).contiguous()
        return self.linear_param(hidden), self.linear_decider(hidden)

    @staticmethod
    def sample_colour(img: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """Colour of each stroke = target image bilinearly sampled at the predicted centre, ``(B,
        N, 3)``.

        Differentiable w.r.t. the centre (as in the official code, where it is part of the forward
        pass).
        """
        grid = (2 * shape[..., :2] - 1).unsqueeze(1)  # (B, 1, N, 2), x first
        sampled = F.grid_sample(img, grid, align_corners=False)  # (B, 3, 1, N)
        return sampled.squeeze(2).permute(0, 2, 1)

    def strokes(
        self, img: torch.Tensor, canvas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(shape (B, N, 5), colour (B, N, 3), decision logits (B, N))`` with the colour of the
        chosen mode."""
        param, decision = self(img, canvas)
        if self.cfg.convention == "positive":
            param = positive_convention(param)
        if self.cfg.colour_mode == "sampled":
            return param, self.sample_colour(img, param), decision.squeeze(-1)
        return param[..., :5], param[..., 5:8], decision.squeeze(-1)


def positive_convention(param: torch.Tensor) -> torch.Tensor:
    """Canonical parameterization of the strokes (option A of T-019): ``w = softplus(w_raw) > 0``,
    ``h = softplus(h_raw) > 0`` and ``theta`` wrapped into ``[0, 1)`` (``theta - floor(theta)``),
    the convention of the ground-truth strokes. ``x``, ``y`` and any further column are unchanged.

    Args:
        param: ``(..., >= 5)`` raw predictions ``x, y, w, h, theta, ...``.
    """
    out = param.clone()
    out[..., 2:4] = F.softplus(param[..., 2:4])
    out[..., 4] = param[..., 4] - torch.floor(param[..., 4])
    return out


def init_official_weights(net: nn.Module, gain: float = 0.02) -> None:
    """The official initialization (``init_type='normal'``): Conv/Linear weights ``N(0, gain)``,
    zero bias,
    BatchNorm weight ``N(1, gain)`` and zero bias; attention input projections and embeddings keep
    PyTorch's
    defaults, as in the official ``init_weights``.
    """
    for module in net.modules():
        if isinstance(module, nn.Conv2d | nn.Linear):
            nn.init.normal_(module.weight, 0.0, gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, gain)
            nn.init.constant_(module.bias, 0.0)


def count_parameters(net: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def sign_with_sigmoid_grad(x: torch.Tensor) -> torch.Tensor:
    """Typed entry point of :class:`SignWithSigmoidGrad`."""
    return torch.as_tensor(SignWithSigmoidGrad.apply(x))  # type: ignore[no-untyped-call]
