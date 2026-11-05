"""The Learning to Paint actor: ResNet-18 with CoordConv input, 5 strokes x 13 numbers per step.

Copied from ``baseline/DRL/actor.py`` and ``baseline/DRL/ddpg.py`` (official code, commit 615c070, MIT) so that the released
``actor.pkl`` loads strictly: ``ResNet(9, 18, 65)``; the input has 9 channels ``(canvas 3, target 3, step / max_step 1,
coordinates 2)``, the 65 outputs (sigmoid, in [0, 1]) are 5 strokes of 13 numbers (10 shape numbers, 3 colour values).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class Actor(nn.Module):
    """``ResNet(num_inputs=9, depth=18, num_outputs=65)`` of the official code (BasicBlock, ``[2, 2, 2, 2]``)."""

    def __init__(self, num_inputs: int = 9, num_outputs: int = 65) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(num_inputs, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 2)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_outputs)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 9, 128, 128)`` -> ``(B, 65)`` in [0, 1]."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.avg_pool2d(x, 4).flatten(1)
        return torch.sigmoid(self.fc(x))


def coord_channels(size: int = 128, device: torch.device | str = "cpu") -> torch.Tensor:
    """CoordConv channels ``(1, 2, size, size)``: ``[i / (size - 1)]`` along rows then columns (official ``coord``)."""
    r = torch.arange(size, dtype=torch.float32, device=device) / (size - 1.0)
    return torch.stack([r[:, None].expand(size, size), r[None, :].expand(size, size)])[None]


def load_actor(path: str | Path, device: torch.device | str = "cpu") -> Actor:
    """Load an actor state dict (the released ``actor.pkl`` or ours) strictly, in eval mode."""
    net = Actor()
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()
