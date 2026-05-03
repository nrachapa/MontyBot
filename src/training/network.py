import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CONFIG
from ..core import Network


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class AlphaZeroNet(Network, nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(CONFIG.input_planes, CONFIG.filters, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(CONFIG.filters) for _ in range(CONFIG.blocks)])
        self.head_p = nn.Sequential(
            nn.Conv2d(CONFIG.filters, 2, 1), nn.Flatten(), nn.Linear(2 * 8 * 8, CONFIG.action_size)
        )
        self.head_v = nn.Sequential(nn.Conv2d(CONFIG.filters, 1, 1), nn.Flatten(), nn.Linear(1 * 8 * 8, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(self.conv(x))
        z = self.blocks(z)
        logits = self.head_p(z)
        value = torch.tanh(self.head_v(z))
        return logits, value
