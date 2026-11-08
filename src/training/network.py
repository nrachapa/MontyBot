import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import Network, NetworkContext


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
    def __init__(
        self,
        filters: int = 128,
        blocks: int = 4,
        input_planes: int = 8,
        action_size: int = 133,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_planes, filters, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks)])
        self.head_p = nn.Sequential(
            nn.Conv2d(filters, 2, 1), nn.Flatten(), nn.Linear(2 * 8 * 8, action_size)
        )
        self.head_v = nn.Sequential(nn.Conv2d(filters, 1, 1), nn.Flatten(), nn.Linear(1 * 8 * 8, 1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: NetworkContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — context (Elo embeddings, history) ignored by ResNet."""
        z = F.relu(self.conv(x))
        z = self.blocks(z)
        logits = self.head_p(z)
        value = torch.tanh(self.head_v(z))
        return logits, value
