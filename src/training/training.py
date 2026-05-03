from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..config import CONFIG
from ..core import TrainingStrategy
from .network import AlphaZeroNet


class AlphaZeroTraining(TrainingStrategy):
    def __init__(self, network: AlphaZeroNet, device: torch.device):
        self.net = network.to(device).train()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=CONFIG.learning_rate)
        self.device = device

    def train_step(self, batch: list[Any]) -> float:
        states, policies, values = zip(*batch)
        x = torch.from_numpy(np.stack([s[0] for s in states]).astype(np.float32)).to(self.device)
        y_p = torch.from_numpy(np.stack(policies).astype(np.float32)).to(self.device)
        y_v = torch.tensor(values, dtype=torch.float32, device=self.device)

        logits, v = self.net(x)
        policy_loss = torch.mean(torch.sum(torch.log_softmax(logits, dim=1) * y_p, dim=1) * (-1.0))
        value_loss = F.mse_loss(v.squeeze(-1), y_v)
        loss = policy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return float(loss.detach().cpu().item())
