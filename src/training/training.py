import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core import TrainingStrategy
from .network import AlphaZeroNet


class AlphaZeroTraining(TrainingStrategy):
    def __init__(self, network: AlphaZeroNet, device: torch.device, learning_rate: float, mixed_precision: bool):
        self.net = network.to(device).train()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.device = device
        # Mixed precision (ticket 09): autocast works on CUDA and MPS
        # (the MPS crash was fixed in torch 2.5.0 — see ultralytics #25494).
        if mixed_precision:
            if self.device.type == "cuda":
                self.autocast = torch.autocast(device_type="cuda")
            else:
                self.autocast = torch.autocast(device_type="mps")
        else:
            self.autocast = contextlib.nullcontext()

    def train_step(self, batch: list[Any]) -> float:
        states, policies, values = zip(*batch)
        x = torch.from_numpy(np.stack([s.tensor for s in states]).astype(np.float32)).to(self.device)
        y_p = torch.from_numpy(np.stack(policies).astype(np.float32)).to(self.device)
        y_v = torch.tensor(values, dtype=torch.float32, device=self.device)

        with self.autocast:
            logits, v = self.net(x)
            policy_loss = torch.mean(torch.sum(torch.log_softmax(logits, dim=1) * y_p, dim=1) * (-1.0))
            value_loss = F.mse_loss(v.squeeze(-1), y_v)
            loss = policy_loss + value_loss

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
        self.opt.step()

        return float(loss.detach().cpu().item())
