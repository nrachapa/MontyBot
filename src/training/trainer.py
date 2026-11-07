from typing import Any

import numpy as np
import torch

from ..config import CONFIG
from .system import Buffer, Factory


class Trainer:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.device = torch.device(config.device)
        self.network = Factory.create_network("alphazero", config)
        self.game = Factory.create_game("chess")
        self.search = Factory.create_search("vectorized_mcts", self.game, network=self.network, config=config)
        self.selfplay = Factory.create_selfplay("parallel", self.game, self.search)
        self.training = Factory.create_training("alphazero", self.network, self.device)
        self.buffer = Buffer()

    def train_iteration(self, iteration: int) -> dict[str, Any]:
        traj = self.selfplay.generate_games(self.config.games_per_iteration)
        self.buffer.add_batch(traj)
        losses = []
        for _ in range(self.config.train_steps):
            batch = self.buffer.sample(self.config.batch_size)
            if not batch:
                break
            losses.append(self.training.train_step(batch))
        return {"loss": float(np.mean(losses)) if losses else 0.0}

    def save_checkpoint(self, filename: str = "model.pt"):
        checkpoint = {"model_state": self.network.state_dict(), "config": self.config.model_dump()}
        torch.save(checkpoint, filename)

    def train(self, iterations: int) -> None:
        print(f"Training for {iterations} iterations...")
        for i in range(iterations):
            res = self.train_iteration(i)
            print(f"Iteration {i}: Loss = {res['loss']:.4f}")
            if i % 100 == 0:
                self.save_checkpoint(f"checkpoint_{i}.pt")

        self.save_checkpoint("model.pt")
        print("Training complete. Model saved to model.pt")


if __name__ == "__main__":  # pragma: no cover
    Trainer().train(5)
