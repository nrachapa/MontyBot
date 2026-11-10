from typing import Any

import numpy as np
import torch

from ..config import CONFIG
from .game import ChessGame
from .network import AlphaZeroNet
from .search import ParallelSelfPlay, VectorizedMCTS
from .system import Buffer
from .training import AlphaZeroTraining


class Trainer:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.device = torch.device(config.device)
        self.network = AlphaZeroNet(
            filters=config.filters,
            blocks=config.blocks,
            input_planes=config.input_planes,
            action_size=config.action_size,
        )
        self.game = ChessGame()
        self.search = VectorizedMCTS(
            self.game,
            self.network,
            simulations=config.simulations,
            batch=config.net_eval_batch,
            gen_device=config.gen_device,
        )
        self.selfplay = ParallelSelfPlay(
            self.game, self.search, config.gen_device, config.batch_size, config.c_discount
        )
        self.training = AlphaZeroTraining(
            self.network, self.device,
            config.learning_rate, config.mixed_precision
        )
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
