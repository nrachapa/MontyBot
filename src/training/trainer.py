from typing import Any

import numpy as np
import torch

from ..config import CONFIG
from .replay import SparseReplayBuffer
from .system import Factory

# Lazy W&B import (optional dependency)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class Trainer:
    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        self.device = torch.device(config.device)
        self.network = Factory.create_network("alphazero", config)
        self.game = Factory.create_game("chess")
        self.search = Factory.create_search("vectorized_mcts", self.game, network=self.network, config=config)
        # Actor overlap (paper Sec. 5.5): n_async_games > 0 starts self-play
        # in a background thread so training doesn't wait for game generation.
        self.selfplay = Factory.create_selfplay(
            "parallel", self.game, self.search,
            n_async_games=1, capped_search_budget=400,
        )
        self.training = Factory.create_training("alphazero", self.network, self.device)
        # Sparse replay buffer (paper Sec. 7)
        self.buffer = SparseReplayBuffer(
            capacity=config.c_capacity if hasattr(config, 'c_capacity') else 2_000_000
        )
        # W&B experiment tracking (paper Sec. 5.1 - monitoring)
        self.wandb_run = None
        if HAS_WANDB:
            self.wandb_run = wandb.init(
                project=config.wandb_project if hasattr(config, 'wandb_project') else "montybot",
                group=config.wandb_group if hasattr(config, 'wandb_group') else None,
                config=config.model_dump(),
            )
        # Pre-fill buffer so training can start immediately.
        self._start_initial_generation()

    def _start_initial_generation(self) -> None:
        """Kick off the first batch of games in the background."""
        self.selfplay._start_async_generation(self.config.games_per_iteration)

    def _get_next_batch(self):
        """Wait for the previous batch and start the next one."""
        trajs = self.selfplay._wait_for_generation()
        self.buffer.add_batch(trajs)
        # Start the next generation in the background.
        self.selfplay._start_async_generation(self.config.games_per_iteration)

    def train_iteration(self, iteration: int) -> dict[str, Any]:
        # Wait for self-play batch, then start next batch in background.
        self._get_next_batch()
        losses = []
        for _ in range(self.config.train_steps):
            batch = self.buffer.sample(self.config.batch_size)
            if not batch:
                break
            losses.append(self.training.train_step(batch))

        metrics = {"loss": float(np.mean(losses)) if losses else 0.0}

        # Log to W&B (paper Sec. 5.1 - monitoring)
        if self.wandb_run and losses:
            self.wandb_run.log({
                "loss": metrics["loss"],
                "learning_rate": self.config.learning_rate,
                "selfplay_throughput": self.config.games_per_iteration,
                "buffer_size": self.buffer.size(),
                "step": iteration,
            })

        return metrics

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
