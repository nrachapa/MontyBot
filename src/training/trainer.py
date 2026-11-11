import os
from typing import Any

import numpy as np
import torch

from ..config import CONFIG
from .replay import SparseReplayBuffer
from .system import Factory, ProgressiveModelTransition

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

        # Distributed setup (paper Sec. 5.1): init process group before DDP
        self._setup_ddp(config)

        self.network = Factory.create_network("alphazero", config)

        # Wrap with DDP if enabled (paper Sec. 5.1)
        if config.ddp_enabled:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.network = torch.nn.parallel.DistributedDataParallel(
                self.network,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )

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
        # Progressive model transition (paper Sec. 9.1)
        self.progressive_model = ProgressiveModelTransition(config)

        # Pre-fill buffer so training can start immediately.
        self._start_initial_generation()

    def _setup_ddp(self, config):
        """Initialize distributed process group when DDP is enabled.

        Graceful when MASTER_ADDR/RANK/WORLD_SIZE env vars are not set
        (single-GPU or non-distributed mode).
        """
        import os
        if not config.ddp_enabled:
            return

        try:
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
            os.environ["RANK"] = os.environ.get("RANK", "0")
            os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
            os.environ["LOCAL_RANK"] = os.environ.get("LOCAL_RANK", "0")

            backend = config.ddp_backend if hasattr(config, "ddp_backend") else "nccl"
            torch.distributed.init_process_group(backend=backend)
        except (RuntimeError, KeyError, ValueError):
            # Single-GPU: gracefully fall back
            torch.distributed = None  # noqa: F811

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
            # Check for model transition (paper Sec. 9.1)
            if self.progressive_model.should_transition(i):
                self._transition_to_larger_model(i)

            res = self.train_iteration(i)
            print(f"Iteration {i}: Loss = {res['loss']:.4f}")
            if i % 100 == 0:
                self.save_checkpoint(f"checkpoint_{i}.pt")

        self.save_checkpoint("model.pt")
        print("Training complete. Model saved to model.pt")

    def _transition_to_larger_model(self, iteration: int) -> None:
        """Transition to larger network size (paper Sec. 9.1)."""
        print(f"\n=== Model Transition at iteration {iteration} ===")
        print(f"Transitioning from {self.progressive_model.current_filters}x{self.progressive_model.current_blocks} "
              f"to {self.config.filters}x{self.config.blocks}")

        # Create larger network
        larger_net = self.progressive_model.create_larger_network(self.network)

        # Create new training components with larger network
        self.network = larger_net
        self.search = Factory.create_search(
            "vectorized_mcts", self.game, network=self.network, config=self.config
        )
        self.training = Factory.create_training("alphazero", self.network, self.device)
        self.progressive_model.transitioned = True

        print("Model transition complete. Larger network now active.")


if __name__ == "__main__":  # pragma: no cover
    Trainer().train(5)
