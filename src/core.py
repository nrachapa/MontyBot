from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field


# Configuration Classes
class BaseConfig(BaseModel):
    # Model Architecture
    filters: int = Field(gt=0)
    blocks: int = Field(gt=0)
    input_planes: int = Field(gt=0)
    action_size: int = Field(gt=0)

    # Training Hyperparameters
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    iterations: int = Field(gt=0)
    mixed_precision: bool = Field()
    simulations: int = Field(gt=0)
    games_per_iteration: int = Field(gt=0)
    train_steps: int = Field(gt=0)

    # Infrastructure/SageMaker
    default_instance_type: str = Field(min_length=1)
    framework_version: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    max_run_seconds: int = Field(gt=0)
    max_wait_seconds: int = Field(gt=0)
    poll_seconds: int = Field(gt=0)
    job_name: str = Field(min_length=1)
    spot_instances: bool = Field()

    # AWS Resources
    region: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    role_arn: str = Field(min_length=1)
    bucket: str = Field(min_length=1)
    prefix: str = Field(min_length=1)

    # Evaluation
    eval_games: int = Field(gt=0)
    eval_threshold: float = Field(gt=0, le=1.0)

    # Runtime
    device: str = Field(default="auto", min_length=1)


# Training Abstract Classes
class Network(ABC):
    """Policy-Value network interface (NCHW -> (policy_logits, value))."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class SearchStrategy(ABC):
    @abstractmethod
    @abstractmethod
    def search(self, state: Any, root: Any = None) -> tuple[np.ndarray, Any]: ...


class SelfPlayStrategy(ABC):
    @abstractmethod
    def generate_games(self, n_games: int) -> list[Any]: ...


class TrainingStrategy(ABC):
    @abstractmethod
    def train_step(self, batch: list[Any]) -> float: ...


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, state: Any) -> float: ...


class Game(ABC):
    @abstractmethod
    def get_initial_state(self) -> Any: ...
    @abstractmethod
    def get_legal_moves(self, state: Any) -> list[int]: ...
    @abstractmethod
    def apply_move(self, state: Any, move_idx: int) -> Any: ...
    @abstractmethod
    def is_terminal(self, state: Any) -> bool: ...
    @abstractmethod
    def get_winner(self, state: Any) -> int: ...


# Infrastructure Abstract Classes
class S3ManagerInterface(ABC):
    @abstractmethod
    def upload_checkpoint(self, checkpoint: dict, iteration: int): ...
    @abstractmethod
    def ensure_bucket_exists(self): ...


class SageMakerManagerInterface(ABC):
    @abstractmethod
    def launch_job(self) -> str: ...
    @abstractmethod
    def wait_for_job(self, job_name: str, poll_seconds: int) -> str: ...
