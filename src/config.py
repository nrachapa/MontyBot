import os
from datetime import datetime, timezone

import torch
from pydantic import Field, field_validator

from .core import BaseConfig

# time should be in this format: 2025-11-23-16-38-58-988 in UTC
utc_time = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]


class MontyBotConfig(BaseConfig):
    """Production configuration"""

    # Model Architecture
    filters: int = Field(default=128, gt=0)
    blocks: int = Field(default=4, gt=0)
    input_planes: int = Field(default=8, gt=0)
    action_size: int = Field(default=133, gt=0)

    # Training Hyperparameters
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=64, gt=0)
    iterations: int = Field(default=1000, gt=0)
    mixed_precision: bool = True
    simulations: int = Field(default=10, gt=0)
    c_discount: float = Field(default=0.01, ge=0.0, lt=0.5)  # AlphaZero: value target = (1-2c)v + 2c*result
    games_per_iteration: int = Field(default=128, gt=0)
    train_steps: int = Field(default=2, gt=0)

    # Infrastructure/SageMaker
    default_instance_type: str = "ml.g4dn.xlarge"
    framework_version: str = "2.2.0"
    python_version: str = "py310"
    max_run_seconds: int = Field(default=10800, gt=0)
    max_wait_seconds: int = Field(default=14400, gt=0)
    poll_seconds: int = Field(default=20, gt=0)
    job_name: str = Field(default=f"montybot-gpu-training-{utc_time}", min_length=1)
    spot_instances: bool = True

    # AWS Resources
    region: str = Field(default="us-west-2", min_length=1)
    # No hardcoded credentials (ADR-0004 local-first): read from the
    # environment. None = unset → SageMaker falls back to its default
    # execution role; the cloud path is an explicit `--cloud` opt-in.
    account_id: str | None = Field(default=os.environ.get("AWS_ACCOUNT_ID"))
    role_arn: str | None = Field(default=os.environ.get("AWS_ROLE_ARN"))
    bucket: str = Field(default="montybot-training", min_length=1)
    prefix: str = Field(default=f"montybot-gpu-training-{utc_time}", min_length=1)

    # Evaluation
    eval_games: int = Field(default=20, gt=0)
    eval_threshold: float = Field(default=0.9, gt=0, le=1.0)

    # Runtime
    # 'auto' prefers MPS on Apple Silicon (the M6 box is the default hardware,
    # ADR-0010) for the training steps. The generation device is separate:
    # unbatched MCTS is ~1.7x slower on MPS than CPU (ADR-0007), so self-play
    # stays on CPU until ticket 10's vectorized loop.
    device: str = Field(default="auto", validate_default=True)
    gen_device: str = Field(default="cpu", validate_default=True)

    @field_validator("device", "gen_device")
    @classmethod
    def resolve_device(cls, v):
        if v == "auto":
            return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        return v

    def model_post_init(self, __context):
        """Handle GPU scaling after validation"""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:
                object.__setattr__(self, "batch_size", 16)
            elif gpu_memory > 16:
                object.__setattr__(self, "batch_size", 64)


# Global configuration instance
CONFIG = MontyBotConfig(device="auto")
