import os
from datetime import UTC, datetime

import torch
from pydantic import Field, field_validator

from .core import BaseConfig

# time should be in this format: 2025-11-23-16-38-58-988 in UTC
utc_time = datetime.now(UTC).strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

# Single AWS-region knob (read once at import, like the credential env vars
# below); the S3 bucket name derives from it (`montybot-training-{region}`).
DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-2")


class MontyBotConfig(BaseConfig):
    """Production configuration"""

    # Model Architecture
    filters: int = Field(default=128, gt=0)
    blocks: int = Field(default=4, gt=0)
    input_planes: int = Field(default=8, gt=0)
    action_size: int = Field(default=133, gt=0)

    # Progressive model sizing (paper Sec. 9.1): start small, grow as training progresses
    initial_filters: int = Field(default=32, gt=0)
    initial_blocks: int = Field(default=2, gt=0)
    size_transition_iteration: int = Field(default=1000, ge=0)  # iteration to switch to full model

    # Training Hyperparameters
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=64, gt=0)
    iterations: int = Field(default=1000, gt=0)
    mixed_precision: bool = True
    simulations: int = Field(default=10, gt=0)
    net_eval_batch: int = Field(default=32, gt=0)
    c_discount: float = Field(default=0.01, ge=0.0, lt=0.5)  # AlphaZero: value target = (1-2c)v + 2c*result
    games_per_iteration: int = Field(default=128, gt=0)
    train_steps: int = Field(default=2, gt=0)

    # Infrastructure/SageMaker
    default_instance_type: str = "ml.g4dn.xlarge"
    framework_version: str = "2.14"
    python_version: str = "py312"
    max_run_seconds: int = Field(default=10800, gt=0)
    max_wait_seconds: int = Field(default=14400, gt=0)
    poll_seconds: int = Field(default=20, gt=0)
    job_name: str = Field(default=f"montybot-gpu-training-{utc_time}", min_length=1)
    spot_instances: bool = True

    # AWS Resources
    # Region is the single AWS-region knob (from the environment); the S3
    # bucket name derives from it (`montybot-training-{region}`).
    region: str = Field(default=DEFAULT_REGION, min_length=1)
    # No hardcoded credentials (ADR-0004 local-first): read from the
    # environment. None = unset -> SageMaker falls back to its default
    # execution role; the cloud path is an explicit `--cloud` opt-in.
    account_id: str | None = Field(default=os.environ.get("AWS_ACCOUNT_ID"))
    role_arn: str | None = Field(default=os.environ.get("AWS_ROLE_ARN"))
    bucket: str = Field(default=f"montybot-training-{DEFAULT_REGION}", min_length=1)
    prefix: str = Field(default=f"montybot-gpu-training-{utc_time}", min_length=1)

    # Evaluation
    eval_games: int = Field(default=20, gt=0)
    eval_threshold: float = Field(default=0.9, gt=0, le=1.0)

    # TensorRT + INT8 quantization (paper Sec. 9.2)
    quantization_aware_training: bool = False  # QAT simulates INT8 rounding/clipping
    tensorrt_engine_path: str = Field(default="")  # path to compiled TensorRT engine

    # Runtime
    # 'auto' prefers MPS on Apple Silicon (the M6 box is the default hardware,
    # ADR-0010) for the training steps. Generation runs where the flat
    # vectorized MCTS (ticket 10) is fastest: the ADR-0007 M6 measurement put
    # flat-MPS first (2.9x over unbatched-CPU), so set gen_device='mps' for
    # local runs; the default stays 'cpu' (cloud containers have no MPS) and
    # 'cuda' for cloud jobs.
    device: str = Field(default="auto", validate_default=True)
    gen_device: str = Field(default="cpu", validate_default=True)

    # Restart states (paper Sec. 6.3): 50% of games start from archived positions
    # with interesting alternatives left to explore
    use_restart_states: bool = True
    restart_state_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    restart_position_history: list = Field(default_factory=list)  # cached positions for restarts

    # Sparse replay storage (paper Sec. 7): store policies sparsely
    sparse_replay: bool = True  # only store actions with visit mass

    # Distributed training (paper Sec. 5.1): multi-GPU DDP support
    ddp_enabled: bool = False  # enable PyTorch DDP distributed training
    ddp_backend: str = "nccl"  # distributed backend: "nccl", "gloo", "mpi"

    # CUDA acceleration (paper Sec. 9): GPU-accelerated move gen and MCTS
    cuda_acceleration: bool = False  # use CUDA for move generation and neural eval

    # SageMaker multi-GPU config
    on_demand: bool = False  # use on-demand instead of spot instances
    instance_gpu_count: int = Field(default=1, ge=1, le=8)
    on_demand_instance_type: str = "ml.g5.4xlarge"  # on-demand GPU instance type

    @field_validator("device", "gen_device")
    @classmethod
    def _resolve_device(cls, v):
        if v == "auto":
            return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        return v


def resolve_device(v: str) -> str:
    """Resolve an 'auto' device string to the available hardware.

  Reads 'auto' literally; anything else is returned unchanged.  Used by
  modules that need the resolved device without pulling the full CONFIG.
    """
    if v == "auto":
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    return v


def detect_cuda_compatible() -> bool:
    """Check if CUDA is available and compute capability >= 6.0 (Volta or newer)."""
    if not torch.cuda.is_available():
        return False
    cc = torch.cuda.get_device_properties(0).major
    return not cc < 6


def resolve_batch_size(device: str) -> int:
    """Scale batch size based on GPU memory (only relevant for CUDA)."""
    if device != "cuda" or not torch.cuda.is_available():
        return 64  # default from config
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_memory < 8:
        return 16
    if gpu_memory > 16:
        return 64
    return 32  # medium GPUs


# Global configuration instance
CONFIG = MontyBotConfig(device="auto")
