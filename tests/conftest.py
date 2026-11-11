import os

import numpy as np
import pytest
import torch

# Capture the original CONFIG state once, before any tests mutate it.
_original_config = None


# Capture the original CONFIG state once, before any tests mutate it.
_original_config = None


@pytest.fixture(autouse=True)
def _env_fast_cpu():
    global _original_config

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTEST_RUNNING"] = "1"  # disable multiproc paths
    np.random.seed(42)
    torch.manual_seed(42)

    # Clear board encoding cache to prevent test pollution (cache grows with each test)
    from src.training import game as game_module
    if hasattr(game_module, '_ENCODE_CACHE'):
        game_module._ENCODE_CACHE.clear()

    if _original_config is None:
        from src.config import CONFIG
        _original_config = CONFIG.model_dump()

    # Hermetic tests: 'auto' resolves to MPS on Apple Silicon (ADR-0010) and
    # would push real networks off CPU; tests pin the resolved training device.
    from src.config import CONFIG
    CONFIG.device = "cpu"

    yield

    # Restore mutable config values to original state
    for k, v in _original_config.items():
        setattr(CONFIG, k, v)
