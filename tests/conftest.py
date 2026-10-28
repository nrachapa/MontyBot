import os

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _env_fast_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTEST_RUNNING"] = "1"  # disable multiproc paths
    np.random.seed(42)
    torch.manual_seed(42)

    # Hermetic tests: 'auto' resolves to MPS on Apple Silicon (ADR-0010) and
    # would push real networks off CPU; tests pin the resolved training device.
    from src.config import CONFIG

    original = CONFIG.device
    CONFIG.device = "cpu"
    yield
    CONFIG.device = original
